# Feature Spec: chutes-api — Miner Host Profile Submission

**Date**: 2026-08-20
**Status**: implemented

---

## Context

A sek8s VM only launches on a host class we have published TEE measurements for: `discover-profile.sh`
captures the host's shape, the miner-side tooling matches it against the baselined topology
fingerprints in `host-tools/scripts/chutes/guest/gpu/profiles.py`, and a host with no matching
measurement fails the launch. Until now an operator with new hardware had no in-band way to ask for
measurements — they had to reach the Chutes team out of band.

This adds the request channel: the miner CLI runs `discover-profile.sh`, signs the resulting JSON
with the miner hotkey, and POSTs it to `api.chutes.ai`. The API parks the document in object storage
keyed by a fingerprint of the host class. Offline tooling (CI/CD or local) reads the bucket,
generates measurements, ships them in the next release, and deletes the object.

- **Packages affected**: `chutes-api` (server router/schemas/util, config, rate limiting, chart)
- **Key files**:
  - `api/server/router.py` — `submit_host_profile` (`POST /servers/tdx/host_profiles`)
  - `api/server/schemas.py` — `HostProfile` (including `HostProfile.fingerprint`) and its blocks,
    `HostProfileSubmissionResponse`
  - `api/server/util.py` — `store_host_profile`
  - `api/rate_limit.py` — `rate_limit_miner` (auth + per-miner metering dependency), `check_rate_limit`
  - `api/constants.py` — size cap and submission limits
  - `api/config/__init__.py` — `host_profile_bucket`, `host_profile_prefix`
  - `charts/values.yaml`, `charts/templates/api-deployment.yaml` — `api.hostProfileBucket` / `api.hostProfilePrefix`
- **Dependencies**: existing S3/R2 client (`settings.s3_client()`), existing sr25519 request auth

---

## Design Decisions

- **On the servers router, under a `/tdx` prefix**: submission is a miner action, but what it acts
  on is server/attestation state, so it sits with `GET /servers/tee/measurements` — the endpoint
  that tells a caller which host classes are already known. The two halves of "is my hardware
  supported / please support my hardware" belong together.

  The prefix is `/tdx`, not `/tee`: everything here is TEE-only now, so `tee` distinguishes nothing,
  whereas the platform will need to distinguish Intel TDX from AMD SEV-SNP once SNP support lands.
  The existing `/tee/measurements` route keeps its path for now — it is public and has external
  consumers — so the two are briefly inconsistent; moving it to `/tdx/measurements` (with `/tee`
  kept as a deprecated alias) is a separate, breaking-ish change.

- **Body-covering signature**: the route uses `get_current_user(registered_to=settings.netuid)` with
  **no** `purpose`. `get_signing_message` only folds the body hash into the signed message when
  `purpose` is absent (a purpose short-circuits it to `hotkey:nonce:purpose`), so omitting it is what
  makes the signature cover the submitted document. Same convention as `POST /servers`. The miner
  signs `hotkey:nonce:sha256(body)`.

- **The body is the profile, not arguments to a call**: the submitted JSON is modeled as
  `HostProfile` — the document itself, which is also exactly what gets stored — rather than a
  `…Args` request wrapper around it, so `fingerprint` lives on the thing it identifies. The blocks
  are named for what they hold rather than for their wire keys, which `discover-profile.sh` chose
  for at-a-glance reading: `host` → `HostProfilePlatform` (DMI board/BIOS identity plus OS release),
  `launch_determinism` → `HostProfileQemu` (QEMU build and the `-cpu` string). Field aliases keep
  the wire format untouched.

- **One field, one home**: `launch_determinism` restates values that belong to other blocks —
  `numa_node_count` and `numa_topology_eligible` (the latter is just `node_count == 2`) duplicate
  `numa`, and `host_cpu_topology` duplicates `cpu`. Those are not modeled: they stay in the stored
  document as extras, but everything here reads them from their canonical block, so a fingerprint
  can never be computed from a stale copy that disagrees with the block it restates.

- **One bucket, one prefix**: profiles land in the main storage bucket under `host-profiles/`,
  beside `audit/`, `forge/`, `jobs/` and `logos/`, rather than in a bucket of their own. A single
  bucket keeps backup/restore and credential management uniform, and costs nothing extra: R2 bills
  storage per GB-month plus Class A/B operations with no per-bucket fee, and this data is a few KB
  per host class — one object per class, deduplicated by fingerprint. The generation job reads it
  with the same bucket-wide token the rest of the platform already uses.

  Two consequences worth knowing. R2 API tokens scope to buckets, not prefixes, so a *permanent*
  credential limited to `host-profiles/` is not possible — if that is ever wanted, mint a
  prefix-scoped temporary credential (short-lived, S3-compatible, derived from a bucket-scoped
  parent token). And R2's per-bucket metrics do not break out by prefix, so prefix-level volume has
  to come from listing it rather than from the dashboard. Lifecycle rules *do* take a prefix, which
  is what matters here: expiry for abandoned submissions is configured on `host-profiles/` alone.
  `HOST_PROFILE_BUCKET` remains available to move them to a dedicated bucket without a code change.

- **Bucket only, no database**: a submission is a transient request, not a record. Once measurements
  ship for the host class, the published measurement *is* the record and the object is deleted. No
  table, no migration, no cleanup job in the API.

- **Fingerprint keying, first-write-wins**: the object key is
  `{prefix}/{sha256(host class identity)}.json`, from `HostProfile.fingerprint` — a property on
  the model itself, so a profile and its identity are never computed apart, and the route, storage
  layer and any future consumer all read the same value. The identity covers only what changes the generated
  measurement — passthrough inventory (GPU ids/count/BARs, NVSwitches, IB NICs), CPU identity and
  topology, host RAM, NUMA layout, and the QEMU/`-cpu` launch arguments. Hostname, BIOS strings and
  the lspci tree are stored but excluded, so a 200-host fleet of one class produces one object rather
  than 200. A re-submission of a known fingerprint HEADs the key, returns `stored: false`, and writes
  nothing — repeat submissions cost the bucket nothing.

- **Exact signed bytes are stored**: the object body is the raw request body, and the hotkey, nonce,
  and signature ride along in object metadata. Whoever generates the measurement can re-verify who
  submitted the document and that it is unmodified.

- **One dependency for auth and metering**: the route declares
  `Depends(rate_limit_miner("host_profile_submit", ...))` and nothing else — that dependency verifies
  the signature, confirms the hotkey is registered and un-blacklisted, then meters it. Auth is a
  *sub*-dependency of it, not a sibling on the route, and that is what makes the ordering safe:
  FastAPI resolves sub-dependencies first, whereas siblings have no guaranteed order and a counter
  keyed on an unverified hotkey header would let anyone lock a miner out by spoofing it. The global
  ceiling (`global_limit`) is metered in the same place for the same reason — a pre-auth global
  counter can be run down with unauthenticated junk, locking out every miner at once.
  `check_rate_limit` is the shared counter underneath; the plain `rate_limit` dependency is
  unchanged for endpoints that only need an anonymous global cap.

- **No status endpoint**: a host class becomes usable when its measurement ships in a sek8s release,
  which requires a code change on that side anyway — the miner-side tooling already detects that. A
  pending/ready API would duplicate it and drift.

---

## Threat model

A submission is **untrusted, unattested, self-reported data**. No TDX quote backs it — a miner
describes hardware, and nothing proves they own it. Everything below follows from that.

**Reaching the endpoint** costs a hotkey registered on the subnet and not blacklisted, plus a valid
sr25519 signature over `hotkey:nonce:sha256(body)` with a nonce inside ±600s. Registration cost is
the sybil bound; the rate limits (10/hotkey/hour, 120/hour overall) are counted only after that
signature verifies, so an unauthenticated flood cannot spend anyone's quota.

**What is validated.** Modeled fields are bounded and pattern-checked (`api/constants.py` holds the
ranges): PCI device ids must be 4 hex chars, the processor id hex, `cpu_args` free of shell
metacharacters, DMI strings free of control characters, counts and sizes inside plausible-hardware
ranges. This matters because those values leave the request: `gpu.pci_device_ids` and `gpu.count`
are written into S3 object metadata — HTTP request headers — and into a log line. Without the
constraints, a CRLF in a device id is a log-injection primitive (the HTTP client rejects the header
itself, so R2 metadata injection fails closed as a 503, but loguru validates nothing), and
unbounded values are a reliable way to make the write fail. After validation, every metadata value
is provably constrained: the hotkey is a verified ss58, the nonce digits, the signature hex, the
fingerprint a SHA-256 digest, the device ids 4 hex chars each.

**What is not validated**, deliberately: unmodeled keys (`pci_topology`, `vbios`, and anything
discover-profile.sh grows later) are free-form. They are only ever *stored* — never used as a
storage key, header value, or log value — and the body is kept byte-for-byte so it stays
signature-verifiable. **The generation job is the trust boundary for those**: it must never
shell-interpolate a field, use one as a filesystem path, or evaluate one. `cpu_args` is constrained
precisely because it describes a command line that job may well construct.

**The storage key is derived, never supplied** — `{prefix}/{sha256}.json`, hex only, so no traversal
or key injection is reachable from the body.

**Fingerprint squatting is possible and accepted.** Because a miner controls the JSON, they can
submit a profile whose identity fields match a host class they do not own. First-write-wins means
that submission's *non-identity* fields (BIOS strings, VBIOS, lspci tree) are what the generation
job sees for that class. Object metadata records the submitting hotkey, so it is attributable, and
the real defense is downstream: a generated measurement is only correct if a real host attests
against it, and measurements ship deliberately in a release rather than automatically. Do not
change that.

**Cost is bounded** by the rate limits: 120 submissions/hour × 256 KiB ≈ 21 GB/month worst case
(~$0.32), with Class A operations inside the free tier, and the lifecycle rule reclaims abandoned
objects. Note that fingerprint dedup does *not* bound this, since the submitter controls the inputs
the fingerprint is computed from.

**Known limitation**: the 256 KiB cap is a policy check, not a memory guard. FastAPI parses the body
into the model, and the body-hashing middleware in `api/main.py` buffers it, before the handler's
size check runs — and the API ingress sets `proxy-body-size: 0`. This is platform-wide behavior for
every POST endpoint rather than anything specific to this one, but it means an oversized body is
read and parsed before it is rejected.

**Never serve this prefix publicly.** Objects are attacker-authored documents; they are stored with
`Content-Type: application/json` in a bucket with public access disabled, and should stay that way.

---

## Contract

### Request

```
POST https://api.chutes.ai/servers/tdx/host_profiles
X-Chutes-Hotkey:    <miner ss58>
X-Chutes-Nonce:     <unix seconds, within 600s of now>
X-Chutes-Signature: <sr25519 hex over "{hotkey}:{nonce}:{sha256(body)}">
Content-Type: application/json

<discover-profile.sh output, verbatim>
```

The hotkey must be registered on `settings.netuid` and not blacklisted (both enforced by
`rate_limit_miner`, before any metering). Bodies over 256 KiB are rejected with 413. Limits: 10
submissions per hotkey per hour, 120 across all miners per hour (429 past either).

Validation is deliberately loose: the blocks that feed the fingerprint (`gpu`, `cpu`, `memory`,
`numa`, `launch_determinism`) are required and typed, a profile reporting no GPUs is rejected, and
every block accepts unknown keys. `discover-profile.sh` can grow fields without a coordinated API
release.

### Response

```json
{
  "fingerprint": "<sha256 hex>",
  "stored": true,
  "detail": "Host profile accepted, measurements will be generated and published in a future release."
}
```

`stored: false` means that host class was already submitted — the correct outcome for the second
host in a fleet, not an error.

### Bucket layout (what the generation tooling consumes)

| | |
| --- | --- |
| Bucket | `STORAGE_BUCKET` (override with `HOST_PROFILE_BUCKET` to isolate them) |
| Key | `{HOST_PROFILE_PREFIX}/{fingerprint}.json` (default prefix `host-profiles`) |
| Body | the exact JSON the miner signed |
| Metadata | `hotkey`, `nonce`, `signature`, `fingerprint`, `gpu-count`, `gpu-pci-device-ids`, `submitted-at` |

The generation job lists the prefix, produces measurements for each object's host class, and deletes
the object once the measurement is published. Put a prefix-scoped lifecycle rule on
`host-profiles/` so abandoned submissions expire on their own — the API never deletes.
