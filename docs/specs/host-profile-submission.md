# Feature Spec: chutes-api — Miner Host Profile Submission

**Date**: 2026-08-20
**Status**: implemented

---

## Context

A sek8s VM only launches on a host class with published TEE measurements. `discover-profile.sh`
captures the host's shape, the miner-side tooling matches it against the baselined fingerprints in
`host-tools/scripts/chutes/guest/gpu/profiles.py`, and an unmatched host fails the launch. Operators
with new hardware had no in-band way to ask for measurements.

This adds the request channel: the miner CLI runs `discover-profile.sh`, signs the JSON with the
miner hotkey, and POSTs it. The API parks the document in object storage keyed by a fingerprint of
the host class; offline tooling (CI or local) reads the bucket, generates measurements, ships them
in a release, and deletes the object.

- **Key files**:
  - `api/server/router.py` — `submit_host_profile` (`POST /servers/tdx/host_profiles`)
  - `api/server/schemas.py` — `HostProfile` (+ `HostProfile.fingerprint`), `HostProfileSubmissionResponse`
  - `api/server/util.py` — `store_host_profile`
  - `api/rate_limit.py` — `rate_limit_miner` (auth + per-miner metering), `check_rate_limit`
  - `api/constants.py` — size cap, submission limits, field bounds
  - `api/config/__init__.py` — `host_profile_bucket`, `host_profile_prefix`
  - `charts/` — `api.hostProfileBucket` / `api.hostProfilePrefix`

---

## Design Decisions

- **On the servers router, under `/tdx`**: submission is a miner action but acts on
  server/attestation state, so it sits with `GET /servers/tee/measurements` — the endpoint saying
  which host classes are already known. `/tdx` rather than `/tee` because everything is TEE-only
  now, and AMD SEV-SNP will need distinguishing from Intel TDX. `/tee/measurements` keeps its path
  for now (public, external consumers), so the two are briefly inconsistent.

- **Body-covering signature**: the route uses `get_current_user(registered_to=settings.netuid)` with
  **no** `purpose`. `get_signing_message` only folds the body hash in when `purpose` is absent (a
  purpose short-circuits to `hotkey:nonce:purpose`), so omitting it is what covers the document.
  Same convention as `POST /servers`.

- **The body is the profile, not arguments**: modeled as `HostProfile` — the document itself, which
  is what gets stored — so `fingerprint` lives on the thing it identifies. Blocks are named for what
  they hold, not their wire keys: `host` → `HostProfilePlatform`, `launch_determinism` →
  `HostProfileQemu`, via field aliases that leave the wire format untouched.

- **A closed schema**: every field is modeled, bounded and pattern-checked; every block forbids
  extras. Accepting unknown keys would hand unvalidated attacker content to a privileged offline
  job, and would not work anyway — the load balancers validate against the published OpenAPI schema.

- **One field, one home**: `launch_determinism` restates `numa` (`numa_node_count`,
  `numa_topology_eligible`) and `cpu` (`host_cpu_topology`). Declared so submissions validate, but
  never read — `fingerprint` takes those from their canonical block, so a disagreeing copy is inert.

- **Bucket only, no database**: a submission is a transient request. Once measurements ship, the
  published measurement is the record and the object is deleted.

- **One bucket, one prefix**: profiles land in the main storage bucket under `host-profiles/`,
  beside `audit/`, `forge/`, `jobs/`, `logos/`. Single bucket keeps backup/restore and credentials
  uniform and costs nothing extra (R2 bills per byte and operation, no per-bucket fee). Two
  consequences: R2 API tokens scope to buckets not prefixes, so a permanent prefix-limited
  credential is impossible (use a temporary credential); and per-bucket metrics do not break out by
  prefix. Lifecycle rules *do* take a prefix, which is what matters. `HOST_PROFILE_BUCKET` moves
  them to a dedicated bucket without a code change.

- **Fingerprint keying, first-write-wins**: key is `{prefix}/{HostProfile.fingerprint}.json`,
  covering only what changes the generated measurement — passthrough inventory, CPU identity and
  topology, host RAM, NUMA layout, QEMU/`-cpu` args. Hostname, BIOS strings and the lspci tree are
  stored but excluded, so a 200-host fleet of one class produces one object. A known fingerprint
  HEADs the key, returns `stored: false`, and writes nothing.

- **Exact signed bytes are stored**, with hotkey/nonce/signature in object metadata, so the
  generation job can re-verify who submitted the document and that it is unmodified.

- **One dependency for auth and metering**: the route declares
  `Depends(rate_limit_miner("host_profile_submit", ...))` and nothing else. Auth is a
  *sub*-dependency, not a sibling: FastAPI resolves sub-dependencies first, whereas siblings have no
  guaranteed order and a counter keyed on an unverified hotkey would let anyone lock a miner out.
  The global ceiling is metered there too — a pre-auth global counter can be run down with
  unauthenticated junk.

- **No status endpoint**: a host class becomes usable when its measurement ships in a release, which
  needs a sek8s code change anyway, and the miner tooling already detects that.

---

## Threat model

A submission is **untrusted, unattested, self-reported data**. No TDX quote backs it — a miner
describes hardware, and nothing proves they own it.

- **Reaching the endpoint** costs a registered, un-blacklisted hotkey and a valid signature over
  `hotkey:nonce:sha256(body)` with a nonce inside ±600s. Registration cost is the sybil bound; rate
  limits (10/hotkey/hour, 120/hour overall) are counted only after the signature verifies.

- **Why the constraints matter**: `gpu.pci_device_ids` and `gpu.count` are written into S3 object
  metadata (HTTP headers) and a log line. Unconstrained, a CRLF in a device id is a log-injection
  primitive — the HTTP client rejects the header itself, so R2 metadata injection fails closed as a
  503, but loguru validates nothing. After validation every metadata value is provably constrained:
  verified ss58, digits, hex, SHA-256 digest, 4-hex device ids.

- **The loosest fields** are `pci_topology` (bounded; newlines and tabs allowed since the tree is
  drawn with them, other control characters rejected) and the DMI strings. Bounded is not the same
  as safe to interpolate: **the generation job must never shell-interpolate, path-join, or evaluate
  a field**. `cpu_args` and `host_cpu_topology` are restricted to a shell-safe charset because they
  describe a command line that job may construct.

- **The storage key is derived, never supplied** — hex only, so no traversal or key injection.

- **Fingerprint squatting is possible and accepted**: a miner can submit a profile whose identity
  fields match a host class they do not own, and first-write-wins means their non-identity fields
  are what the generation job sees. Metadata records the hotkey, so it is attributable; the real
  defense is downstream — a measurement is only correct if a real host attests against it, and
  measurements ship deliberately in a release. Do not change that.

- **Cost is bounded** by the rate limits: 120/hour × 256 KiB ≈ 21 GB/month worst case (~$0.32), with
  Class A operations inside the free tier. Fingerprint dedup does *not* bound this, since the
  submitter controls the fingerprint inputs.

- **Known limitation**: the 256 KiB cap is a policy check, not a memory guard. The body-hashing
  middleware buffers the body and FastAPI parses it before the handler's size check runs, and the
  API ingress sets `proxy-body-size: 0`. Platform-wide for every POST endpoint, not specific here.

- **Never serve this prefix publicly.**

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

Bodies over 256 KiB are rejected with 413; unknown or malformed fields with 422.

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

### Schema compatibility

Forbidding extras makes the coupling one-way:

- A `discover-profile.sh` **older** than the API is fine — absent keys take defaults.
- A `discover-profile.sh` that **adds** a key 422s until the API models it.

Release order: add the field to `HostProfile` (with a bound and a pattern), deploy the API, then
roll out the script. `tests/unit/test_host_profile_submission.py` holds a complete sample document
and asserts it validates — that is the test that fails when the script grows a field.

### Bucket layout (what the generation tooling consumes)

| | |
| --- | --- |
| Bucket | `STORAGE_BUCKET` (override with `HOST_PROFILE_BUCKET` to isolate them) |
| Key | `{HOST_PROFILE_PREFIX}/{fingerprint}.json` (default prefix `host-profiles`) |
| Body | the exact JSON the miner signed |
| Metadata | `hotkey`, `nonce`, `signature`, `fingerprint`, `gpu-count`, `gpu-pci-device-ids`, `submitted-at` |

The generation job lists the prefix, produces measurements for each object's host class, and deletes
the object once published. Put a prefix-scoped lifecycle rule on `host-profiles/` so abandoned
submissions expire — the API never deletes.
