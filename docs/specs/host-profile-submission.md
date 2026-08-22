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
in a release, and promotes the object from `pending/` to `measured/`.

- **Key files**:
  - `api/server/router.py` — `submit_host_profile` (`POST /servers/tdx/host_profiles`)
  - `api/server/schemas.py` — `HostProfile` (+ `HostProfile.fingerprint`), `HostProfileSubmissionResponse`
  - `api/server/util.py` — `store_host_profile`, `resolve_host_profile_status`,
    `topology_measurement_status`, `list_measured_topologies`, `promote_host_profile`,
    `reconcile_host_profiles`
  - `host_profile_reconciler.py` + `charts/templates/host-profile-reconciler-cronjob.yaml` —
    notify + reconcile job
  - `api/notify.py` — Discord webhook alerts
  - `api/server/host_profile_fingerprint.py` — `python -m` fingerprint utility for the generator
  - `api/rate_limit.py` — `rate_limit_miner` (auth + per-miner metering), `check_rate_limit`
  - `api/constants.py` — size cap, submission limits, field bounds
  - `api/config/__init__.py` — `host_profile_bucket`, `host_profile_prefix`,
    `TeeMeasurementConfig.fingerprint`, `_load_tee_measurements`
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

- **Bucket only, no database**, and profiles under `measured/` are **retained permanently**. A
  fingerprint is a one-way hash: it cannot be inverted back to the topology inputs it was computed
  from. Regenerating RTMR0 after a firmware or QEMU change needs those inputs, so discarding a
  profile once its measurement ships would throw away the only copy. Only `pending/` — submissions
  that were never generated — expires.

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

- **The API owns the fingerprint, and answers the status question** (reversing an earlier decision
  in this spec that there should be no status endpoint). That decision assumed the miner could tell
  for itself whether its hardware was supported. It cannot, without reimplementing the fingerprint —
  and a second implementation that drifts silently breaks the accepted/pending distinction. So the
  miner sends raw platform metadata and gets back a status word; computation, keying and matching
  all live here. If the key definition changes, it changes in one place.

- **The fingerprint is the id linking the two halves**: submissions (bucket object key) and
  measurements (`fingerprint` on each `hardware` entry). It is computed once, by
  `HostProfile.fingerprint`, at submission — never recomputed by a second code path.

- **Measurements stay in the config map**, not a database. The git history of the values file is the
  audit trail for what was accepted and when, reviewed like any other change. Status is answered by
  reading the loaded measurement set directly — no new store, and no cache: the parse costs ~5ms and
  submissions are capped at 120/hour, while the attestation path already pays the same cost far more
  often. Caching would only have bought staleness.

- **Gated to measurements ≥1.4.0**: 1.4.0 is the first VM version shipping the CLI that calls this
  endpoint, so nothing older ever asks. The gate is what makes the answer *true* rather than merely
  convenient — `accepted` means "the version you are running can launch here", and a host class
  measured only on 1.3.x cannot launch 1.4.0, so reporting `accepted` off a 1.3.x entry would be a
  lie. It also scopes the `fingerprint` backfill: the 24 pre-1.4.0 entries never need one, because
  1.4.0 entries are generated with a fingerprint from the start.

- **`dry_run` for check-without-write**: computes and reports status but stores nothing. Serves a
  pure "is my hardware supported?" check, and lets the release workflow mint a fingerprint for a
  profile that was never submitted. Without it, only `accepted` and `pending` are reachable, since a
  real submission either matches a measurement or gets parked.

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
  submitter controls the fingerprint inputs. Note the `pending/` expiry rule is what reclaims junk;
  `measured/` grows only when a real measurement ships, so it stays small.

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

Optional query param: `dry_run=true` to compute status without storing.

Bodies over 256 KiB are rejected with 413; unknown or malformed fields with 422.

### Response

```json
{
  "fingerprint": "<sha256 hex>",
  "status": "pending",
  "stored": true,
  "detail": "Host profile stored; measurements will be generated and published in a future release."
}
```

`stored: false` means nothing was written — either the host class is already `accepted`, or it was
already submitted (the correct outcome for the second host in a fleet, not an error).

### Schema compatibility

Forbidding extras makes the coupling one-way:

- A `discover-profile.sh` **older** than the API is fine — absent keys take defaults.
- A `discover-profile.sh` that **adds** a key 422s until the API models it.

Release order: add the field to `HostProfile` (with a bound and a pattern), deploy the API, then
roll out the script. `tests/unit/test_host_profile_submission.py` holds a complete sample document
and asserts it validates — that is the test that fails when the script grows a field.

### Status

Every submission returns `status`, computed from the fingerprint:

Only measurements at or above `MIN_HOST_PROFILE_MEASUREMENT_VERSION` (1.4.0) are considered:

| condition | status |
| --- | --- |
| a **non-rc** measurement ≥1.4.0 carries the fingerprint | `accepted` — this host class can launch |
| otherwise, an object under `pending/` or `measured/` | `pending` — queued or generated |
| none of the above | `unknown` — only reachable with `dry_run`, since a real submission gets parked |

**rc is a property of a version, not of a topology.** A topology is measured or it is not, so there
is no measured-but-rc state: rc measurements are simply skipped when reading topology status, and a
fingerprint carried only by an rc version reports `pending` off the bucket. Correct — nothing it can
launch is published yet.

**Status comes from the measurement config, never from the `measured/` prefix.** The config is the
sole truth for attestability; `measured/` is just the retained generation store. Deriving `accepted`
from the prefix would let the bucket and the config drift into disagreeing about what can launch.

A real submission stores whenever we do not already hold the profile — *including* when a
measurement already covers it, since a fingerprint cannot be inverted and an accepted host class
with no stored profile could never be regenerated. `dry_run` never writes.

`dry_run=true` (query param, default false) computes and reports status without writing. It uses the
same signed auth and consumes the same quota as a real submission.

### Release-workflow contract (fingerprint propagation)

The generator **propagates** the fingerprint onto each `hardware` entry it publishes. It must never
recompute it with its own algorithm: a mismatch silently breaks accepted-detection, and a host class
that has measurements keeps reporting `pending` forever.

Two sources, both single-sourced from `HostProfile.fingerprint`:

1. **Submission-driven topologies** — the fingerprint *is* the bucket object key
   (`host-profiles/pending/{fingerprint}.json`). Read it off the key, copy it onto the entry, then
   move the object to `measured/`.
2. **Seed / build-time topologies** never submitted through the API:

   ```
   python -m api.server.host_profile_fingerprint <profile.json>   # or - for stdin
   ```

   which prints the fingerprint for a discover-profile.sh document. (`dry_run=true` against a
   running API returns the same value and is the alternative if invoking this repo is inconvenient.)

### `GET /servers/tdx/topologies` (public)

Unauthenticated, redis-cached, anonymously rate-limited — same shape as `GET /tee/measurements`,
and designed to be read alongside it.

```json
[{"fingerprint": "<sha256 hex>", "profile": { ...discover-profile document... }}]
```

One entry per object under `measured/` — the generated set. The profile is the stored document
re-emitted in its **wire shape** (`host`, `launch_determinism`, ...), stripped of the fields that
identify the individual machine rather than the host class: `hostname` and `timestamp`
(`HOST_PROFILE_PRIVATE_FIELDS`). The submitter's hotkey/nonce/signature live in S3 object metadata,
which this path never reads, so they cannot leak. Everything else — gpu/cpu/memory/numa/qemu plus
BIOS, board and the lspci tree — is generic host-class data and is exactly what reproducing RTMR0
needs.

Two consumers:

- **Independent verification.** Join to `GET /tee/measurements` on `fingerprint`: regenerate RTMR0
  from the inputs here and compare it to the published measurement. A quote holder can also see
  which host class their own RTMR0 corresponds to.
- **The sek8s `chutes-cvm generate-measurements` CLI**, which reads topologies from here rather
  than S3. That is why it is public: the generation side needs topologies, not bucket credentials.

### Bucket lifecycle job (notify + reconcile)

`host_profile_reconciler.py`, on a CronJob. Two responsibilities, both owned by the API rather than
the CLI, so the generation side stays storage-agnostic:

- **Notify** — a new fingerprint under `pending/` posts a Discord alert. Deduplicated through a
  redis set, and only marked once the webhook actually accepts it, so an outage re-alerts rather
  than silently swallowing the only notification. It **never triggers generation** — a human
  decides.
- **Reconcile** — the measurement config is the source of truth. For every fingerprint published
  there, if `pending/{fp}` exists it is moved to `measured/{fp}`. Idempotent, and copy-then-delete
  so a crash mid-move leaves the object in both prefixes (the next run resolves it) rather than in
  neither. Reconcile follows the config regardless of `rc`: an rc entry means the topology *was*
  generated, so its profile belongs in the retained set even though it cannot launch yet.

### Bucket layout (what the generation tooling consumes)

| | |
| --- | --- |
| Bucket | `STORAGE_BUCKET` (override with `HOST_PROFILE_BUCKET` to isolate them) |
| Queued | `{HOST_PROFILE_PREFIX}/pending/{fingerprint}.json` — where submissions land; expires |
| Generated | `{HOST_PROFILE_PREFIX}/measured/{fingerprint}.json` — retained permanently |
| Body | the exact JSON the miner signed |
| Metadata | `hotkey`, `nonce`, `signature`, `fingerprint`, `gpu-count`, `gpu-pci-device-ids`, `submitted-at` |

A host class is **known** if an object exists under *either* prefix; neither is ever overwritten, so
a generated profile is never re-queued.

The generation job lists `pending/`, produces measurements for each object's host class, and
**moves** the object to `measured/`. That move is release-workflow, not API — this service only lays
out the two prefixes and reads them. The API never deletes and never writes to `measured/`.

Put the expiry lifecycle rule on **`pending/` only**: those are submissions nobody generated. A rule
covering `measured/` would delete topology inputs that cannot be reconstructed.

### Measurement config: `fingerprint` on `hardware` entries

```yaml
measurements:
  - version: "1.4.0"
    mrtd: "..."
    hardware:
      - name: "8xh200"
        rtmr0: "..."
        fingerprint: "5fcb3e1c..."   # optional; 64 hex chars; links to the submitted profile
        expected_gpus: ["h200"]
        gpu_count: 8
```

Optional and validated only when present (64 hex chars, lower-cased), because existing entries
predate the field. Surfaced on `GET /servers/tee/measurements` — public transparency, so a third
party can see which host class a measurement covers.

**An entry ≥1.4.0 without a fingerprint is unmatchable**: its host class reports `pending` forever
even though it launches fine. Every 1.4.0+ entry must carry one. Pre-1.4.0 entries are outside the
gate and need no backfill.

---

## Rollout Notes

1. **Create the `pending/` expiry lifecycle rule** scoped to that prefix alone — never
   `measured/`, and never the bare `host-profiles/` prefix.
2. **No backfill of the 24 pre-1.4.0 entries** — they sit below
   `MIN_HOST_PROFILE_MEASUREMENT_VERSION` and are ignored by status. This is deliberate: their
   fingerprints cannot be recovered from source anyway (`HostProfile.fingerprint` hashes host RAM,
   BAR size, VRAM, NUMA and NIC counts, which a `TopologyFingerprint` in `profiles.py` does not
   carry), so backfilling them would mean running `discover-profile.sh` on a live host of each of
   the 24 classes. Instead, **every 1.4.0 `hardware` entry must ship with a `fingerprint`** — it is
   generated alongside RTMR0, so there is nothing to reconstruct.
3. **Freshness after publish**: status reads the measurement set live, so a newly published
   measurement takes effect as soon as the ConfigMap remount propagates — no cache to wait out.
   (`GET /servers/tee/measurements` is separately cached for `TEE_MEASUREMENTS_CACHE_TTL`, so newly
   published `fingerprint` values appear *there* on that delay; status is unaffected.)

---

## Attestation failure messages

The sek8s initramfs reads the response body's `detail` on a failed boot attestation and prints it,
so an operator sees a reason rather than a bare status code. The attestation exception layer already
maps `AttestationError.message` -> `detail`; what changed is that the messages now say what to *do*:

- `MeasurementMismatchError` — names the actual condition (no registered measurement for this
  host's topology x QEMU) and points at `chutes-cvm discover-profile`. **The wording is deliberately
  identical for "nothing matched" and "an rc measurement you are not authorized for"**: those two
  must stay indistinguishable, so the message may not reveal which occurred.
- `NonceError` and the nonce dependencies — say that nonces are single-use and short-lived, and
  which call issues the one being asked for.
- `NoClientCertError` / `InvalidClientCertError` — name the CVM proxy and the per-boot certificate.
- `InvalidQuoteError` / `InvalidSignatureError` — distinguish "could not parse" from "could not
  verify against Intel collateral".

**The fingerprint is not included**, despite being the most useful thing to print. It is not
derivable from a quote: a quote carries MRTD/RTMRs, and RTMR0 is a one-way hash chain over the
topology, not the topology itself. Mapping RTMR0 -> fingerprint would need a measurement entry for
that topology — which by definition does not exist in the failure case. The operator gets the
fingerprint from the submission response instead, where it is computed from the real document.
