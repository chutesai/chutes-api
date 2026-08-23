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
miner hotkey, and POSTs it. The API records the document keyed by a fingerprint of
the host class in the `host_profiles` table; offline tooling generates measurements, ships them in a
release, and the reconciler marks the row measured.

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
  - `api/server/schemas.py` — `HostProfileRecord` ORM model
  - `api/migrations/20260821120000_host_profiles.sql`
  - `api/config/__init__.py` — `discord_webhook_url`, `TeeMeasurementConfig.fingerprint`,
    `_load_tee_measurements`

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

- **A table, not object storage** (reversing the original "bucket only, no database" decision).
  That call was right for what a profile *was*: a transient request, deleted once measurements
  shipped. Two things changed it. Retention became permanent — a fingerprint is a one-way hash that
  cannot be inverted back to the topology inputs an RTMR0 regen needs, so the row is the only copy.
  And the day-to-day question turned out to be *review*: "which GPU types are waiting?". Permanent
  structured data you query belongs in the database. Concretely it buys:

  - **Queryability** — `WHERE profile @> '{"gpu": {"pci_device_ids": ["2901"]}}'` over a GIN index,
    instead of a LIST plus N GetObjects and reading raw JSON by hand.
  - **An atomic lifecycle** — `measured_at` replaces the `pending/` → `measured/` object move, so
    promotion is one UPDATE with no copy-then-delete crash window.
  - **Cheaper reads** — the topologies endpoint is one SELECT rather than N+1 round trips to R2.
  - **Attribution as a column** — `miner_hotkey` is queryable rather than S3 metadata.

- **The signed bytes are not kept, and nothing is ever deleted.** Two decisions that follow from
  what this table is *for*.

  The sr25519 signature is admission control at the endpoint: a request that fails it is rejected
  and never reaches the table, so a row existing already means it verified. Retaining `raw_body` to
  re-check that later would only defend against someone who can write to this table — an adversary
  nothing else in the schema defends against — at the cost of a second copy of every document that
  can silently disagree with `profile`. `miner_hotkey` stays as attribution for triage; `nonce` and
  `signature` went with the bytes, since a signature you cannot verify against anything is dead
  weight.

  (There is no cheaper middle: the signed message is `hotkey:nonce:sha256(body)`, so storing just
  the 64-char hash would preserve verifiability — but JSONB normalisation means it cannot be
  recomputed from `profile`, so it would only prove someone signed *something*.)

  There is also no expiry. Retention is the point: a measured row is the only copy of its
  topology's inputs, and a pending row nobody generated for is still the record that someone asked.
  If a reason to remove one ever appears, it should be a soft delete that keeps the history.

- **Fingerprint keying, first-write-wins**: key is `{prefix}/{HostProfile.fingerprint}.json`,
  covering only what changes the generated measurement — passthrough inventory, CPU identity and
  topology, host RAM, NUMA layout, QEMU/`-cpu` args. Hostname, BIOS strings and the lspci tree are
  stored but excluded, so a 200-host fleet of one class produces one object. A known fingerprint
  HEADs the key, returns `stored: false`, and writes nothing.

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

- **The fingerprint is the id linking the two halves**: submissions (the row's primary key) and
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

- **Why the constraints matter**: `gpu.pci_device_ids` and `gpu.count` reach log lines, Discord
  alerts, and the public topologies endpoint. Unconstrained, a CRLF in a device id is a
  log-injection primitive (loguru validates nothing). After validation every such value is provably
  constrained: 4-hex device ids, bounded counts.

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
  submitter controls the fingerprint inputs. Nothing reclaims junk, since nothing is deleted — the
  rate limits are the only bound, and at a few KB per host class the table stays small regardless.

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
| otherwise, a row on file (pending or measured) | `pending` — queued or generated |
| none of the above | `unknown` — only reachable with `dry_run`, since a real submission gets parked |

**rc is a property of a version, not of a topology.** A topology is measured or it is not, so there
is no measured-but-rc state: rc measurements are simply skipped when reading topology status, and a
fingerprint carried only by an rc version reports `pending` off the table. Correct — nothing it can
launch is published yet.

**Status comes from the measurement config, never from `measured_at`.** The config is the sole
truth for attestability; `measured_at` just records that generation happened. Deriving `accepted`
from the column would let the table and the config drift into disagreeing about what can launch.

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

1. **Submission-driven topologies** — the fingerprint *is* the row's primary key. Read it from
   `host_profiles` (or from the submission response) and copy it onto the entry; the reconciler
   marks the row measured once the config carries it.
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

One row per measured host class — the generated set. The profile is the stored document
re-emitted in its **wire shape** (`host`, `launch_determinism`, ...), stripped of the fields that
identify the individual machine rather than the host class: `hostname` and `timestamp`
(`HOST_PROFILE_PRIVATE_FIELDS`). The submitter's `miner_hotkey` is a column this query never
selects, so it cannot leak. Everything else — gpu/cpu/memory/numa/qemu plus
BIOS, board and the lspci tree — is generic host-class data and is exactly what reproducing RTMR0
needs.

Two consumers:

- **Independent verification.** Join to `GET /tee/measurements` on `fingerprint`: regenerate RTMR0
  from the inputs here and compare it to the published measurement. A quote holder can also see
  which host class their own RTMR0 corresponds to.
- **The sek8s `chutes-cvm generate-measurements` CLI**, which reads topologies from here rather
  than storage directly. That is why it is public: the generation side needs topologies, not
  database credentials.

### Lifecycle job (notify + reconcile)

`host_profile_reconciler.py`, on a CronJob. Both responsibilities are the API's rather than the
CLI's, so the generation side stays storage-agnostic. Neither deletes anything:

- **Notify** — a pending row with no `notified_at` posts a Discord alert naming the fingerprint and
  the GPU count/ids, so triage does not require opening the document. `notified_at` is stamped only
  after the webhook accepts, so an outage re-alerts rather than silently swallowing the only
  notification. It **never triggers generation** — a human decides.
- **Reconcile** — the measurement config is the source of truth. Every published fingerprint with a
  pending row is marked measured in one atomic UPDATE. rc counts here: an rc entry means the
  topology *was* generated, even though it cannot launch yet.

### Storage layout

| | |
| --- | --- |
| Table | `host_profiles`, primary key `fingerprint` |
| `profile` | JSONB, the document in wire shape; GIN-indexed for containment queries |
| `miner_hotkey` | who submitted it (attribution for triage) |
| `measured_at` | NULL = pending; set = measured and retained permanently |
| `notified_at` | when the "new host class" alert went out |

Review queries, which are the reason this is a table:

```sql
-- What hardware is waiting on measurements?
SELECT fingerprint, profile->'gpu'->>'count' AS gpus, profile->'gpu'->'pci_device_ids' AS ids
FROM host_profiles WHERE measured_at IS NULL ORDER BY created_at;

-- Every host class with a given GPU.
SELECT fingerprint FROM host_profiles WHERE profile @> '{"gpu": {"pci_device_ids": ["2901"]}}';
```

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
