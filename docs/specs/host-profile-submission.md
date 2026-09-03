# Feature Spec: chutes-api — Miner Host Profile Submission

**Date**: 2026-08-21
**Status**: implemented

---

## Context

A sek8s VM only launches on a host class with published TEE measurements. `discover-profile.sh`
captures the host's shape, the miner-side tooling matches it against the baselined fingerprints in
`host-tools/scripts/chutes/guest/gpu/profiles.py`, and an unmatched host fails the launch.

This is the request channel for new hardware: the miner CLI runs `discover-profile.sh`, signs the
JSON with the miner hotkey, and POSTs it. The API records the document in `host_profiles`, keyed by
a fingerprint of the host class, and answers whether that class can launch yet. Offline tooling
generates measurements and ships them in a release; the reconciler then marks the row measured.

- **Key files**:
  - `api/server/router.py` — `submit_host_profile`, `list_host_profiles`
  - `api/server/schemas.py` — `HostProfile` (+ `HostProfile.fingerprint`), `HostProfileRecord` ORM
    model, `HostProfileSubmissionResponse`, `HostProfileResponse`
  - `api/server/util.py` — `store_host_profile`, `resolve_host_profile_status`,
    `host_profile_measurement_status`, `host_profile_is_known`, `list_pending_profiles`,
    `list_measured_host_profiles`, `reconcile_host_profiles`
  - `api/migrations/20260821120000_host_profiles.sql`
  - `host_profile_reconciler.py` + `charts/templates/host-profile-reconciler-cronjob.yaml`
  - `api/notify.py` — Discord webhook alerts
  - `api/rate_limit.py` — `rate_limit_miner` (auth + per-miner metering), `check_rate_limit`
  - `api/constants.py` — size cap, submission limits, field bounds, version gate
  - `api/config/__init__.py` — `discord_webhook_url`, `TeeMeasurementConfig.fingerprint`

---

## Design Decisions

- **On the servers router, under `/tdx`**: submission is a miner action but acts on
  server/attestation state, so it sits with `GET /servers/tee/measurements` — the endpoint saying
  which host classes are already known. `/tdx` rather than `/tee` because everything is TEE-only
  now, and AMD SEV-SNP will need distinguishing from Intel TDX. `/tee/measurements` keeps its path
  (public, external consumers), so the two are briefly inconsistent.

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

- **Machine-identifying fields are never persisted.** `hostname` and `timestamp` are declared
  `exclude=True` on `HostProfile`, so a real `discover-profile.sh` document still validates but
  they are dropped from every `model_dump` — including the one that writes the row. The column
  therefore holds only host-class data, and the public endpoint cannot leak them regardless of how
  it serialises, with no filter to keep in sync. A host class does not have a hostname.

- **One field, one home**: `launch_determinism` restates `numa` (`numa_node_count`,
  `numa_topology_eligible`) and `cpu` (`host_cpu_topology`). Declared so submissions validate, but
  never read — `fingerprint` takes those from their canonical block, so a disagreeing copy is inert.

- **The API owns the fingerprint and answers the status question.** A miner cannot tell for itself
  whether its hardware is supported without reimplementing the fingerprint, and a second
  implementation that drifts silently breaks the accepted/pending distinction. So the miner sends
  raw platform metadata and gets back a status word; computation, keying and matching all live here.
  If the key definition changes, it changes in one place.

- **The fingerprint is the id linking the two halves**: submissions (the row's primary key) and
  measurements (`fingerprint` on each `hardware` entry). It is computed once, by
  `HostProfile.fingerprint`, at submission — never recomputed by a second code path.

- **Fingerprint keying, first-write-wins**: the primary key is `HostProfile.fingerprint`, covering
  only what changes the generated measurement — passthrough inventory, CPU identity and topology,
  host RAM, NUMA layout, QEMU/`-cpu` args. Hostname, BIOS strings and the lspci tree are stored but
  excluded, so a 200-host fleet of one class produces one row. A known fingerprint hits
  `ON CONFLICT DO NOTHING`, returns `stored: false`, and writes nothing.

- **Profiles live in Postgres.** They are permanently retained (a fingerprint is a one-way hash that
  cannot be inverted back to the topology inputs an RTMR0 regen needs, so the row is the only copy),
  and the day-to-day question is review: "which GPU types are waiting?". A table gives:

  - **Queryability** — `WHERE profile @> '{"gpu": {"pci_device_ids": ["2901"]}}'` over a GIN index.
  - **An atomic lifecycle** — `measured_at` is one UPDATE, with no half-moved state.
  - **Cheap reads** — the public listing endpoint is a single SELECT.
  - **Attribution as a column** — `miner_hotkey` is queryable.

- **The signed bytes are not retained.** The sr25519 signature is admission control at the endpoint:
  a request that fails it is rejected and never reaches the table, so a row existing already means
  it verified. Keeping the exact bytes to re-check later would only defend against someone who can
  write to this table — an adversary nothing else in the schema defends against — at the cost of a
  second copy of every document that can silently disagree with `profile`. `nonce` and `signature`
  are not kept either, since a signature that cannot be verified against anything is dead weight;
  `miner_hotkey` remains as attribution for triage.

- **Nothing is ever deleted.** A measured row is the only copy of its topology's inputs, and a
  pending row nobody generated for is still the record that someone asked. If a reason to remove one
  ever appears, it should be a soft delete that preserves the history.

- **One dependency for auth and metering**: the route declares
  `Depends(rate_limit_miner("host_profile_submit", ...))` and nothing else. Auth is a
  *sub*-dependency, not a sibling: FastAPI resolves sub-dependencies first, whereas siblings have no
  guaranteed order and a counter keyed on an unverified hotkey would let anyone lock a miner out.
  The global ceiling is metered there too — a pre-auth global counter can be run down with
  unauthenticated junk.

- **Measurements stay in the config map**, not the database. The git history of the values file is
  the audit trail for what was accepted and when, reviewed like any other change. Status reads the
  loaded measurement set directly — no cache: the parse costs ~5ms and submissions are capped at
  120/hour, while the attestation path already pays the same cost far more often. Caching would only
  buy staleness.

- **Gated to measurements ≥1.4.0**: 1.4.0 is the first VM version shipping the CLI that calls this
  endpoint, so nothing older ever asks. The gate is what makes the answer *true* — `accepted` means
  "the version you are running can launch here", and a host class measured only on 1.3.x cannot
  launch 1.4.0. It also scopes the `fingerprint` requirement: the 24 pre-1.4.0 entries need none,
  because 1.4.0 entries are generated with one from the start.

- **`dry_run` for check-without-write**: computes and reports status but stores nothing. Serves a
  pure "is my hardware supported?" check, and lets the release workflow mint a fingerprint for a
  profile that was never submitted. Without it, only `accepted` and `pending` are reachable, since a
  real submission either matches a measurement or is recorded.

---

## Threat model

A submission is **untrusted, unattested, self-reported data**. No TDX quote backs it — a miner
describes hardware, and nothing proves they own it.

- **Reaching the endpoint** costs a registered, un-blacklisted hotkey and a valid signature over
  `hotkey:nonce:sha256(body)` with a nonce inside ±600s. Registration cost is the sybil bound; rate
  limits (10/hotkey/hour, 120/hour overall) are counted only after the signature verifies.

- **Why the field constraints matter**: `gpu.pci_device_ids` and `gpu.count` reach log lines,
  Discord alerts, and the public listing endpoint. Unconstrained, a CRLF in a device id is a
  log-injection primitive (loguru validates nothing). After validation every such value is provably
  constrained: 4-hex device ids, bounded counts.

- **The loosest fields** are `pci_topology` (bounded; newlines and tabs allowed since the tree is
  drawn with them, other control characters rejected) and the DMI strings. Bounded is not the same
  as safe to interpolate: **the generation job must never shell-interpolate, path-join, or evaluate
  a field**. `cpu_args` and `host_cpu_topology` are restricted to a shell-safe charset because they
  describe a command line that job may construct.

- **Fingerprint squatting is possible and accepted**: a miner can submit a profile whose identity
  fields match a host class they do not own, and first-write-wins means their non-identity fields
  are what the generation job sees. `miner_hotkey` makes it attributable; the real defense is
  downstream — a measurement is only correct if a real host attests against it, and measurements
  ship deliberately in a release. Do not change that.

- **Growth is bounded** by the rate limits (120/hour × 256 KiB worst case) and by fingerprint keying
  collapsing a fleet to one row. Dedup does not bound a determined submitter, since they control the
  fingerprint inputs, and nothing is deleted — but at a few KB per host class the table stays small.

- **Known limitation**: the 256 KiB cap is a policy check, not a memory guard. The body-hashing
  middleware buffers the body and FastAPI parses it before the handler's size check runs, and the
  API ingress sets `proxy-body-size: 0`. Platform-wide for every POST endpoint, not specific here.

---

## Contract

### `POST /servers/tdx/host_profiles`

```
POST https://api.chutes.ai/servers/tdx/host_profiles
X-Chutes-Hotkey:    <miner ss58>
X-Chutes-Nonce:     <unix seconds, within 600s of now>
X-Chutes-Signature: <sr25519 hex over "{hotkey}:{nonce}:{sha256(body)}">
Content-Type: application/json

<discover-profile.sh output, verbatim>
```

Optional query param: `dry_run=true` to compute status without storing. It uses the same signed auth
and consumes the same quota as a real submission.

Bodies over 256 KiB are rejected with 413; unknown or malformed fields with 422.

```json
{
  "fingerprint": "<sha256 hex>",
  "status": "pending",
  "stored": true,
  "detail": "Host profile stored; measurements will be generated and published."
}
```

`stored: false` means nothing was written — either the host class is already `accepted`, or it was
already submitted (the correct outcome for the second host in a fleet, not an error).

### Status

Only measurements at or above `MIN_HOST_PROFILE_MEASUREMENT_VERSION` (1.4.0) are considered:

| condition | status |
| --- | --- |
| a **non-rc** measurement ≥1.4.0 carries the fingerprint | `accepted` — this host class can launch |
| otherwise, a row on file (pending or measured) | `pending` — queued or generated |
| none of the above | `unknown` — only reachable with `dry_run`, since a real submission is recorded |

**rc is a property of a version, not of a topology.** A topology is measured or it is not, so there
is no measured-but-rc state: rc measurements are skipped when reading status, and a
fingerprint carried only by an rc version reports `pending` off the table — nothing it can launch is
published yet.

**Status comes from the measurement config, never from `measured_at`.** The config is the sole truth
for attestability; `measured_at` only records that generation happened. Deriving `accepted` from the
column would let the table and the config drift into disagreeing about what can launch.

A real submission stores whenever we do not already hold the profile — *including* when a
measurement already covers it, since a fingerprint cannot be inverted and an accepted host class
with no stored profile could never be regenerated.

### Schema compatibility

Forbidding extras makes the coupling one-way:

- A `discover-profile.sh` **older** than the API is fine — absent keys take defaults.
- A `discover-profile.sh` that **adds** a key 422s until the API models it.

Release order: add the field to `HostProfile` (with a bound and a pattern), deploy the API, then roll
out the script. `tests/unit/test_host_profile_submission.py` holds a complete sample document and
asserts it validates — that is the test that fails when the script grows a field.

### `GET /servers/tdx/host_profiles` (public)

The GET sibling of the submission endpoint: unauthenticated, redis-cached (per variant), anonymously
rate-limited.

```
GET /servers/tdx/host_profiles                       # measured only (default)
GET /servers/tdx/host_profiles?include_pending=true  # + host classes awaiting generation
```

```json
[{"fingerprint": "<sha256 hex>", "measured": true, "profile": { ...discover-profile document... }}]
```

**Measured only by default.** That is the set a third party can actually verify: join `fingerprint`
to `GET /servers/tee/measurements`, regenerate RTMR0 from the inputs here, and compare. A quote
holder can also see which host class their own RTMR0 corresponds to. No flags to reason about, and
an unverified claim is never handed to someone who did not ask for one.

**`include_pending=true` is for the measurement generator.** Pending host classes are its work
queue, and they have to be reachable somehow: a profile becomes measured only once measurements are
generated for it, and generation has to fetch it first. Publishing only the measured set would make
a newly submitted host class permanently unreachable by the pipeline meant to act on it. Each entry
carries `measured` so the generator can tell the queue from the rest.

A pending entry records that some registered miner submitted this hardware shape; nothing attests
that they own it. Only a measured entry with a matching published measurement says anything about
what can launch.

Profiles are returned exactly as stored, in the **wire shape** (`host`, `launch_determinism`, ...).
Nothing is filtered on read, because the machine-identifying fields never enter the column;
`miner_hotkey` is a column this query never selects. What remains — gpu/cpu/memory/numa/qemu plus
BIOS, board and the lspci tree — is host-class data and is exactly what reproducing RTMR0 needs.

### Release-workflow contract (fingerprint propagation)

The generator **propagates** the fingerprint onto each `hardware` entry it publishes. It must never
recompute it with its own algorithm: a mismatch silently breaks accepted-detection, and a host class
that has measurements keeps reporting `pending` forever.

The fingerprint always comes from the API, which is the only thing that computes it:

- **Submitted host classes** — `GET /servers/tdx/host_profiles` returns it alongside each profile,
  and `POST /servers/tdx/host_profiles` returns it in the submission response.
- **Seed / build-time host classes** — submit the `discover-profile.sh` document like any other.
  That records it, returns its fingerprint, and makes the host class appear on the GET once
  measured, so third parties can reproduce its RTMR0. `dry_run=true` returns the same value without
  recording, for a look before committing to it.

An entry ≥1.4.0 without a `fingerprint` is unmatchable: its host class reports `pending` forever even
though it launches fine.

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

| column | |
| --- | --- |
| `fingerprint` | primary key; `HostProfile.fingerprint` |
| `profile` | JSONB, the document in wire shape; GIN-indexed for containment queries |
| `miner_hotkey` | who submitted it (attribution for triage) |
| `created_at` | submission time |
| `measured_at` | NULL = pending; set = measured |
| `notified_at` | when the "new host class" alert went out |

Review queries, which are the reason this is a table:

```sql
-- What hardware is waiting on measurements?
SELECT fingerprint, profile->'gpu'->>'count' AS gpus, profile->'gpu'->'pci_device_ids' AS ids
FROM host_profiles WHERE measured_at IS NULL ORDER BY created_at;

-- Every host class with a given GPU.
SELECT fingerprint FROM host_profiles WHERE profile @> '{"gpu": {"pci_device_ids": ["2901"]}}';
```

### Measurement config: `fingerprint` on `hardware` entries

```yaml
hardware:
  - name: "8xh200"
    rtmr0: "..."
    fingerprint: "5fcb3e1c..."   # 64 hex chars; links to the submitted profile
    expected_gpus: ["h200"]
    gpu_count: 8
```

Optional and validated only when present, because pre-1.4.0 entries predate the field. Surfaced on
`GET /servers/tee/measurements` — public transparency, so a third party can see which host class a
measurement covers.

---

## Attestation failure messages

The sek8s initramfs reads the response body's `detail` on a failed boot attestation and prints it, so
an operator sees a reason rather than a bare status code. The attestation exception layer maps
`AttestationError.message` → `detail`; the messages say what to *do*:

- `MeasurementMismatchError` — names the condition (no registered measurement for this host's
  topology × QEMU) and points at `chutes-cvm discover-profile`. **The wording is deliberately
  identical for "nothing matched" and "an rc measurement you are not authorized for"**: those two
  must stay indistinguishable, so the message may not reveal which occurred.
- `NonceError` and the nonce dependencies — say that nonces are single-use and short-lived, and
  which call issues the one being asked for.
- `NoClientCertError` / `InvalidClientCertError` — name the CVM proxy and the per-boot certificate.
- `InvalidQuoteError` / `InvalidSignatureError` — distinguish "could not parse" from "could not
  verify against Intel collateral".

**The fingerprint is not included**, despite being the most useful thing to print. It is not
derivable from a quote: a quote carries MRTD/RTMRs, and RTMR0 is a one-way hash chain over the
topology, not the topology itself. Mapping RTMR0 → fingerprint would need a measurement entry for
that topology, which by definition does not exist in the failure case. The operator gets the
fingerprint from the submission response instead.

---

## Rollout Notes

1. **Apply the migration** (`20260821120000_host_profiles.sql`) before deploying; the endpoint writes
   to `host_profiles` from the first request.
2. **Every 1.4.0 `hardware` entry must ship with a `fingerprint`** — generated alongside RTMR0, so
   there is nothing to reconstruct. Pre-1.4.0 entries sit below the version gate and need none; their
   fingerprints could not be recovered from source anyway, since `HostProfile.fingerprint` hashes
   host RAM, BAR size, VRAM, NUMA and NIC counts that a `TopologyFingerprint` does not carry.
3. **Status is read live** from the measurement config, so a newly published measurement takes
   effect as soon as the ConfigMap remount propagates. `GET /servers/tee/measurements` and
   `GET /servers/tdx/host_profiles` are separately cached for `TEE_MEASUREMENTS_CACHE_TTL`, so new
   values appear *there* on that delay; status is unaffected.
4. **Set `hostProfileReconciler.discordWebhookSecretName`** to enable submission alerts; unset
   disables alerting and reconcile still runs.
