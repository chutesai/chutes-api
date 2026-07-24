# Feature Spec: Chute Log Shipper — Validator (chutes-api) side

**Date**: 2026-07-23
**Status**: draft
**Target repo**: `chutes-api` (this doc is self-contained; copy it into that repo)

---

## Context

When a chute crashes/errors **before its instance is registered**, its logs are lost. Every current
path needs the validator to know the chute's `host:port`, which only exists once an `Instance` row
is created on launch-config *claim*, after verification (`api/instance/router.py`, `Instance(...)`
construction ~line 1451). A chute that dies before claim leaves a `LaunchConfig` with `failed_at`
and **no Instance**, so `GET /miner/instance_logs` (`api/miner/router.py:651`) and the
`encrypted_logs` capture (started from `api/instance/router.py:118`) both have nothing to read.

The fix: an **in-guest agent** (in the miner's confidential VM, specified separately) watches chute
pods via the in-cluster k8s API and **streams their logs to the validator** starting at pod-start,
independent of instance registration. This spec is the **validator side**: authenticate those
shipments, decide when capture stops, cache the logs briefly, and expose them to owners + support.

Companion behavior the guest relies on:
- The guest ships batches; **the validator's HTTP response tells it whether to keep going or stop.**
  Default: stop when the instance activates. A per-chute override keeps it going for debugging.
- The guest authenticates as the miner (see §2), keyed to a `config_id` it read from pod labels.

- **Packages/areas affected**: `api/instance/router.py`, `api/miner/router.py`,
  `api/encrypted_logs/*` (reuse helpers; **preserve the owner read**), mTLS termination/verification
  for the API route (auth), `api/log.py` / OpenSearch stack (storage + Grafana), config.
- **Related existing code (verified)**:
  - mTLS client-cert verify: `verify_leaf_cert_signed_by_ca(client_cert, server.vm_root_ca_cert)`
    via `api/registry/router.py:_verify_mtls_client_cert` (L76-100); `require_mtls_domain()` +
    `extract_client_cert()` gate the mTLS-domain routes; `server.vm_root_ca_cert` column at
    `api/server/schemas.py:402`; registration handler `api/server/router.py:254` (`tdx-attestation.chutes.ai`).
    The plain `api.chutes.ai` ingress does **not** verify client certs.
  - Cutoff reference (but **do not reuse verbatim** — see §3): `_is_instance_activated(config_id)`
    `api/encrypted_logs/capture.py:138` returns `True` when the instance is activated **or absent**.
  - Existing Redis constants: `LOG_STREAM_TTL_SECONDS=14400` (4h), `MAX_CAPTURE_SECONDS=1800`,
    `LOG_STREAM_MAXLEN=5000`, keys `encrypted_logs:{instance_id}` / `encrypted_log_sessions:{chute_id}`
    (`api/encrypted_logs/capture.py`).
  - `chutes_support` role gate: `api/instance/router.py` `stream_logs` authz (L3008-3017).
  - Router mounts: `instance_router`=`/instances`, `miner_router`=`/miner`,
    `encrypted_logs_router`=`/encrypted_logs` (`api/main.py:157,161,171`).
- **Migrations**: add a per-chute debug-override flag store (Redis key or a column); OpenSearch
  index template + ILM policy if OpenSearch is chosen.

---

## Design Decisions

- **Central cache, bounded, support-readable.** Store shipped logs at the validator (OpenSearch)
  with a 12–24h TTL, keyed by `config_id`, readable by Chutes-internal support. Chosen over leaving
  logs on the VM because VM-resident `pods/log` dies with pod GC / node reboot — exactly when
  post-mortem debugging happens — and because VM-resident retrieval would need a config_id→node
  locator pushed here anyway.
- **Validator-controlled cutoff, signalled by the response.** The API owns all cutoff logic; the
  guest just obeys the HTTP response. Keep it dead simple: a **distinct status code** (e.g. `200` =
  keep sending, a specific code such as `409` = stop) — optionally with a small body reason. Default:
  stop once the instance is activated (`_is_instance_activated`); a per-chute override flag keeps it
  going for debugging recurring post-activation failures.
- **mTLS is enforced** (no hotkey fallback). See §2.
- **The new store is support-readable, access-controlled + audited (no owner-only encryption).**
  This is a *new, additive* store for support + the pre-registration window — the existing
  owner-only `encrypted_logs` path is unchanged (§5), so there is no confidentiality regression for
  owners. Support-readability of the new store is acceptable because **chute stdout is operational
  logs, not user request/response bodies** (prompts/responses do not flow through stdout by design —
  the `stream_logs` docstring asserts this). Residual risk (a chute author logging something
  sensitive) is handled by access control + audit now; optional scrubbing on the public path later.
- **Config-keyed for the new store.** The shipped/pre-registration store and its reads key on
  `config_id`; there is no `Instance` (and thus no `instance_id`, no `host:port`) in the window this
  feature exists to cover. **The existing owner read keeps its current keying/interface unchanged**
  (see §5) — do not force it onto `config_id`.

---

## API Changes

### 1. Ingest endpoint
`POST /instances/launch_config/{config_id}/logs`
- Placed alongside the existing launch-config lifecycle sub-resources `graval` / `activate` on the
  `instance_router` (`/instances`), keeping all `config_id`-keyed operations together. (Not under
  `/miner`, which holds miner-CLI *read* endpoints.)
- **Request body:**
  ```json
  {
    "chute_id": "…", "deployment_id": "…", "server_ip": "…",
    "pod_phase": "Pending|Running|Failed|Succeeded",
    "exit_code": 137, "reason": "…",
    "seq": 42, "captured_at": "2026-07-23T…Z",
    "lines": [ { "ts": "…", "log": "…" } ]
  }
  ```
  `seq` is a monotonically increasing per-`(config_id, boot)` counter for gap detection + replay
  dedupe. `server_ip` is the node IP the pod ran on (the guest reads it from the pod's
  `status.hostIP`); the validator may cross-check it against the authenticated server record
  resolved from the mTLS identity.
- **Response (the cutoff signal):** keep it simple — a **distinct HTTP status code** the guest keys
  off (e.g. `200` = keep sending; a specific code such as `409` = stop), optionally with a small
  `{ "reason": "…" }` body. No richer protocol than that.
- **Idempotency:** dedupe by `(config_id, seq)`; a replayed/duplicate batch is a no-op.

### 2. Authentication — mTLS (enforced, no fallback)
The guest presents its per-boot VM **registry mTLS client leaf**; the validator authenticates the
shipment by that cert, binding each shipment to the attested boot ("logs came from the server that
booted this instance"). **mTLS is required, no hotkey fallback.**

**Reuse the existing mTLS proxy; don't enable client-cert on the main API host.**
- `api.chutes.ai` and the mTLS domain are both plain DNS records to the same GCP LBs (no Cloudflare).
  The split exists because nginx's `ssl_verify_client` is scoped to the **server block / SNI**, not
  to a path (the TLS handshake, where the client cert is requested, precedes the URL) — so client-cert
  *requesting* can't be limited to specific API paths; enabling it would apply to the **entire
  `api.chutes.ai` server block**, i.e. every client's handshake. A dedicated server block isolates it.
  (With `optional_no_ca` non-mTLS clients still connect, so this is a boundary-cleanliness preference,
  not a hard wall — but reusing the dedicated proxy is clearly the right call.) Confirmed: the plain
  `api.chutes.ai` ingress has no client-cert config; `require_mtls_domain()` / `_get_client_certificate`
  reject `X-Client-Cert` unless it arrives via the dedicated mTLS proxy.
- **Domain = `cvm.chutes.ai` (new, vendor-neutral), with `tdx-attestation.chutes.ai` kept as a
  backward-compatible alias.** The name is `cvm.chutes.ai` (Confidential VM — covers TDX + AMD
  SEV-SNP; the channel carries attestation, LUKS, registry, `vm-root-ca`, and logs). **Old VMs have
  `tdx-attestation.chutes.ai` baked into their measured images and are already booted — that name
  must keep working unchanged.** So do NOT migrate; *add* `cvm.chutes.ai` alongside it:
  - **DNS:** add `cvm.chutes.ai` → same GCP LB; keep the `tdx-attestation.chutes.ai` record.
  - **nginx:** add `cvm.chutes.ai` to the proxy's `server_name`
    (`charts/templates/attestation-proxy-cm.yaml` — today
    `{{ .Values.attestationProxy.serverName | default "tdx-attestation.chutes.ai" }}`; make it accept
    both), and ensure the proxy TLS cert SANs cover **both** names (or a `*.chutes.ai` wildcard).
  - **App:** `require_mtls_domain()` must accept **both** hosts during the transition (likely a
    single configured value today — make it a list).
  - New VM image builds point at `cvm.chutes.ai`; old VMs keep hitting `tdx-attestation.chutes.ai`.
    New endpoints (this log ingest, and future ones) are "cvm-first," but since both names hit the
    same nginx/app they resolve identically.
- **The mechanism already exists.** The attestation-proxy nginx
  (`charts/templates/attestation-proxy-cm.yaml`) listens on `:8443` with
  `ssl_verify_client optional_no_ca` and **proxies to the same FastAPI app**
  (`api.<ns>.svc:8000`), forwarding `X-Client-Cert $ssl_client_escaped_cert` +
  `X-Client-Verify $ssl_client_verify` on specific `location` blocks (`/servers/boot/attestation`,
  `/servers/{x}/luks/attest`), with `location / { return 404; }` for the rest.
- **To add log ingest:** add **one `location` block** to that configmap forwarding the new path to
  the same `api:8000` upstream with the `X-Client-Cert`/`X-Client-Verify` headers (it 404s otherwise);
  the FastAPI route uses `require_mtls_domain()` + `extract_client_cert()` and verifies via
  **`verify_leaf_cert_signed_by_ca(client_cert, server.vm_root_ca_cert)`**
  (`api/registry/router.py:_verify_mtls_client_cert`; `server.vm_root_ca_cert` at
  `api/server/schemas.py:402`). Optionally a separate DNS-only subdomain (e.g. `logs.chutes.ai`) to
  the same nginx pattern if you want to isolate log traffic from attestation traffic on the LB.
- **Placement — prefer `/servers/{vm_name}/…`.** The mTLS proxy today carries only `/servers/…`
  location patterns, so `POST /servers/{vm_name}/logs` (or `/servers/{vm_name}/launch_config/{config_id}/logs`)
  slots in beside `/servers/boot/attestation` and `/servers/{x}/luks/attest` and matches the existing
  VM-identity/mTLS grouping. (The alternative `/instances/launch_config/{config_id}/logs` would need a
  new location shape on this proxy and sits oddly next to the JWT-authed `graval`/`activate` on the
  *plain* host.) Put `config_id` in the path/body.
- Then verify `LaunchConfig(config_id).miner_hotkey == authenticated miner_hotkey` (the mTLS identity
  resolves to `(miner_hotkey, vm_name)` via the `Server` record) and that `config_id` exists.

### 3. Cutoff logic (API-owned)
The API decides; the response code carries the decision (§1).
- **Default:** *keep sending* while there is no instance yet OR the instance exists but is not
  activated; *stop* once the instance is **activated**.
- **⚠ Do NOT reuse `_is_instance_activated(config_id)` verbatim** (`api/encrypted_logs/capture.py:138`).
  It returns `True` (→ "stop") when the instance row is **absent** — correct for the existing capture
  (which starts *after* instance creation), but **wrong here**: in the pre-registration window there
  is legitimately no instance yet, and returning "stop" would kill exactly the capture this feature
  exists for. Write a cutoff that distinguishes "no instance yet → continue" from "activated → stop".
  Suggested truth table: no instance → continue; instance exists, `activated_at is None` → continue;
  `activated_at` set → stop; `LaunchConfig.failed_at` set → the pod has failed, so the guest will hit
  pod-terminal and stop shipping anyway (validator may also return stop after the final batch).
- **Per-chute debug override:** if a flag is set (Redis key `pod_logs_debug:{chute_id}` or a column),
  keep sending regardless of activation. Add an admin/support endpoint to set/clear it; document that
  enabling it captures post-activation output.
- **Backstops:** stop for unknown/mismatched `config_id`; a max-capture-duration guard so a
  never-activating, never-terminating pod cannot stream forever.

### 4. Storage — OpenSearch, structured for filtering + Grafana
- **Store:** **OpenSearch**, so the same store backs both programmatic reads and a support **Grafana**
  dashboard. **Note (verified):** `api/log.py` does not write to OpenSearch directly — it emits
  JSON to **stdout**, and an out-of-repo **Fluent Bit DaemonSet → OpenSearch → Grafana** pipeline
  (not deployed by this chart) is what indexes it. So choose the ingestion path in-repo: either
  (a) emit each chute-log line as a **structured stdout JSON record** carrying the fields below, so
  the existing Fluent Bit pipeline ships it (simplest, but mixes into the validator's own log stream
  — use a distinct marker/field so it lands in a dedicated index), or (b) **write directly to a
  dedicated OpenSearch index** from the ingest handler. Confirm the pipeline/ownership with ops.
- **Each log document is JSON with, at minimum, these queryable fields** (so support can filter in
  Grafana / OS queries):
  - `chute_id`
  - `config_id`
  - `server_ip` — the miner node IP the pod ran on (from the pod's `status.hostIP`, shipped by the
    guest; optionally cross-checked against the authenticated server record)
  - `miner_hotkey`
  - plus: `deployment_id`, `outcome` (running|failed|activated), `ts`, `log` (the line)
- **Retention:** 12–24h via an OpenSearch ILM policy. Support-readable at rest (no owner-only
  encryption for this store).
- Add an index template + ILM policy; ensure the fields above are mapped as keyword/date for
  filtering.

### 5. Read paths
- **Support read:** query the OpenSearch store, gated to the `chutes_support` role (mirror the
  `stream_logs` role check, `api/instance/router.py:3006`), surfaced via the **Grafana** dashboard
  filterable by `chute_id` / `config_id` / `server_ip` / `miner_hotkey` / `outcome`. **Audit** every
  support read (who/what/when).
- **CLI / miner (pre-registration fallback):** extend `GET /miner/instance_logs` (`stream_miner_logs`,
  `api/miner/router.py:651`) to serve from the store keyed by `config_id` when there is no live
  instance / `failed_at` is set (today it 422s at L713-718). Authed by the launch JWT (`sub=config_id`)
  — no new key. **Verified caveat:** this endpoint is **public-chute-only** (403s on private, L697-702)
  and rejects already-activated (L707-711) — it is *not* an owner endpoint. Keep those guards unless
  deliberately widening scope.
- **Owner read — KEEP EXACTLY AS TODAY (verified keying):**
  - *Live logs:* `GET /instances/{instance_id}/logs` — `get_current_user(purpose="logs")`, authorized
    for owner / shared / `chutes_support`, keyed by **`instance_id`** (`api/instance/router.py:2973`,
    authz L3008-3017; owner branch requires the chute to be **private**).
  - *Startup/encrypted logs:* `GET /encrypted_logs/{chute_id}/sessions` then
    `GET /encrypted_logs/{chute_id}/sessions/{instance_id}/chunks` — `get_current_user(purpose="chutes")`,
    owner-only, keyed by **`chute_id`** (+ `instance_id` for chunks), returns base64 **encrypted**
    chunks for client-side (owner) decryption (`api/encrypted_logs/router.py:16-81`).
  - **Do not change these paths, auth, keying, or the client-side-decrypt contract.** They are
    `instance_id`/`chute_id`-keyed and (for encrypted) owner-only — fundamentally different from the
    new `config_id`-keyed, support-readable store. So the new store does **not** replace them; only
    *add* a `config_id`-keyed owner read for the pre-registration window that no existing path covers
    (a chute that never got an `instance_id`). The owner-only `encrypted_logs` path and its 4h Redis
    store therefore **stay** (do not retire in Phase 2).

---

## Goal

Success (Phase 1) =
- `POST /instances/launch_config/{config_id}/logs` authenticates the guest, verifies config
  ownership, stores lines in the cache keyed by `config_id`, and returns `continue`/`stop`.
- A chute that **crashes before registration** has its logs in the cache within seconds and
  retrievable via `chutes-miner instance-logs` and the support Grafana view.
- A chute that **activates** gets `stop` on the next batch; setting the override returns `continue`.
- Replayed/duplicate batches are idempotent; unknown/mismatched `config_id` is rejected.
- Support reads are access-controlled and audited; owners can retrieve their own logs.

---

## Constraints

- Async-first; reuse existing Redis/OpenSearch clients and the `encrypted_logs` helpers.
- No owner-only encryption for the new store (support-readable by decision) — but **enforce access
  control + audit**.
- Bounded ingestion: per-config size/line caps, TTL, `seq` dedupe.
- Config-keyed everywhere; do not depend on `Instance`/`instance_id` existing.

---

## Failure Conditions

- Accepts shipments on a route that does not verify the client cert (mTLS is mandatory — no hotkey
  or unauthenticated path), or without verifying `LaunchConfig.miner_hotkey == authenticated
  miner_hotkey` (cross-miner log injection).
- Keys the pre-registration store/reads by `instance_id` (there is none in this window).
- Changes or breaks the existing owner read path/keying.
- OpenSearch documents missing any of the required queryable fields (`chute_id`, `config_id`,
  `server_ip`, `miner_hotkey`) — support can't filter in Grafana.
- Never returns `stop` (unbounded ingestion) — missing activation check, override handling, or
  max-duration/unknown-config backstops.
- Corrupts ordering on out-of-order/gapped `seq`, or double-stores on replay.
- Exposes support reads without access control or without audit logging.
- No retention/TTL (unbounded storage growth).

---

## Rollout Notes

- **Ordering:** ship this before/with the guest agent, or have the guest gated off until this
  exists; a guest shipment to a missing endpoint should fail closed (no crash, ret/backoff).
- **Phase 2 (validator side):** cut the support/live log path over from proxying the per-chute 8001
  log server to this store; migrate `log_prober` off its 8001 `/logs` assertion; extend to running
  logs (default-off, via override/policy) so Grafana has full-lifecycle logs. **Preserve the owner
  read** — only retire the parts of `stream_logs`/`encrypted_logs` that are not the owner interface,
  or re-point the owner interface onto the new store while keeping its UX/keying identical (per §5,
  confirm current owner keying first). (Guest/miner also repoint the readiness probe off 8001 → 8000
  and delete the 8001 server + NodePort — see the sek8s spec.)
- **Grafana:** stand up the datasource over the OpenSearch store + a dashboard filterable by
  `chute_id` / `config_id` / `server_ip` / `miner_hotkey` / `outcome`.
