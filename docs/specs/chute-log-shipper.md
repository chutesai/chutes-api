# Feature Spec: Chute Log Shipper — Validator (chutes-api) side

**Date**: 2026-07-23
**Status**: implemented (branch `feat/chute-logs`) — validator side in `api/chute_logs/` + the ingest
endpoint on the instance router + the Loki chart. The two coordinated guest changes (send
`deployment_id`; treat `403`/`404` as terminal) are tracked in the sek8s spec.
**Revised**: 2026-07-29 — aligned with the **finalized guest contract** (sek8s `feat/log-service`,
wire contract frozen 2026-07-27) and the **shipped CVM mTLS migration** (this branch,
`docs/specs/cvm-mtls-migration.md`). See "Alignment deltas" below for what changed from the
2026-07-23 draft and which changes are **coordinated** (require a matching guest-side edit).
**Companion (guest) spec**: `sek8s/docs/specs/chute-log-shipper.md`.

---

## Context

When a chute crashes/errors **before its instance is registered**, its logs are lost. Every current
path needs the validator to know the chute's `host:port`, which only exists once an `Instance` row
is created on launch-config *claim*, after verification (`api/instance/router.py`, `Instance(...)`
construction ~line 1451). A chute that dies before claim leaves a `LaunchConfig` with `failed_at`
and **no Instance**, so `GET /miner/instance_logs` (`api/miner/router.py:651`) and the
`encrypted_logs` capture (started from `api/instance/router.py:118`) both have nothing to read.

The fix: an **in-guest agent** (in the miner's confidential VM — `sek8s` `chute-log-shipper.service`,
already implemented) watches chute pods via the CRI socket + `/var/log/pods` and **streams their
logs to the validator** starting at pod-start, independent of instance registration. This spec is the
**validator side**: authenticate those shipments, decide when capture stops, store the logs briefly,
and expose them to owners + the miner CLI. Chutes-internal support reads are **out of scope for this
repo** (see "Storage" — the ops-managed Grafana consumes the store directly).

Companion behavior the guest relies on:
- The guest ships batches; **the validator's HTTP response tells it whether to keep going or stop.**
  `204 No Content` = stop; any other 2xx = keep sending. Default: stop when the instance activates.
  A per-chute override keeps it going for debugging.
- **Terminal rejects** (`403`/`404`) also stop the guest (coordinated guest change — see §2/§3).
- The guest authenticates by presenting its per-boot registry mTLS leaf; identity is resolved
  **server-side** from that cert + the request path + the proxy (the guest self-asserts no identity).

---

## Alignment deltas (2026-07-23 draft → shipped guest + infra)

The guest wire contract was frozen **after** the original draft, and the CVM mTLS proxy the draft
speculated about is now real. The concrete changes:

1. **Request body is log-lines-only + `deployment_id`.** The draft's rich body (`chute_id`,
   `server_ip`, `pod_phase`, `exit_code`, `reason`, `seq`, `captured_at`) is gone. The guest sends
   `{"deployment_id": "…", "logs": [{"ts","stream","log"}]}` — everything else is derived server-side.
   (`deployment_id` is a **coordinated addition**: the guest reads the `chutes/deployment-id` pod
   label — already present, previously discarded — and will send it top-level.)
2. **Cutoff is `204` = stop, any other 2xx = keep** (draft said `200`/`409`). This is the guest's
   `stop_status_code` (`sek8s .../log_shipper/config.py:35`).
3. **No `seq`.** Dedupe on `(config_id, ts)` nanosecond timestamp; reliable delivery (retry +
   advance-cursor-on-success) replaces gap detection. `captured_at` also dropped.
4. **`server_ip` is not in the body** — it is taken from the proxy (`X-Real-IP` / `X-Forwarded-For`,
   which the cvm-proxy sets to `$remote_addr`; the Service uses `externalTrafficPolicy: Local` to
   preserve the real client IP).
5. **`chute_id` / `miner_hotkey` are derived** from `LaunchConfig(config_id)` (both are columns:
   `api/instance/schemas.py:174,184`), not sent.
6. **Path is settled: `POST /instances/launch_config/{config_id}/logs`** on `cvm.chutes.ai` — hardcoded
   in the guest (`sek8s .../log_shipper/config.py:141`). The draft's alternative `/servers/{vm_name}/…`
   is **not** used.
7. **Terminal rejects stop the guest.** The shipped guest retries *all* non-2xx; a **coordinated
   guest change** makes `403`/`404` terminal (stop + log the specific reason). Until that lands the
   guest will retry these transiently — bounded by backoff, not fatal, but noisy. (There is no
   wall-clock capture cap on either side; capture ends on lifecycle state — see §3.)
8. **Store is a dedicated in-namespace Loki, not the ops OpenSearch cluster.** Support surfacing is
   out of scope for this repo (see "Storage"). The draft's OpenSearch index-template/ILM +
   support-read endpoint + audit are removed; a Loki-idiomatic label/field split replaces them.

---

## Design Decisions

- **Dedicated Loki, bounded, Grafana-fed.** Store shipped logs at the validator in a **dedicated
  single-instance Loki** deployed by this repo's chart into the `chutes` namespace (alongside Redis
  etc.), with **24h** retention, so the feature is self-contained and does **not** mix into the
  ops-managed monitoring OpenSearch cluster. The ops team's existing Grafana (OAuth, on the internal
  Tailscale network) consumes it as a datasource for internal/support viewing — **this repo neither
  stands up Grafana nor exposes a support-read endpoint nor audits support reads** (those live at the
  Grafana/OAuth layer). Chosen over VM-resident logs because `pods/log` dies with pod GC / node reboot
  — exactly when post-mortem debugging happens.
- **Direct push to Loki, never via stdout.** The ingest handler pushes lines straight to Loki's HTTP
  push API (`POST /loki/api/v1/push`). It must **not** tee chute lines to the API's own logger/stdout:
  the API pod's stdout is already scraped by Fluent Bit into the ops store, so teeing would leak chute
  logs into exactly the store we're keeping them out of. Push ≠ stdout ⇒ Fluent Bit never sees them
  ⇒ no duplication, no mixing. (This is why the draft's "structured stdout + Fluent Bit" ingestion
  option is rejected.)
- **Loki label discipline (not OpenSearch keyword mapping).** Loki punishes high-cardinality labels.
  So **labels are low-cardinality only** (`app="chute-log-shipper"`, `stream`); the
  high-cardinality identifiers (`config_id`, `chute_id`, `server_ip`, `miner_hotkey`, `deployment_id`)
  ride as **structured JSON fields in the log line**, filtered via LogQL (`| json | config_id="…"`).
  Grafana filtering by any of those fields works unchanged; it is line-filtering, not a keyword index.
- **Validator-controlled cutoff, signalled by the response.** The API owns all cutoff logic; the guest
  obeys the status code: **`204` = stop, any other 2xx = keep**, `403`/`404` = terminal reject (guest
  stops + logs), other non-2xx/connection error = transient (guest retries with backoff). Default:
  stop once the instance is activated; a per-chute override flag keeps it going for debugging.
- **mTLS is enforced, via the shipped cvm-proxy** (no hotkey fallback). See §2.
- **Support-readability without owner-only encryption is acceptable** for this new, additive store:
  chute stdout is operational logs, not user request/response bodies (the `stream_logs` docstring
  asserts prompts/responses do not flow through stdout by design). The existing owner-only
  `encrypted_logs` path is unchanged (§5) — no confidentiality regression.
- **Confidentiality is enforced by construction, not by a stored field (§6).** A **private chute's
  logs must never reach anyone but its owner or Chutes support.** That guarantee rests on three
  things, not on the presence of a `user_id` field: (a) the API resolves authorization from the DB at
  **read time** and then builds the LogQL with a **server-forced matcher** on the authenticated
  principal — the client never supplies raw LogQL and can only *narrow*, never widen; (b) the
  **mutable** `public` flag is **re-resolved from the DB at read time** (it is never stamped into the
  store, so there is no point-in-time value to trust) for any public-only surface; (c) **Loki is never
  client-reachable** —
  no ingress, a default-deny NetworkPolicy admits only the API (via the `loki-access` label) plus any
  reader granted through the ops-repo values (`loki.extraIngressFrom`). We stamp the immutable
  owner **`user_id`** into each doc as the *scope target* for the forced matcher, but the store's
  fields are never the access decision. **`public` is deliberately not stamped** — it is mutable, so
  it is always re-resolved from the DB at read time (§6). See §6.
- **Config-keyed for the new store.** The shipped/pre-registration store and its reads key on
  `config_id`; there is no `Instance` (and thus no `instance_id`, no `host:port`) in the window this
  feature exists to cover. **The existing owner read keeps its current keying/interface unchanged**
  (§5) — do not force it onto `config_id`.

---

## API Changes

### 1. Ingest endpoint
`POST /instances/launch_config/{config_id}/logs`
- Placed on the `instance_router` (`/instances`) beside the launch-config lifecycle sub-resources,
  keeping `config_id`-keyed operations together. This is the exact path the guest targets on the
  `cvm.chutes.ai` mTLS host.
- **Request body (final):**
  ```json
  {
    "deployment_id": "…",
    "logs": [
      { "ts": "2026-07-27T00:00:00.123456789Z", "stream": "stdout", "log": "…" }
    ]
  }
  ```
  `ts` is an RFC3339 **nanosecond** timestamp; `stream` ∈ {`stdout`,`stderr`}. `config_id` comes from
  the **path**; the `Server` (and thus `miner_hotkey`/`vm_name`) comes from the **CA-matched mTLS
  leaf** (§2); `chute_id`/`user_id` from the **`LaunchConfig`→`Chute` lookup**; `server_ip` from the
  **proxy** (`X-Real-IP` / `X-Forwarded-For`). None of these are in the body.
- **Response (the cutoff signal):**
  - `204 No Content` → **stop** capturing this pod (default: instance activated).
  - any other 2xx (use `200`) → **keep sending**.
  - `404` → unknown/nonexistent `config_id` → guest **stops + logs** (coordinated guest change).
  - `403` → cert fails CA verification, or `LaunchConfig.miner_hotkey ≠ authenticated miner`
    (cross-miner injection) → guest **stops + logs**.
  - other non-2xx / connection error → guest treats as **transient** (retry with backoff).
- **Idempotency:** dedupe by `(config_id, ts)` (nanosecond ts is effectively unique per line in a
  single container stream); a replayed/duplicate batch is a no-op. No `seq`.
- **Bounded ingest:** the guest caps batches (default 500 lines / 1 MiB, 16 KiB/line); the handler
  should still guard against absurd bodies. `client_max_body_size 0` on the proxy is unlimited —
  enforce a sane cap in the app.

### 2. Authentication — mTLS via the shipped cvm-proxy (enforced, no fallback)
The guest presents its per-boot VM **registry mTLS client leaf** (reused, no new leaf). The validator
authenticates the shipment by that cert, binding it to the attested boot. **mTLS is required.**

**The infra already exists on this branch** (`docs/specs/cvm-mtls-migration.md`):
- `cvm.chutes.ai` → the `cvm-proxy` (nginx, `charts/templates/cvm-proxy-cm.yaml`), terminating mTLS
  with `ssl_verify_client optional_no_ca`, forwarding `X-Client-Cert` / `X-Client-Verify` and
  injecting the provenance secret `X-Cvm-Proxy-Auth` = `CVM_PROXY_SECRET`. `_get_client_certificate`
  trusts `X-Client-Cert` only when a valid proxy secret is present; the public `api.chutes.ai` ingress
  strips the provenance headers. `require_cvm_proxy()` (`api/server/util.py:120`) fails closed (503 if
  `CVM_PROXY_SECRET` unset, 403 if not via the proxy).
- **Backward-compat naming is already handled** by the two-proxy design (`tdx-attestation.chutes.ai`
  remains for legacy 1.3.x via `attestation-proxy`; 1.4.0+ CVM traffic is `cvm.chutes.ai`). This
  endpoint is **cvm-only** (1.4.0+ guest), so it uses `require_cvm_proxy()` — no legacy alias needed.

**To add log ingest:**
- **Proxy:** extend `cvm-proxy-cm.yaml`. The current single `location` regex is anchored to
  `^/servers/(nonce|boot/attestation|…)$`; add the log path to it (or a second `location`) so
  `POST /instances/launch_config/{config_id}/logs` is forwarded to `api.<ns>.svc:8000` with the same
  `X-Client-Cert` / `X-Client-Verify` / `X-Cvm-Proxy-Auth` / `X-Real-IP` headers (everything else
  still `return 404`). The `location /` 404 default means an un-added path is rejected, so this is the
  only proxy change.
- **Route:** gate with `require_cvm_proxy()` + `extract_client_cert()` (implemented on the
  `instance_router` as `POST /launch_config/{config_id}/logs`).
- **Identity resolution — CA match (the intended design; implemented in
  `api/chute_logs/service.py` `authenticate_shipment`):** the VM mTLS client leaf uses a **common CN
  shared across all VMs**, so the leaf's *subject* is not a VM identity. VM identity comes from **which
  CA signed the leaf** — each server records its own per-boot CA (`Server.vm_root_ca_cert`) at
  provision time. So:
  1. `LaunchConfig(config_id)` → owning `miner_hotkey` (404 if the config doesn't exist → guest stops).
  2. Load that miner's `Server` rows with a recorded CA (`Server.vm_root_ca_cert IS NOT NULL`).
  3. Accept iff the leaf verifies against **one** of those CAs
     (`verify_leaf_cert_signed_by_ca(client_cert, server.vm_root_ca_certificate)`,
     `api/server/util.py:983`); no match → **403** (guest stops).
  - Cross-miner injection is impossible **by construction**: only the config-owner's CAs are tried, so
    a leaf signed by another miner's VM CA cannot verify — no separate hotkey cross-check is needed.
  - **Miner-scoped, not IP-pinned, on purpose:** pre-registration the launch config is not yet bound
    to a server, so the pod could be on any of the miner's VMs; matching against the miner's CA set is
    the correct granularity. Each server currently has a unique IP, so IP→server resolution is also
    available, but it isn't needed for auth here. `server_ip` (from `X-Real-IP` / `X-Forwarded-For`) is
    stored as a line field only.

### 3. Cutoff logic (API-owned)
The API decides; the response code carries the decision (§1).
- **Default:** *keep sending* (`200`) while there is no instance yet OR the instance exists but is not
  activated; *stop* (`204`) once the instance is **activated**.
- **⚠ Do NOT reuse `_is_instance_activated(config_id)` verbatim** (`api/encrypted_logs/capture.py:138`).
  It returns `True` (→ "stop") when the instance row is **absent** — correct for the existing capture
  (which starts *after* instance creation), but **wrong here**: in the pre-registration window there
  is legitimately no instance yet, and returning "stop" would kill exactly the capture this feature
  exists for. Truth table: **no instance → continue**; instance exists, `activated_at is None` →
  continue; `activated_at` set → **stop (204)**; `LaunchConfig.failed_at` set → the pod has failed, so
  the guest hits pod-terminal and stops shipping anyway (validator may also `204` after the final
  batch).
- **Per-chute debug override:** resolve `chute_id` from `LaunchConfig(config_id)` first, then if
  Redis key `pod_logs_debug:{chute_id}` is set, keep sending (`200`) regardless of activation. Add an
  admin/support endpoint to set/clear it; document that enabling it captures post-activation output.
- **Terminal rejections:** unknown/nonexistent `config_id` → **404** (guest stops); ownership mismatch
  → **403** (guest stops).
- **No wall-clock cap (by design).** Capture stops only on lifecycle state — activation, `failed_at`,
  or the config being deleted (miner preempt / validator teardown all set one of these). A config that
  never reaches a terminal state is an *anomaly whose logs we want for debugging*, not something to
  time out; and ingest is already bounded per-batch and by Loki retention, so nothing runs away. The
  proper guards for a genuinely immortal pod live upstream (k8s Job `active_deadline_seconds`) and in a
  future shipper-sent *terminated* signal (deferred follow-up) — not a timer in the log path.

### 4. Storage — dedicated Loki, direct push
- **Store:** a **dedicated single-instance Loki** in the `chutes` namespace, deployed by this repo's
  chart (`charts/`), **24h** retention (Loki compactor + `retention_period`). Separate from the
  ops-managed monitoring OpenSearch cluster. Ops adds it as a Grafana datasource.
- **Ingestion:** the handler pushes to `POST /loki/api/v1/push` (streams keyed by labels; values
  `[<ts_ns>, <line>]`) using the already-present async HTTP client. **Never** teed to stdout (see
  Design Decisions).
- **Label vs field split (Loki-idiomatic):**
  - **Labels (low-cardinality):** `app="chute-log-shipper"`, `stream` (`stdout|stderr`). No
    lifecycle/phase label — capture is binary (see §3), and a per-line phase marker carried no
    useful signal (most lines are pre-terminal regardless of how the launch ends).
  - **JSON line fields (high-cardinality, LogQL-filterable):** `config_id`, `chute_id`, `server_ip`,
    `miner_hotkey`, `deployment_id`, **`user_id`** (chute owner — the forced-matcher target for owner
    reads, §6), plus `ts`, `stream`, and `log`.
  - `user_id` is **derived server-side at ingest** from `config_id → LaunchConfig.chute_id → Chute`
    (`Chute.user_id` is an immutable FK — `api/chute/schemas.py:219`). **`public` is deliberately not
    stamped:** `Chute.public` is a mutable Boolean (`api/chute/schemas.py:226`) that can flip inside the
    retention window, so public/private is always re-resolved from the DB at read time (§6), never from
    the store.
- **Retention:** 24h via Loki config (no per-index ILM template needed).

### 5. Read paths
All authorization/isolation rules for these live in §6; this section is *what* each surface exposes.

> **Status (this branch):** only ingestion + the internal Grafana read path ship. Both external
> reads below — the miner fallback and the owner read — are **deferred to a standalone,
> security-reviewed follow-up** so the new store is validated internally before any user/miner can
> read from it. The router mounts at `/logs`; only the support-role debug-override endpoints
> (`/logs/debug/{chute_id}`) are exposed now. The read primitive `service.read_config_logs`
> (server-forced to the owner's `user_id`) exists and is unit-tested, so the follow-up adds only the
> routes + auth.

- **Support / internal:** **out of scope for this repo.** The ops Grafana (OAuth + Tailscale) queries
  the dedicated Loki directly, filterable by `chute_id` / `config_id` / `server_ip` / `miner_hotkey`
  via LogQL. No support-read API endpoint, no `chutes_support` support gate, and no
  support-read audit are built here. (Trusted internal surface; unfiltered by design.)
- **CLI / miner (pre-registration fallback) — DEFERRED (follow-up):** extend `GET /miner/instance_logs`
  (`stream_miner_logs`, `api/miner/router.py`) to serve from Loki keyed by `config_id` when there is no
  live instance / `failed_at` is set (today it 422s). Authed by the launch JWT (`sub=config_id`) — no
  new key. **Verified caveat for the follow-up:** this endpoint is **public-chute-only** (403s on
  private) and rejects already-activated — keep those guards; the public check must be **re-resolved
  from the DB at read time** (§6). `public` is not stamped into the store, precisely so there is no
  point-in-time value to accidentally trust. *This branch leaves `stream_miner_logs` unchanged (still
  422s); the extension lands in the standalone follow-up.*
- **Owner read (`config_id`-keyed, pre-registration only) — DEFERRED (follow-up):** a `config_id`-keyed
  owner read for the window no existing path covers (a chute that never got an `instance_id`),
  server-forced to the authenticated owner (§6), so the **future end-user surface** can front the same
  store behind a different auth layer without cross-tenant leakage. *Not exposed in this branch — the
  `service.read_config_logs` primitive exists and is tested, but the route is intentionally not
  registered until the security-reviewed follow-up.* Do **not** change the existing owner paths:
  - *Live logs:* `GET /instances/{instance_id}/logs` — `get_current_user(purpose="logs")`, keyed by
    **`instance_id`** (`api/instance/router.py:2973`, authz L3008-3017; owner branch requires private).
  - *Startup/encrypted logs:* `GET /encrypted_logs/{chute_id}/sessions` +
    `…/sessions/{instance_id}/chunks` — `get_current_user(purpose="chutes")`, owner-only, keyed by
    **`chute_id`**, client-side-decrypt (`api/encrypted_logs/router.py:16-81`).
  - **Do not change these paths, auth, keying, or the decrypt contract.** The new store *adds* the
    `config_id`-keyed owner read; it does not replace them, and the 4h Redis `encrypted_logs` store
    stays (do not retire in Phase 2).

### 6. Confidentiality & isolation (private-chute logs must never leak)
**Invariant: a private chute's logs are visible only to its owner and to Chutes support.** Enforced by
construction — the Loki fields are query *scope*, never the access *decision*:

1. **Loki is not client-reachable.** No ingress; a **default-deny** NetworkPolicy admits only
   in-namespace pods labelled `loki-access` (the API). Additional readers (e.g. the ops Grafana) are
   granted **per-cluster via `loki.extraIngressFrom` in the ops-repo values** — the chart names no
   external infra. Every field-level protection below is
   void if a user can reach Loki directly, so this perimeter is the primary control.
2. **Clients never supply raw LogQL.** The API builds the query. Client input is limited to *narrowing*
   parameters (time range, a specific `config_id`/`chute_id` the caller is already authorized for);
   it can never alter or remove the forced matcher.
3. **Authorization is DB-resolved at read time, then a server-forced matcher scopes the query** to the
   authenticated principal (belt: DB authz; suspenders: forced matcher — a query bug still can't return
   another principal's lines):
   - **Owner / future end-user:** authenticate → DB confirms the caller `user_id` owns the requested
     `chute_id` → force `| json | user_id="<authed_user_id>"` (+ any narrowing `chute_id`/`config_id`).
     Returns the caller's private **and** public logs, only theirs.
   - **Miner CLI:** launch JWT (`sub=config_id`) → DB **re-check the chute is currently public** and the
     `config_id` belongs to that miner → force `| json | config_id="<jwt config_id>"` (`config_id` is
     unique to that miner's launch attempt ⇒ no cross-miner bleed). Never returns private-chute logs.
   - **Support:** ops Grafana, trusted, unfiltered.
4. **Public/private is resolved from the DB at read time, never stamped.** `Chute.public` can flip
   public→private inside the 24h window; the miner path therefore re-resolves current public/private
   from the DB at read time. `user_id` is safe to gate on (ownership is effectively immutable).
   `public` is deliberately **not** stamped into the store — there is no point-in-time value to fall
   back on, by design.
5. **Fallback if physical partitioning is later required:** Loki multi-tenancy via per-owner
   `X-Scope-OrgID`. Not the default — it explodes tenant count and breaks support's cross-tenant view;
   the non-reachable + forced-matcher + read-time-authz design above is sufficient.

---

## Goal

Success (Phase 1) =
- `POST /instances/launch_config/{config_id}/logs` authenticates the guest via the cvm-proxy mTLS
  leaf, verifies config ownership, pushes lines to the dedicated Loki keyed by `config_id`, and
  returns `204` (stop) / `200` (continue) / `403`/`404` (terminal reject).
- A chute that **crashes before registration** has its logs in Loki within seconds and retrievable
  via `chutes-miner instance-logs` and the ops Grafana view.
- A chute that **activates** gets `204` on the next batch; setting the override returns `200`.
- Replayed/duplicate batches are idempotent on `(config_id, ts)`; unknown `config_id` → 404;
  cross-miner shipment → 403.
- Owners can retrieve their own pre-registration logs (`config_id`-keyed), and the store is shaped so
  the future end-user surface can reuse it under different auth.

---

## Constraints

- Async-first; reuse existing Redis + async-HTTP clients; reuse the `encrypted_logs` helpers where
  they fit (do **not** repurpose their keying).
- No owner-only encryption for the new store; access control lives on the read surfaces (CLI JWT,
  owner auth, ops Grafana OAuth).
- Bounded ingestion: per-config `(config_id, ts)` dedupe, body-size cap, 24h Loki retention.
- Config-keyed everywhere; do not depend on `Instance`/`instance_id` existing.
- **Never tee chute lines to the API's own stdout/logger** (would leak into the ops store via
  Fluent Bit).

---

## Failure Conditions

- Accepts shipments on a route that does not verify the client cert / provenance secret (mTLS is
  mandatory — via `require_cvm_proxy()` + `extract_client_cert()`), or without verifying
  `LaunchConfig.miner_hotkey == authenticated miner_hotkey` (cross-miner log injection).
- Keys the pre-registration store/reads by `instance_id` (there is none in this window).
- Changes or breaks the existing owner read path/keying (§5).
- Uses `409`/`200`-only cutoff instead of **`204` = stop**, or returns a *retryable* status for
  unknown/unauthorized configs instead of the **terminal `404`/`403`** the guest stops on.
- Relies on `seq`/`captured_at`/`server_ip`/`chute_id` **in the body** (none are sent) instead of
  deriving them; or dedupes on anything but `(config_id, ts)`.
- Makes `config_id` (or another high-cardinality id) a **Loki label** (cardinality explosion) instead
  of a JSON line field.
- **Tees chute lines to stdout**, leaking them into the ops OpenSearch cluster via Fluent Bit.
- **Exposes a private chute's logs to a non-owner/non-support caller** (§6) via any of: Loki reachable
  by clients (missing ingress-block / NetworkPolicy); a client-supplied or client-widenable LogQL query
  instead of a server-forced matcher; gating a public-only surface on a **stamped** `public` field
  (it is deliberately not stamped) instead of re-resolving current public/private from the DB at read
  time; or omitting the owner
  `user_id` forced matcher on the owner/end-user path.
- Loki docs missing any queryable field (`config_id`, `chute_id`, `server_ip`, `miner_hotkey`,
  `deployment_id`, `user_id`, `stream`).
- No retention (unbounded Loki growth) — must be 24h.
- Never returns `204` (unbounded ingestion) — missing activation check / override handling.

---

## Rollout Notes

- **Ordering:** ship this (or a stub `204`/`200` endpoint) before/with the guest; a guest shipment to
  a missing endpoint fails closed (retry/backoff, no crash). The cvm-proxy + `CVM_PROXY_SECRET` must
  be provisioned (already done on this branch per `cvm-mtls-migration.md`) before the route enforces.
- **Coordinated guest changes** (owned by sek8s, land together): (a) send top-level `deployment_id`;
  (b) treat `403`/`404` as terminal (stop + log the reason) rather than retrying.
- **Phase 2 (validator side):** cut the support/live-log path over from proxying the per-chute 8001
  log server to this store; migrate `log_prober` off its 8001 `/logs` assertion; extend to running
  logs (default-off, via override/policy) so Grafana has full-lifecycle logs. **Preserve the owner
  read** (§5). (Guest/miner repoint the readiness probe off 8001 → 8000 and delete the 8001 server +
  NodePort — see the sek8s spec.)
- **End state — the VM push becomes the *sole* log mechanism, retiring both `instance_logs` and
  `encrypted_logs`.** The pod-resident paths are unreliable: the per-chute 8001 server and `pods/log`
  die with pod GC / node reboot and don't exist at all pre-registration. Shipping from the **VM**
  (independent of pod lifecycle) is the durable source, so it should eventually back **all** owner /
  support / CLI log reads, not just the pre-registration window. The design here is already shaped for
  that — `config_id`-keyed ingest, owner `user_id` stamped per line, and server-forced reads mean the
  owner live-log read can move onto Loki without a new data model. Three things must be settled when we
  make that cutover, and are called out now so we don't design them out:
  1. **Confidentiality trade.** `encrypted_logs` is owner-only, client-side-decrypted — the validator
     cannot read it. Retiring it moves owner logs into this **support-readable** store. That is only
     acceptable under the standing assumption that chute **stdout is operational logs, not request/
     response bodies** (§Design Decisions / §6). Re-confirm that assumption (and consider optional
     scrubbing on the owner/end-user path) before retiring the ECIES channel — do not retire it in
     Phase 1.
  2. **Running-log scale.** Shipping steady-state logs for **every** chute (not just pre-registration
     crashes) is a different volume class. The single filesystem-backed Loki here is sized for the
     bounded pre-registration window; full-lifecycle ingestion needs an object-store backend
     (S3/GCS) + horizontal scale + real retention sizing. Treat the current single-instance Loki as
     the Phase-1 store, not the end-state topology.
  3. **Cutoff + guest capture.** Full-lifecycle means the guest ships running logs (its default-off
     running-log capture flips on) and the validator stops returning `204` at activation for those
     configs — i.e. the activation cutoff becomes policy-driven rather than the default.
- **Grafana:** ops points their existing OAuth/Tailscale Grafana at the new Loki datasource; no
  Grafana work in this repo.
