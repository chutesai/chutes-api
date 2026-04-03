# Feature Spec: TEE server maintenance window

Use the sections **Goal**, **Constraints**, **Output Format**, and **Failure Conditions** as a **Prompt Contract** for this task (see [AGENT.md](../../AGENT.md) at repo root).

**Date**: 2026-04-02  
**Status**: draft

**Scope note (this iteration):** Implement **maintenance window + preflight/confirm + auto-purge + boot completion** (`last_maintenance_completed_at`, active slot cleared). **Out of scope:** `scoring_penalty_multiplier`, cron-based penalties for outdated `measurement_version`, and changes to `metasync` / `INSTANCES_QUERY`.

---

## Context

Miners upgrading a TEE host need to **remove instances** from routing before reboot. Today, **miner-initiated** `delete_instance` can trigger **thrash** (no `valid_termination`) and **last-instance** scoring penalties (`api/instance/router.py`). This spec adds a **validator-controlled global window**, a **preflight** endpoint (see below), and a **confirm** step. On **successful confirm**, the platform **automatically terminates all instances** on that server (validator-driven eviction) using the same **purge** machinery as **watchtower** (`watchtower.purge` / `purge_and_notify`), with **`valid_termination = true`** and a dedicated **`deletion_reason`**, so routing and caches update **proactively**—the miner does not rely on manual deletes to drain the box. **Policy** must align **last-instance** / bounty handling on these purges with the maintenance story (see below).

- **Packages affected**: `api` (primary)
- **Key files**:
  - `api/server/router.py`, `api/server/service.py`, `api/server/schemas.py` — server model, preflight + confirm maintenance, boot attestation **completion** handling (`last_maintenance_completed_at` + clear active slot)
  - `api/config.py` (or equivalent settings module) — maintenance window and policy constants
  - `watchtower.py` — `purge` / `purge_and_notify` (or factored shared helper): reuse for maintenance-initiated teardown (`valid_termination=True`, notifications, cache invalidation)
  - `api/instance/router.py` — **Optional** fallback: if any code path still allows miner `delete_instance` while maintenance is active, keep protected behavior; primary drain is **declare → auto-purge**
  - `api/instance/util.py`, `api/node/schemas.py` — correlate `Instance` → `Node` → `Server` (forward-compatible with **planned** shared-IP TEE; see **TEE addressing: today vs planned** above)
  - `api/migrations/*.sql` — new columns on `servers`
  - `api/constants.py` — thrash constants (read-only context; no change required unless tests need it)
- **Dependencies**: Existing FastAPI, SQLAlchemy async sessions, `get_current_user` / miner auth patterns for server routes.

### TEE addressing: today vs planned

- **Today (production TEE):** The platform **enforces one public IP per server** for TEE. In that world, correlating an instance to a server via **`instances.host` and `servers.ip`** can match reality—but it is still **indirect** (instance does not carry `server_id`).
- **Planned (future feature):** **Multiple TEE servers may share one public IP** (e.g. NAT / shared egress). That is **not** live yet; wording elsewhere in this spec describes that **future** so maintenance does not bake in an IP-only assumption that will break later.
- **This spec:** Design **instance → server** resolution using **`Instance` → `Node` → `Server`** and **`(miner_hotkey, server name)`** for policy identity so behavior stays correct **now** and remains **forward-compatible** when shared-IP TEE ships.
- **Boot attestation vs instance paths (intentional split):** The **boot** endpoint today does **not** carry **server name** in params; the API ties the request to a **`Server`** row using **client IP** (e.g. `extract_ip(request)` → `servers.ip`), which is **correct under current one-IP-per-server TEE**. **Maintenance completion** on successful boot should use **that same IP → server association** for this iteration. **Planned:** when **multi-server-per-IP** ships, **extend boot attestation** (or related handshake) with an explicit **`vm_name` / server name** (and/or `server_id`) so the VM can disambiguate; maintenance completion then keys primarily off **`(miner_hotkey, name)`** (or `server_id`), not IP. **Implementation:** keep boot-side resolution in a **small helper** (e.g. `resolve_server_for_maintenance_boot_completion(...)`) so swapping **IP-only → name-aware** is a **localized** change once the attestation contract is updated.

---

## Design Decisions

- **No human admin:** Allow/deny is **fully automatic** inside the API. The **validator** owns **when** the feature applies via **published config** (environment / settings loaded by this service). No separate approve/deny HTTP routes.
- **Global window = admission window:** **`[maintenance_window_start, maintenance_window_end]`** is the period when a miner may **start** a new cycle (**preflight** / **confirm**). If `now` is **outside** that interval, **deny** new entry (**403** / **404**—document the choice). This is independent of **how long** a server stays “in maintenance” after approval:
  - **Already approved:** If the server has an **active slot** (`maintenance_deadline_at > now()`) because it **confirmed** before `window_end`, the miner may **finish** (reboot, boot attestation) **after** `window_end`. That is **allowed**—we do not revoke an in-flight slot when the clock passes `window_end`.
  - **No new admits after close:** Once the admission window has closed, **no further preflight/confirm** for anyone until the validator publishes a **new** window (or the same env vars move forward in time). So a **`last_maintenance_completed_at`** timestamp that falls **after** `window_end` does **not** by itself “re-open” abuse: the miner **cannot** call preflight/confirm again until the **next** admission window anyway.
- **One successful completion per server per admission campaign (anti-abuse inside an open window):** Persist **`last_maintenance_completed_at`** on `servers` when a cycle **finishes successfully** (boot success below). When evaluating **preflight** / **confirm**, if `now` is **inside** the current `[window_start, window_end]` **and** **`last_maintenance_completed_at` is not null** and lies **inside that same interval**, **deny** this server (already consumed one successful completion for this campaign). If **`last_maintenance_completed_at` is null** or **`last_maintenance_completed_at < window_start`** (prior campaign), the server **may** enter again **provided** `now` is still inside the admission window and other checks pass. **Rationale:** Without this, a miner could **confirm → purge → boot → immediately preflight/confirm again** **on the same calendar admission window** and repeat protected churn. **Note:** The **admission window** check and this timestamp check **stack**: after `window_end`, new entry is already impossible; **`last_maintenance_completed_at`** mainly guards **multiple successful cycles** while the window is **still open** (e.g. fast reboot same afternoon).
- **Per-server slot:** On successful declare, persist `maintenance_deadline_at = now + maintenance_grace_hours` (validator constant). Only **one active slot per server** at a time; re-declare while active returns **409** unless policy allows idempotent refresh (default: **409**).
- **Per-miner concurrency (validator capacity knob):** A miner may have at most **`maintenance_max_concurrent_servers_per_miner`** servers in an **active** maintenance state at once (**default `1`**). **Active** means `maintenance_deadline_at IS NOT NULL` and `maintenance_deadline_at > now()` (same notion as “slot not expired”). Before accepting **confirm**, **count** distinct `servers` rows for that `miner_hotkey` meeting that condition; if **count ≥ limit**, return **409** with a clear message (and optionally `current_slots` / `limit` in JSON). An active slot **ends** when **`maintenance_deadline_at` has passed** (**without** setting `last_maintenance_completed_at`—miner failed or abandoned upgrade; **lazy** re-evaluation and/or optional **cron** to null `maintenance_declared_at` / `maintenance_deadline_at` only) or when **successful boot** runs the **completion** path below (**active** fields cleared, **`last_maintenance_completed_at` set**). **No miner `DELETE`** to clear maintenance. **Operator-only** override (if ever needed) is out of scope here.
- **Two-step flow (preflight + confirm):**
  - **Preflight:** Read-only check whether **confirm** would succeed **right now**—same auth as confirm (miner + `server_id`). Returns **`eligible: true`** or **`eligible: false`** with structured reasons (outside window, **already completed maintenance in this window** (`last_maintenance_completed_at`), sole-survivor `chute_id`s, concurrency cap, already **active** slot on this server, not TEE, etc.). **No DB writes**, no purges.
  - **Preferred method:** `GET /servers/{server_id}/maintenance/preflight` — avoids overloading **HTTP OPTIONS** (often tied to **CORS** in gateways and hard to document as domain logic).
  - **Optional alias:** `OPTIONS /servers/{server_id}/maintenance` with **same auth and JSON response body** as preflight, if product wants literal “OPTIONS preflight”; document that clients must send **miner credentials** on OPTIONS (non-standard for browser CORS but fine for API clients).
  - **Confirm:** `POST /servers/{server_id}/maintenance/confirm` (or **`PUT /servers/{server_id}/maintenance`** if you prefer idempotent “enter state”) — **re-runs** all checks (must match preflight outcome unless state changed between calls), then **commits** maintenance slot and runs **auto-purge**. If preflight was **eligible** but state changed before confirm, confirm may **409**; clients should **re-preflight**.
- **TEE + ownership:** Declare only for `Server.is_tee == true` and `server_id` + `HOTKEY_HEADER` passing existing ownership checks (`check_server_ownership`).
- **Sole-survivor rule (product choice — document in PR):**
  - **Option A (recommended default):** If **any** active instance on that server is the **only** `active` instance globally for its `chute_id`, **declare fails** with **409** and a JSON body listing blocking `{ chute_id, instance_id? }`. **No auto-terminate** in that case (network keeps last copy until another miner scales up). **Auto-bounty on deny** remains **deferred**.
  - **Option B:** Allow declare even with sole survivors; **auto-terminate** must run with **`valid_termination = true`** and **explicit skip** of the last-instance `compute_multiplier` / bounty penalty for **each** purge in that batch (same semantics as a protected manual delete). This risks **capacity gaps** unless combined with strong autoscaler / bounty policy—use only if intentional.
- **Instance → server at delete time (forward-compatible):** **Do not** use `instances.host == servers.ip` as the **primary** link. That join is **consistent with today’s one-IP-per-server TEE rule** but will become **ambiguous** when **planned** multi-server-per-IP TEE exists. **Primary** resolution: **`Instance` → `instance_nodes` → `Node` → `Server`** (`nodes.server_id` → `servers.server_id`). For logging, policy, and any “logical server” checks, treat **`(servers.miner_hotkey, servers.name)`** as the stable human-facing identity (unique per miner today). Implementation detail: if an instance could ever attach to nodes from two servers, define deterministic precedence (e.g. reject or use declaring server’s nodes only)—expected case is one server per instance.
- **Boot completion → `Server` row (current vs planned):** **Now:** use **request IP → `servers.ip`** (matching existing boot attestation behavior; no server name in boot params today). **Later:** boot attestation gains **server name** (or `server_id`) in params; switch the helper above to prefer **name + hotkey** (or `server_id`) and treat IP as secondary. This spec requires: apply **`last_maintenance_completed_at` / active-slot clear** only on the **intended** `servers` row—**today** that is implied by unique IP per server; **tomorrow** that requires explicit identity from the client.
- **Auto-terminate on confirm (primary path):** After **successful** confirm (DB commit sets maintenance slot), **enumerate all instances** on that server via **Instance → Node → Server** (not host/IP as primary), then for each instance invoke shared **purge** logic (as watchtower does): delete `instances` row, update `instance_audit` with `valid_termination = true`, `deletion_reason` e.g. `tee maintenance`, fail jobs, `notify_deleted`, `invalidate_instance_cache`, etc. **Order:** persist slot **before** purges so concurrent logic can see maintenance. **Performance:** many instances may require **sequential async** purges or a **background task**; document whether confirm **HTTP** waits for all purges to finish or returns after scheduling (prefer **wait** for small N, **task** for large N with idempotent retry on failure).
- **Last-instance / bounty on auto-purge:** For each purged instance, apply the **same** rules as a protected manual delete: **skip** the last-instance `compute_multiplier` / bounty slash when policy says maintenance teardown is **valid** (recommended: **always skip** for purges in this flow—confirm in review). Implement by **extracting** shared logic from `delete_instance` into a callable used by both **miner delete** and **maintenance purge**, or by extending `purge()` to accept flags / hooks for maintenance mode—avoid duplicating three-way SQL.
- **Miner-initiated delete while slot active:** Rare if auto-purge drained the server; if any instance remains (partial failure, race, or future edge case), **`delete_instance`** should still treat **active maintenance + window + deadline** like today (`valid_termination`, same last-instance policy).
- **Successful boot = end of cycle (not “wipe and allow re-declare”):** When **boot attestation** succeeds **for the server row being upgraded** and the attested `measurement_version` is **>=** configured `maintenance_target_measurement_version` (semver compare consistent with existing `semcomp` usage): set **`last_maintenance_completed_at = now()`**, then **clear only the active-slot fields** (`maintenance_declared_at`, `maintenance_deadline_at`, and any ephemeral confirm metadata)—**do not** clear **`last_maintenance_completed_at`**. **After that:** any **new** preflight/confirm requires **`now`** inside a **future** admission window **and** the completion timestamp check above (e.g. if the **same** `[window_start, window_end]` is still configured and `now` is still inside it, a completion that landed **inside** that interval blocks a second round; if the admission window has **closed**, entry is already denied regardless of where **`last_maintenance_completed_at`** fell).
- **Explicit non-goals (this spec):** No `scoring_penalty_multiplier` column; no cron adjusting scores for outdated VMs; no `INSTANCES_QUERY` / miner stats query changes for penalties.

### Identity, per-window limits, and abuse model

- **“One maintenance per server per global window”** is only as strong as **server identity**. `server_id` is often **ephemeral** (e.g. new Kubernetes node UID after reprovision). If the miner **deletes** the `servers` row and re-registers, or wipes storage and gets a **new** `server_id` / **new** `name`, the API sees a **new** server: we **cannot** infer they already consumed a slot on a logically “same” machine unless we add **durable tracking** outside `servers`.
- **Rename / reprovision loop:** A miner could declare maintenance, tear down the VM, re-register under a **new** `server_id` (and possibly a **new** `name`), and be eligible again in the same campaign. **Why this may be weak abuse:**
  - Each cycle implies **real downtime** and **lost compute / earnings** during reprovision and redeploy.
  - Maintenance protection only affects **how deletes are classified** (thrash + last-instance treatment); it does not mint extra rewards. The “profit” is avoiding scoring penalties on churn, which is bounded by how much they actually delete and redeploy.
- **Residual risk:** A miner could seek **more** `valid_termination`-style deletes than intended by policy if they can cheaply rotate server identities. Mitigation is **economic** (outage cost), **per-miner concurrency**, **`last_maintenance_completed_at` per window** (same `server_id` / row), **no miner cancel** after purge, plus optional **product** mitigations below.
- **Optional hardening (follow-up, if needed):** Append-only **`server_maintenance_events`** with columns like `(miner_hotkey, server_name, campaign_id, declared_at)` and a **unique** constraint on `(campaign_id, miner_hotkey, server_name)` — stops **reuse of the same name** in one window after row delete, but **does not** stop a miner who picks a **new name** each time. Stronger binding would need an **immutable** hardware or enrollment identifier (out of scope unless another feature provides it).

---

## API Changes

- **New endpoints** (names illustrative; align with existing `/servers` prefix):
  - `GET /servers/{server_id}/maintenance/preflight` — miner auth, **no side effects**. Response e.g. `{ "eligible": bool, "reasons": [...], "blocking_chute_ids": [...], "current_slots": n, "limit": m, ... }`.
  - **Optional:** `OPTIONS /servers/{server_id}/maintenance` — **same auth** and **same JSON** as preflight (document CORS interaction if applicable).
  - `POST /servers/{server_id}/maintenance/confirm` — miner auth; body optional `{}` or `{ "ack": true }` if you want an explicit client ack. **Re-validates** all rules, then sets maintenance columns and **auto-purges**. Returns server id, `maintenance_deadline_at`, **list of `instance_id`s purged** (or async job id), echo of window/grace if desired.
  - **Alternative name:** `PUT /servers/{server_id}/maintenance` for confirm only—pick one in implementation and document.
  - **No** `DELETE /servers/.../maintenance` for miners.
  - `GET /servers/maintenance/policy` or `GET /servers/maintenance/window` — **read-only** global JSON: window bounds, grace hours, target measurement version, **`maintenance_max_concurrent_servers_per_miner`**, miner’s **current active slot count** (for UX). No secrets.
- **Schema changes (`servers` table):** Add nullable columns, e.g.:
  - `maintenance_declared_at` (timestamptz, nullable) — set at **confirm**
  - `maintenance_deadline_at` (timestamptz, nullable) — `now + grace` at confirm
  - **`last_maintenance_completed_at` (timestamptz, nullable)** — set **only** on **successful** boot attestation that ends the cycle; **never** cleared by normal flow (survives until next window config / migration / operator action if ever needed)
  - optionally `maintenance_target_measurement_version` (text, nullable) — copy from settings at confirm time for audit
  - **Optional (audit):** `maintenance_confirmed_at` (timestamptz, nullable) — only if you want an explicit “first confirm time” for support/metrics; **not** needed for abuse prevention: after the first **confirm**, instances on that server are **purged**; if the host **never** reboots and rejoins, there are **no** instances there to **auto-purge** again, so a second **confirm** in the same admission window is largely a **no-op** for purge volume (it may still refresh slot timestamps—implementation can **409** if an active slot already exists, or allow harmless retry).
- **Migrations:** New timestamped file under `api/migrations/` adding the above columns; keep `api/server/schemas.py` `Server` model in sync (this repo holds `Server` in that module, not `orms.py`—follow local convention).

---

## Goal

Success = a miner can **preflight** then **confirm** maintenance **only during the configured global window**, subject to **`last_maintenance_completed_at`** (no second successful cycle for that server in the **same** window bounds), **sole-survivor policy** (Option A or B above), and **per-miner concurrent-slot limit** (default **one** server at a time); on successful **confirm** the API **auto-purges** with **`valid_termination`** and the **agreed last-instance policy**; **successful boot** sets **`last_maintenance_completed_at`** and clears **active** slot fields on the **correct** server row; after **deadline expiry** without successful boot, active fields clear **without** setting completion (retry allowed next preflight if window still open). **No** miner **DELETE**. **No** scoring / metasync changes for outdated versions in this iteration.

Testable criteria:

- Migration applies cleanly; `Server` ORM matches DB.
- Preflight returns **`eligible: false`** when **`last_maintenance_completed_at`** lies in the current configured window, when outside window, sole-survivor (Option A), or at concurrency cap; confirm returns **403/409** consistently.
- After successful **confirm**, **all** targeted instances are **gone** from `instances` (or async job completes reliably), `instance_audit` shows **`valid_termination`** and maintenance reason, caches invalidated; **no** spurious thrash on miner redeploy after upgrade.
- Optional: miner `delete_instance` under active slot still correct if any instance left.
- Boot attestation success sets **`last_maintenance_completed_at`** and clears **active** slot columns only; subsequent preflight in **same** window returns **ineligible**.
- `GET` policy endpoint returns expected shape when window open/closed.
- **Preflight** and **confirm** agree when state is unchanged; after state change, confirm may fail until re-preflight.

---

## Constraints

- Follow [AGENT.md](../../AGENT.md): **no new dependencies**; config via `api.config.settings` / env; **async** handlers; **Ruff** clean; add **tests** where behavior is non-trivial.
- **Do not** add `scoring_penalty_multiplier`, penalty cron, or `INSTANCES_QUERY` edits in this task.
- **Do not** hardcode window times or measurement targets in Python—**settings only**.
- Keep changes **focused**: prefer small helpers in `api/server/` (e.g. `util.py` or `service.py`) over cross-cutting refactors.

---

## Output Format

1. `api/migrations/YYYYMMDDHHMMSS_server_maintenance.sql` — `ALTER TABLE servers ADD COLUMN …` (+ indexes only if query patterns justify).
2. `api/server/schemas.py` — New columns on `Server` SQLAlchemy model.
3. `api/config.py` (or settings model) — Fields for window start/end, grace hours, target measurement version, **`maintenance_max_concurrent_servers_per_miner`** (default **1**, minimum **1** in validation), and optional feature flag.
4. `api/server/router.py` — New routes; reuse `get_current_user` / hotkey patterns from existing server routes.
5. `api/server/service.py` (or new helper module) — `preflight_maintenance` / `confirm_maintenance` (include **`completed_at` vs window** check); **deadline expiry** nulls active fields only; **`resolve_server_for_maintenance_boot_completion`** (IP-based **now**, name-based **later**); hook from boot attestation success to set **`last_maintenance_completed_at`** + clear active fields when measurement OK.
6. `api/instance/util.py` or `watchtower.py` or small `api/instance/maintenance_purge.py` — **Shared** “maintenance purge one instance” used by **confirm** batch; wraps or extends `purge` with **`valid_termination=True`**, correct `deletion_reason`, and **last-instance policy** (refactor `delete_instance` penalty block into shared helper where possible).
7. `api/instance/router.py` — Keep protected `delete_instance` branch for edge cases (optional if auto-purge is exhaustive).
8. `tests/unit/` (and/or integration) — Second preflight/confirm **denied** after boot success in same window; **allowed** after window config rolls forward or `last_maintenance_completed_at` before new `window_start`; deadline expiry **without** boot does **not** set completion; **Instance → Node → Server** resolution.

---

## Failure Conditions

- Maintenance protection applies **outside** the global window or **after** `maintenance_deadline_at`.
- **Confirm** returns **success** when **any** blocking sole-survivor chute exists (**Option A**), when miner already holds **≥ limit** concurrent active slots, or when outside the global window (**confirm** must **re-validate** every check; outcomes must match preflight unless state legitimately changed).
- A miner-facing **`DELETE`** exists that **clears** maintenance (must **not** ship).
- Auto-purge uses **`valid_termination = false`** or omits last-instance protection → **thrash** or **wrong scoring** on redeploy.
- Instances **remain routable** after successful **confirm** (purge incomplete / wrong server scope).
- Boot success **omits** `last_maintenance_completed_at` or **wipes** it on every boot (would allow **re-declare** abuse in one window).
- Preflight/confirm allowed when **`last_maintenance_completed_at`** is inside the **current** configured window.
- Boot completion runs on **wrong** `servers` row (**acceptable risk today** with one IP per server; **unacceptable** once shared-IP ships without boot param change) or runs without measurement check.
- Delete / enumerate path uses **IP-only** as **primary** correlation instead of **Instance → Node → Server** (breaks **planned** shared-IP TEE; weak even today).
- Schema drift: migration applied but `Server` model missing columns (or reverse).
- **Any** dependency added without explicit approval.
- Scoring / metasync penalty code added despite scope.

---

## Rollout Notes

- **Config:** Document new env vars in PR description and optionally in `dev/dev.md` / README snippet (e.g. `TEE_MAINTENANCE_WINDOW_START`, `TEE_MAINTENANCE_WINDOW_END`, `TEE_MAINTENANCE_GRACE_HOURS`, `TEE_MAINTENANCE_TARGET_MEASUREMENT_VERSION`, `TEE_MAINTENANCE_MAX_CONCURRENT_SERVERS_PER_MINER` default **1**—final names follow `settings` naming).
- **Deploy order:** Migrate DB → deploy API with new settings; until window envs are set, feature should be **off** or **outside window** (define default: e.g. null window = declare always 403).
- **Operational:** Miners script **`GET …/preflight` → `POST …/confirm`**; `GET …/policy` for global window/limit; validators set window globally per campaign.

---

## Follow-ups (not this spec)

- Auto-bounty when declare is blocked (sole survivor).
- `scoring_penalty_multiplier` + cron + `INSTANCES_QUERY` / miner stats alignment for outdated VMs post-window.
- **`server_maintenance_events`** (or similar) if per-campaign limits must survive server row deletion or **reuse of `server_name`**.
- **Boot attestation API:** add **`vm_name` / server name** (or `server_id`) to request params; update **`resolve_server_for_maintenance_boot_completion`** to use it—required for **NAT / multi-server-per-IP** TEE (no ambiguous IP mapping).

