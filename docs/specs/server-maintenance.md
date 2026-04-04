# Feature Spec: TEE server maintenance window

Use the sections **Goal**, **Constraints**, **Output Format**, and **Failure Conditions** as a **Prompt Contract** for this task (see [AGENT.md](../../AGENT.md) at repo root).

**Date**: 2026-04-02 (revised 2026-04-03)  
**Status**: draft (refined)

**Scope note (this iteration):** Implement **maintenance window + preflight/confirm + auto-purge + boot completion** (`Server.version` updated, `maintenance_pending_window_id` cleared on success). **Out of scope:** `scoring_penalty_multiplier`, cron-based penalties for outdated `measurement_version`, changes to `metasync` / `INSTANCES_QUERY`, and deferred `valid_termination` (see **Acknowledged abuse vector** below).

---

## Context

Miners upgrading a TEE host need to **remove instances** from routing before reboot. Today, **miner-initiated** `delete_instance` can trigger **thrash** (no `valid_termination`) and **last-instance** scoring penalties (`api/instance/router.py`). This spec adds a **validator-controlled global window**, a **preflight** endpoint (see below), and a **confirm** step. On **successful confirm**, the platform **automatically terminates all instances** on that server (validator-driven eviction) using the same **purge** machinery as **watchtower** (`watchtower.purge` / `purge_and_notify`), with **`valid_termination = true`** and a dedicated **`deletion_reason`**, so routing and caches update **proactively**—the miner does not rely on manual deletes to drain the box. **Policy** must align **last-instance** / bounty handling on these purges with the maintenance story (see below).

- **Packages affected**: `api` (primary)
- **Key files**:
  - `api/server/router.py`, `api/server/service.py`, `api/server/schemas.py` — server model, preflight + confirm maintenance, boot attestation **completion** handling (`Server.version` updated, `maintenance_pending_window_id` cleared)
  - `api/config.py` (or equivalent settings module) — `tee_maintenance_max_miner_concurrency`; **not** primary store for window bounds / target (those live in **`tee_upgrade_windows`**)
  - `watchtower.py` — `purge` / `purge_and_notify` (or factored shared helper): reuse for maintenance-initiated teardown (`valid_termination=True`, notifications, cache invalidation)
  - `api/instance/router.py` — **Optional** fallback: if any code path still allows miner `delete_instance` while maintenance is active, keep protected behavior; primary drain is **confirm → auto-purge**
  - `api/instance/util.py`, `api/node/schemas.py` — correlate `Instance` → `Node` → `Server` (forward-compatible with **planned** shared-IP TEE; see **TEE addressing: today vs planned** below)
  - `api/migrations/*.sql` — new table **`tee_upgrade_windows`**; new columns on **`servers`** (`maintenance_pending_window_id`, `version`)
  - `api/constants.py` — thrash constants (read-only context; no change required unless tests need it)
- **Dependencies**: Existing FastAPI, SQLAlchemy async sessions, `get_current_user` / miner auth patterns for server routes.

### TEE addressing: today vs planned

- **Today (production TEE):** The platform often has **one public IP per server**, but **IP alone is not sufficient** for maintenance completion once **multiple logical servers** or **IP changes** exist. **`Instance` → `Node` → `Server`** remains the primary path for instance teardown.
- **Planned / shared-IP:** **Multiple TEE servers may share one public IP** (e.g. NAT / shared egress). **Boot attestation must not rely on IP alone** to find the `servers` row for maintenance completion.
- **This spec:** Use **`(miner_hotkey, server name)`** for server identity wherever the API must pick a single **`Server`** row (aligned with **`servers` unique `(miner_hotkey, name)`** in [`api/server/schemas.py`](../../api/server/schemas.py)).
- **Boot attestation params (already present):** [`BootAttestationArgs`](../../api/server/schemas.py) includes **`vm_name`** and **`miner_hotkey`**. **Contract:** **`vm_name` must match `Server.name`** for that miner's registered server (same string the miner used at registration / LUKS linkage). **Maintenance completion** on successful boot must **resolve the `Server` row by `(args.miner_hotkey, args.vm_name)` → `servers.name`** (primary). Optionally **cross-check** request IP against `servers.ip` when present (log or soft-validate); do **not** use IP as the only key when **`vm_name`** is available.
- **First boot / no `Server` row yet:** Boot attestation can run **before** server registration. If **no** row matches **`(miner_hotkey, vm_name)`**, **skip** maintenance completion and `version` update (no-op): there is no **`servers`** row to update, and **preflight/confirm** only apply to **existing** servers anyway. After **`register_server`** (or equivalent), subsequent boots find the row and can clear maintenance if a slot was set.
- **Implementation:** Centralise in **`resolve_server_for_maintenance_boot_completion(db, miner_hotkey, vm_name, request_ip?)`** (or fold into `process_boot_attestation`): lookup by **hotkey + name**, return **`None`** if absent, then run `version` update + maintenance fields only when a **`Server`** exists.

---

## Design Decisions

- **No human admin (miner-facing):** Allow/deny is **fully automatic** inside the API for miners. **Validators** control windows by **rows in the database** (see **`tee_upgrade_windows`** below)—no miner-facing approve/deny routes. **This iteration:** no HTTP admin API for inserting windows (use migration seed, SQL runbook, or internal tooling); follow-up can add operator routes.
- **Global window = admission window (DB-backed):** The **active** upgrade window is the row in **`tee_upgrade_windows`** for which **`upgrade_window_start <= now() <= upgrade_window_end`** (see **Resolving the active window**). If **no** row qualifies, **deny** new **preflight** / **confirm** (**403** / **404**—document the choice). Servers that **confirmed** before the window closes may **finish** (reboot, boot attestation) **after** `upgrade_window_end`—we do not revoke an in-flight slot. **No new admits after close:** until validators **insert** a new window row (or extend an existing row's end time). Past rows remain for **history and audit**.
- **Why a table instead of env-only:** Environment variables give **no durable history** and encourage **silent overwrites**. A table yields **one row per coordinated target** (`target_measurement_version`), explicit **`upgrade_window_start` / `upgrade_window_end`**, **`created_at`**, and a full **audit trail** of past cutovers. **`GET …/policy`** (and internal checks) **read the active row from the DB**, optionally **cached** (short TTL Redis/in-process) to avoid hitting the DB on every preflight; cache must **invalidate or expire** quickly enough that window changes take effect promptly (or invalidate on write when an admin API exists).
- **Rollout identity = `tee_upgrade_windows` row:** **`target_measurement_version`** on the row is the **logical "which upgrade"** string (normalised once at insert). **`id`** distinguishes rows; enforce **`UNIQUE (target_measurement_version)`** so there is **at most one window definition per target** (adjust if you ever need a rare re-run for the same target—then drop uniqueness and key completion by **`id`** only).
- **Already at or above target (no pointless purge / anti-re-entry):** **Preflight** / **confirm** **deny** if `semcomp(server.version, active_window.target_measurement_version) >= 0`. **Rationale:** Maintenance exists to move hosts **onto** the mandated image; purging when already compliant would only create churn and scoring noise. **Source of truth:** **`Server.version`** — updated on **every successful boot attestation** (see boot completion below). **If `version` is `None`** (no boot recorded yet), **do not** treat as "already at target." This check also **prevents re-entry** after a completed maintenance cycle: once a server boots with `version >= target`, the version check blocks subsequent preflight/confirm for the same window's target.
- **No per-server deadline / grace period:** There is **no** `maintenance_deadline_at` or `grace_hours`. The server stays "in maintenance" (`maintenance_pending_window_id` set) until boot completion succeeds or the window closes. This is simpler and avoids false precision—the window's own `upgrade_window_end` provides the outer bound for the rollout.
- **Per-miner concurrency (validator capacity knob):** A miner may have at most **`tee_maintenance_max_miner_concurrency`** servers in an **active** maintenance state at once (**default `1`**). **Active** means **`maintenance_pending_window_id IS NOT NULL`** and the referenced window is the **currently active** one (stale slots for closed windows are lazily cleared—see below). Before accepting **confirm**, **count** distinct `servers` rows for that `miner_hotkey` meeting that condition; if **count ≥ limit**, return **409** with a clear message (and `current_slots` / `limit` in JSON). An active slot **ends** when **successful boot** clears `maintenance_pending_window_id`, or is **lazily cleared** when the referenced window is no longer active. **No miner `DELETE`** to clear maintenance. **Operator-only** override (if ever needed) is out of scope here.
- **Two-step flow (preflight + confirm):**
  - **Preflight:** `GET /servers/{server_id}/maintenance/preflight` — read-only check whether **confirm** would succeed **right now**—same auth as confirm (miner + `server_id`). Returns **`eligible: true`** or **`eligible: false`** with structured reasons (no active DB window, **already at or above active target** via `Server.version`, sole-survivor `chute_id`s, concurrency cap, already **active** slot on this server, not TEE, etc.). When the server has a **pending maintenance** (`maintenance_pending_window_id` set), includes a structured entry with `current_version` and `target_version` so the miner can see the gap. **No DB writes**, no purges (reads **active window** from DB or cache).
  - **Confirm:** `PUT /servers/{server_id}/maintenance` — **re-runs** all checks (must match preflight outcome unless state changed between calls), then **commits** maintenance slot and runs **auto-purge**. Idempotent **enter maintenance** semantics are acceptable for **PUT**. If preflight was **eligible** but state changed before confirm, confirm may **409**; clients should **re-preflight**.
- **TEE + ownership:** **Preflight / confirm** only for `Server.is_tee == true` and `server_id` + `HOTKEY_HEADER` passing existing ownership checks (`check_server_ownership`).
- **Sole-survivor rule (fixed policy):** If **any** active instance on that server is the **only** `active` instance globally for its `chute_id`, **`PUT …/maintenance` fails** with **409** and a JSON body listing blocking `{ chute_id, instance_id? }`. **Never** auto-terminate the globally last instance via maintenance: the network keeps that copy until another miner scales up. **Preflight** surfaces the same blocking set. **Auto-bounty on deny** remains **deferred** (follow-up).
- **Instance → server at delete time (forward-compatible):** **Do not** use `instances.host == servers.ip` as the **primary** link. That join is **consistent with today's one-IP-per-server TEE rule** but will become **ambiguous** when **planned** multi-server-per-IP TEE exists. **Primary** resolution: **`Instance` → `instance_nodes` → `Node` → `Server`** (`nodes.server_id` → `servers.server_id`). **The platform does not support** an instance attached to nodes belonging to **more than one** `server_id`; no cross-server merge or precedence rules are required. For logging, policy, and any "logical server" checks, treat **`(servers.miner_hotkey, servers.name)`** as the stable human-facing identity (unique per miner today).
- **Boot completion → `Server` row:** Use **`BootAttestationArgs.miner_hotkey`** + **`BootAttestationArgs.vm_name`** → **`Server`** where **`servers.name = vm_name`** (and **`servers.miner_hotkey`** matches). **If no row:** first-boot / pre-registration path—**no** maintenance completion, **no** `version` update. **Always** set **`Server.version = measurement_version`** from the matched measurement config on successful boot (regardless of maintenance state). If **`maintenance_pending_window_id`** is set and `semcomp(measurement_version, pending_window.target_measurement_version) >= 0`: **clear** `maintenance_pending_window_id = None`. If measurement is **below** target: log warning, leave `maintenance_pending_window_id` set (miner must try again). The updated `version` prevents re-entry via the "already at target" check.
- **Auto-terminate on confirm (primary path):** After **successful** confirm (DB commit sets `maintenance_pending_window_id`), **enumerate all instances** on that server via **Instance → Node → Server** (not host/IP as primary), then for each instance invoke shared **purge** logic (as watchtower does): delete `instances` row, update `instance_audit` with `valid_termination = true`, `deletion_reason` e.g. `tee maintenance`, fail jobs, `notify_deleted`, `invalidate_instance_cache`, etc. **Order:** persist slot **before** purges so concurrent logic can see maintenance. **Performance:** many instances may require **sequential async** purges or a **background task**; document whether confirm **HTTP** waits for all purges to finish or returns after scheduling (prefer **wait** for small N, **task** for large N with idempotent retry on failure).
- **Last-instance / bounty on auto-purge:** Because **confirm** is **denied** when a sole survivor exists on the server, the maintenance purge batch **must not** include a globally last instance for any `chute_id`. For each instance purged in this flow, use **`valid_termination = true`**, maintenance **`deletion_reason`**, and shared **`purge`** machinery; last-instance bounty / multiplier slash **does not apply** to these rows (they are not last-global by construction). Implement via a shared helper or **`purge()`** flags—avoid duplicating `delete_instance` penalty logic.
- **Miner-initiated delete while slot active:** Rare if auto-purge drained the server; if any instance remains (partial failure, race, or future edge case), **`delete_instance`** should still treat **active maintenance slot** (`maintenance_pending_window_id IS NOT NULL`) with `valid_termination`, same last-instance policy.
- **Stale pending slot cleanup:** If the upgrade window closes and `maintenance_pending_window_id` still points to it (miner never completed), the slot is stale. Lazy clear: when preflight/confirm encounters `maintenance_pending_window_id` referencing a window that is no longer active, treat as cleared (or clear it on read). This allows the miner to enter a new window without manual intervention.
- **Pending upgrade visibility (miner UX):** When a server has `maintenance_pending_window_id` set (especially after a boot that didn't reach the target), surface the version gap clearly: **preflight** includes a structured `maintenance_pending` reason with `current_version` and `target_version`; **policy** endpoint includes a `pending_servers` list for the calling miner; **`GET /servers/{server_id}`** includes `version` and maintenance status.
- **Explicit non-goals (this spec):** No `scoring_penalty_multiplier` column; no cron adjusting scores for outdated VMs; no `INSTANCES_QUERY` / miner stats query changes for penalties.

### Acknowledged abuse vector (v1 accept)

Auto-purge grants `valid_termination=True` at confirm time, **before** the upgrade is proven. A miner could confirm maintenance purely to get a penalty-free purge of all their instances, then never actually upgrade. **Mitigations:** economic cost (zero earnings while offline), sole-survivor check (can't kill last global instance), validator-controlled windows (can't enter at will), per-miner concurrency cap. **Closed fully by** the planned **scoring penalty follow-up** (`scoring_penalty_multiplier` + cron for outdated VMs post-window). **Deferred option (stronger):** purge with `valid_termination=False` at confirm, upgrade to `True` on successful boot completion.

### Identity, per-window limits, and abuse model

- **"One maintenance per server per global window"** is enforced by the **`Server.version >= target`** check: once a server completes maintenance (boots with new version), `version` is updated and the server cannot re-enter for the same target. `server_id` is often **ephemeral** (e.g. new Kubernetes node UID after reprovision). If the miner **deletes** the `servers` row and re-registers, or wipes storage and gets a **new** `server_id` / **new** `name`, the API sees a **new** server (with `version = NULL`): we **cannot** infer they already consumed a slot on a logically "same" machine unless we add **durable tracking** outside `servers`.
- **Rename / reprovision loop:** A miner could enter maintenance, tear down the VM, re-register under a **new** `server_id` (and possibly a **new** `name`), and be eligible again for the **same** configured target. **Why this may be weak abuse:**
  - Each cycle implies **real downtime** and **lost compute / earnings** during reprovision and redeploy.
  - Maintenance protection only affects **how deletes are classified** (thrash + last-instance treatment); it does not mint extra rewards. The "profit" is avoiding scoring penalties on churn, which is bounded by how much they actually delete and redeploy.
- **Residual risk:** A miner could seek **more** `valid_termination`-style deletes than intended by policy if they can cheaply rotate server identities. Mitigation is **economic** (outage cost), **per-miner concurrency**, **no miner cancel** after purge, plus optional **product** mitigations below.
- **Optional hardening (follow-up, if needed):** Append-only **`server_maintenance_events`** with columns like `(miner_hotkey, server_name, upgrade_window_id, confirmed_at)` and a **unique** constraint on `(upgrade_window_id, miner_hotkey, server_name)` — stops **reuse of the same name** in one rollout after row delete, but **does not** stop a miner who picks a **new name** each time. Stronger binding would need an **immutable** hardware or enrollment identifier (out of scope unless another feature provides it).

### Table `tee_upgrade_windows` (historical record, one row per target)

| Column | Type | Notes |
|--------|------|--------|
| **`id`** | bigserial PK | Stable row identity for FKs from **`servers`**. |
| **`upgrade_window_start`** | timestamptz | Admission opens (new **preflight** / **confirm** allowed). |
| **`upgrade_window_end`** | timestamptz | Admission closes for **new** entries; in-flight slots may still finish. |
| **`target_measurement_version`** | text | Minimum attested measurement for this cutover; **normalised** on insert. **`UNIQUE`** recommended (one row per target version). |
| **`created_at`** | timestamptz | When the row was inserted (audit). Default `now()`. |

**Operational pattern:** `INSERT` a new row when shipping a **new** mandated image line; **UPDATE** `upgrade_window_end` to **now** (or past) to **end** admits for that cutover before opening the next. Old rows **stay** in the table as history.

### `servers` table — new columns

| Column | Type | Notes |
|--------|------|--------|
| **`maintenance_pending_window_id`** | bigint, nullable, FK → `tee_upgrade_windows.id` | Set at **confirm**; "in maintenance" signal. Cleared on successful boot completion (version >= target) or lazily when the referenced window is no longer active. |
| **`version`** | text, nullable | Current attested measurement version. Updated on **every** successful boot attestation (regardless of maintenance). Used by the "already at target" check and exposed in server metadata for miners. |

### Resolving the "active" window

**Definition:** The **active** window is the single row (if any) such that **`upgrade_window_start <= now() <= upgrade_window_end`**. If **multiple** rows overlap (operator error), implementation must pick a **deterministic** rule (e.g. **highest `id`**, or **latest `created_at`**) and **log a warning**; validators should avoid overlaps.

**No active row:** No new **preflight** / **confirm**; feature is "closed" until a new row qualifies.

**Caching:** Load active row via a small helper used by preflight, confirm, policy, and boot completion; cache the result for a **short TTL** (or invalidate on writes) so policy GETs do not hammer the DB.

### Rollout identity, single artifact, and what belongs in maintenance

**Single published VM artifact:** The release pipeline exposes **only the latest** VM image. Once **0.3.1** is published, miners **cannot** fetch **0.3.0**. Any miner who **starts** an upgrade after that point is on the **current** image line.

**What maintenance windows are for (policy):** Use **coordinated admission windows** primarily for **major / minor** (or **breaking / validator-mandated**) TEE image moves. **Patch** releases: miners upgrade **on their own schedule** **without** this API—**no** maintenance-scoped protections for that path in this spec.

**Problem (minor bump mid-window):** A row exists with target **T0**; the pipeline publishes a newer image and the old one is **gone**.

**Operator response (recommended — end and replace, no overlap):**

1. **End** the current row's admission: set **`upgrade_window_end`** to **now** (or past) on that row so it is no longer "active."
2. **`INSERT`** a **new** row with **`target_measurement_version = T1`**, new **`upgrade_window_start` / `upgrade_window_end`**, and **`created_at`**.
3. Miners who **completed** the old cutover have `version >= T0` → if `T1 > T0`, the "already at target" check allows them to enter the **new** window (version < T1). Miners **in flight** (confirmed under old row, not yet booted): at boot, `Server.version` is updated; if it meets the **new** target too, the stale `maintenance_pending_window_id` (pointing at the old row) is lazily cleared on next preflight. If not, they remain pending until they boot with a sufficient version.

**Overlapping concurrent windows:** **Out of scope** in v1—operators should **not** insert overlapping `[start, end]` ranges; if they do, deterministic resolution + warning applies.

**Implementation note:** **`GET …/policy`** returns the **active** row's **`id`**, bounds, **`target_measurement_version`**, **`tee_maintenance_max_miner_concurrency`** from settings, the miner's active slot count, and a **`pending_servers`** list for the calling miner.

---

## API Changes

- **New endpoints** (names illustrative; align with existing `/servers` prefix):
  - `GET /servers/{server_id}/maintenance/preflight` — miner auth, **no side effects**. Response e.g. `{ "eligible": bool, "reasons": [...], "blocking_chute_ids": [...], "current_slots": n, "limit": m, ... }`. When the server has pending maintenance, includes structured `maintenance_pending` reason with `current_version` and `target_version`.
  - `PUT /servers/{server_id}/maintenance` — miner auth; body optional `{}` or `{ "ack": true }` if you want an explicit client ack. **Re-validates** all rules, then sets `maintenance_pending_window_id` and **auto-purges**. Returns server id, **list of `instance_id`s purged** (or async job id), echo of window if desired. On failure returns 403/409 with structured error matching preflight reasons.
  - **No** `DELETE /servers/.../maintenance` for miners.
  - `GET /servers/maintenance/policy` — **read-only** global JSON: **active** window **`id`**, **`upgrade_window_start` / `upgrade_window_end`**, **`target_measurement_version`**, **`tee_maintenance_max_miner_concurrency`** (from settings), miner's **current active slot count**, and **`pending_servers`** list (servers with `maintenance_pending_window_id` set, each with `server_id`, `name`, `version`, `target_version`). Served from DB (via cache). No secrets.
- **Updated endpoints:**
  - `GET /servers/{server_id}` — include **`version`** and **maintenance status** (`maintenance_pending_window_id`, resolved `target_version` when pending) in the response.
- **Schema changes — new table `tee_upgrade_windows`:** As in the table above; add **`UNIQUE (target_measurement_version)`** if policy is strictly one row per target.
- **Schema changes (`servers` table):** Add 2 nullable columns:
  - **`maintenance_pending_window_id` (FK → `tee_upgrade_windows.id`, nullable)** — set to the **active** window's **`id`** at **confirm**; cleared on successful boot completion or lazily when the referenced window is no longer active.
  - **`version` (text, nullable)** — current attested measurement version; updated on every successful boot attestation; used by "already at target" check and exposed in server metadata.
- **Migrations:** New timestamped SQL under `api/migrations/` creating **`tee_upgrade_windows`** and altering **`servers`**; keep `api/server/schemas.py` models in sync (this repo holds `Server` in that module, not `orms.py`—follow local convention).

---

## Goal

Success = a miner can **preflight** then **confirm** only when a **DB-backed active `tee_upgrade_windows` row** exists and `now` is inside **`[upgrade_window_start, upgrade_window_end]`**, and **not** when **already at or above** that row's **`target_measurement_version`** (checked via `Server.version`). Subject to **sole-survivor rule** (deny **409**—never purge the globally last instance) and **per-miner concurrent-slot limit** (default **one** server at a time); on successful **confirm** the API **auto-purges** with **`valid_termination`** (no globally last instances in batch—see sole-survivor rule); **successful boot** updates **`Server.version`** and clears **`maintenance_pending_window_id`** on the **correct** server row (resolved by `miner_hotkey` + `vm_name`); stale pending slots (window closed, never completed) are **lazily cleared**. Validators add **new table rows** (and end old rows) for each coordinated cutover—**history** remains in **`tee_upgrade_windows`**. **No** miner **DELETE**. **No** scoring / metasync changes for outdated versions in this iteration.

Testable criteria:

- Migration applies cleanly; `Server` ORM matches DB.
- Preflight returns **`eligible: false`** when **no active window row**, **already >= active target** (via `Server.version`), outside **`[start, end]`**, **sole-survivor** blocking any instance, at concurrency cap, or already in maintenance; confirm returns **403/409** consistently.
- After successful **confirm**, **all** targeted instances are **gone** from `instances` (or async job completes reliably), `instance_audit` shows **`valid_termination`** and maintenance reason, caches invalidated; **no** spurious thrash on miner redeploy after upgrade.
- Optional: miner `delete_instance` under active slot still correct if any instance left.
- Boot attestation success updates **`Server.version`** and clears **`maintenance_pending_window_id`** when measurement >= target; subsequent preflight returns **ineligible** (version >= target).
- Boot attestation with measurement **below** target still updates `Server.version` but leaves `maintenance_pending_window_id` set; preflight/policy surfaces the version gap.
- `GET` policy endpoint returns expected shape when window open/closed, includes `pending_servers`.
- **Preflight** and **confirm** agree when state is unchanged; after state change, confirm may fail until re-preflight.

---

## Constraints

- Follow [AGENT.md](../../AGENT.md): **no new dependencies**; **window bounds and targets** live in **`tee_upgrade_windows`** (not env); **settings** for concurrency limit; **async** handlers; **Ruff** clean; add **tests** where behavior is non-trivial.
- **Do not** add `scoring_penalty_multiplier`, penalty cron, or `INSTANCES_QUERY` edits in this task.
- **Do not** hardcode window times or target versions in application code—load from the **DB** (active row) or documented migration seeds.
- Keep changes **focused**: prefer small helpers in `api/server/` (e.g. `util.py` or `service.py`) over cross-cutting refactors.

---

## Output Format

1. `api/migrations/YYYYMMDDHHMMSS_server_maintenance.sql` — `CREATE TABLE tee_upgrade_windows (...)`; indexes to resolve **active** row quickly (e.g. on `(upgrade_window_start, upgrade_window_end)` or as justified by queries); `ALTER TABLE servers ADD COLUMN …` / FKs.
2. `api/server/schemas.py` — `TeeUpgradeWindow` (or equivalent) model + new columns (`maintenance_pending_window_id`, `version`) on **`Server`**.
3. `api/config.py` (or settings model) — **`tee_maintenance_max_miner_concurrency`**; **not** window start/end/target (those are DB rows).
4. `api/server/router.py` — New routes (policy, preflight, confirm); update `GET /servers/{server_id}` to include `version` and maintenance status; reuse `get_current_user` / hotkey patterns from existing server routes.
5. `api/server/service.py` (or new helper module) — **`get_active_upgrade_window()`** (DB + cache); `preflight_maintenance` / `confirm_maintenance` (already-at-target via `Server.version`, set **`maintenance_pending_window_id`** at confirm); **stale slot lazy clear** when referenced window is no longer active; **`resolve_server_for_maintenance_boot_completion(db, miner_hotkey, vm_name, …)`** using **`BootAttestationArgs`**; extend **`process_boot_attestation`** (or call hook after success) to set **`Server.version`** and clear **`maintenance_pending_window_id`** when measurement OK; **no-op** when **no `Server`** row (**first boot**).
6. `api/instance/util.py` or `watchtower.py` or small `api/instance/maintenance_purge.py` — **Shared** "maintenance purge one instance" used by **confirm** batch; wraps or extends `purge` with **`valid_termination=True`** and maintenance **`deletion_reason`** (batch excludes globally last instances—see sole-survivor rule).
7. `api/instance/router.py` — Keep protected `delete_instance` branch for edge cases (optional if auto-purge is exhaustive).
8. `tests/unit/` (and/or integration) — Preflight **denied** when **no row / outside window / already >= target**; **allowed** when eligible; confirm sets `maintenance_pending_window_id` and purges; boot completion updates `version` and clears pending slot; stale slot lazy clear; sole-survivor blocks confirm; **Instance → Node → Server** resolution.

---

## Failure Conditions

- Maintenance protection applies **outside** the global window (preflight/confirm must check window bounds).
- **Confirm** returns **success** when **any** blocking **sole-survivor** `chute_id` exists, when miner already holds **≥ limit** concurrent active slots, or when outside the active DB window (**confirm** must **re-validate** every check; outcomes must match preflight unless state legitimately changed).
- A miner-facing **`DELETE`** exists that **clears** maintenance (must **not** ship).
- Auto-purge uses **`valid_termination = false`** or omits last-instance protection → **thrash** or **wrong scoring** on redeploy.
- Instances **remain routable** after successful **confirm** (purge incomplete / wrong server scope).
- Boot success **omits** `Server.version` update or **wipes** `maintenance_pending_window_id` without checking measurement >= target.
- **Wrongly** allow preflight/confirm when the server is **already >= active target** (pointless purge — `Server.version` check must deny).
- **Confirm** or purge **succeeds** while a **globally sole-survivor** instance for any `chute_id` would be terminated (**must** remain **409** / no purge—see fixed sole-survivor rule).
- Boot completion runs on **wrong** `servers` row (must use **`(miner_hotkey, vm_name)` → `servers.name`**, not **IP-only**) or runs without measurement check.
- Maintenance completion runs when **no** `servers` row exists (**first boot**) and incorrectly mutates state (should **no-op**).
- Delete / enumerate path uses **IP-only** as **primary** correlation instead of **Instance → Node → Server** (breaks **planned** shared-IP TEE; weak even today).
- Schema drift: migration applied but `Server` model missing columns (or reverse).
- **Any** dependency added without explicit approval.
- Scoring / metasync penalty code added despite scope.

---

## Rollout Notes

- **Database:** Document **`tee_upgrade_windows`** and the runbook: **`INSERT`** a row to open a cutover (**`target_measurement_version`**, **`upgrade_window_start` / `upgrade_window_end`**, **`created_at`**); **`UPDATE`** `upgrade_window_end` to end admits; **never** delete old rows if you want history (or archive separately). Optionally document in `dev/dev.md`.
- **Settings:** `TEE_MAINTENANCE_MAX_MINER_CONCURRENCY`—final name follows `settings` naming.
- **Deploy order:** Migrate DB (table + server columns) → deploy API → **`INSERT`** first window row when ready; if **no** row is active for `now`, preflight/confirm deny new entry.
- **Operational:** Miners **`GET …/preflight` → `PUT …/maintenance`**; **`GET …/policy`** reflects the **active DB row** (cached) and shows `pending_servers`. Validators manage **rows**, not env window clocks.

---

## Follow-ups (not this spec)

- Auto-bounty when confirm is blocked (sole survivor).
- `scoring_penalty_multiplier` + cron + `INSTANCES_QUERY` / miner stats alignment for outdated VMs post-window.
- **Deferred `valid_termination`**: purge with `valid_termination=False` at confirm, upgrade to `True` on successful boot completion (closes the "free purge" abuse vector).
- **`server_maintenance_events`** (or similar) if per-rollout limits must survive server row deletion or **reuse of `server_name`**.
- **Optional:** **`server_id` in `BootAttestationArgs`** if product wants an explicit key beyond **`vm_name`**; **`vm_name` is already required** today ([`BootAttestationArgs`](../../api/server/schemas.py)).
- **Audit view:** `CREATE VIEW v_server_maintenance` joining `servers` → `tee_upgrade_windows` to expose derived columns (`maintenance_declared_at`, `last_maintenance_completed_at`, `target_measurement_version`) without widening the `servers` table.
