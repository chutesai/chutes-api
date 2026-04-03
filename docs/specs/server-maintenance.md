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
  - `api/config.py` (or equivalent settings module) — **optional** `maintenance_max_concurrent_servers_per_miner`, default **grace** if not on window row, cache TTL; **not** primary store for window bounds / target (those live in **`tee_upgrade_windows`**)
  - `watchtower.py` — `purge` / `purge_and_notify` (or factored shared helper): reuse for maintenance-initiated teardown (`valid_termination=True`, notifications, cache invalidation)
  - `api/instance/router.py` — **Optional** fallback: if any code path still allows miner `delete_instance` while maintenance is active, keep protected behavior; primary drain is **confirm → auto-purge**
  - `api/instance/util.py`, `api/node/schemas.py` — correlate `Instance` → `Node` → `Server` (forward-compatible with **planned** shared-IP TEE; see **TEE addressing: today vs planned** above)
  - `api/migrations/*.sql` — new table **`tee_upgrade_windows`**; new columns on **`servers`**
  - `api/constants.py` — thrash constants (read-only context; no change required unless tests need it)
- **Dependencies**: Existing FastAPI, SQLAlchemy async sessions, `get_current_user` / miner auth patterns for server routes.

### TEE addressing: today vs planned

- **Today (production TEE):** The platform often has **one public IP per server**, but **IP alone is not sufficient** for maintenance completion once **multiple logical servers** or **IP changes** exist. **`Instance` → `Node` → `Server`** remains the primary path for instance teardown.
- **Planned / shared-IP:** **Multiple TEE servers may share one public IP** (e.g. NAT / shared egress). **Boot attestation must not rely on IP alone** to find the `servers` row for maintenance completion.
- **This spec:** Use **`(miner_hotkey, server name)`** for server identity wherever the API must pick a single **`Server`** row (aligned with **`servers` unique `(miner_hotkey, name)`** in [`api/server/schemas.py`](../../api/server/schemas.py)).
- **Boot attestation params (already present):** [`BootAttestationArgs`](../../api/server/schemas.py) includes **`vm_name`** and **`miner_hotkey`**. **Contract:** **`vm_name` must match `Server.name`** for that miner’s registered server (same string the miner used at registration / LUKS linkage). **Maintenance completion** on successful boot must **resolve the `Server` row by `(args.miner_hotkey, args.vm_name)` → `servers.name`** (primary). Optionally **cross-check** request IP against `servers.ip` when present (log or soft-validate); do **not** use IP as the only key when **`vm_name`** is available.
- **First boot / no `Server` row yet:** Boot attestation can run **before** server registration. If **no** row matches **`(miner_hotkey, vm_name)`**, **skip** maintenance completion (no-op): there is no **`servers`** row to update, and **preflight/confirm** only apply to **existing** servers anyway. After **`register_server`** (or equivalent), subsequent boots find the row and can clear maintenance if a slot was set.
- **Implementation:** Centralise in **`resolve_server_for_maintenance_boot_completion(db, miner_hotkey, vm_name, request_ip?)`** (or fold into `process_boot_attestation`): lookup by **hotkey + name**, return **`None`** if absent, then run measurement + maintenance fields only when a **`Server`** exists.

---

## Design Decisions

- **No human admin (miner-facing):** Allow/deny is **fully automatic** inside the API for miners. **Validators** control windows by **rows in the database** (see **`tee_upgrade_windows`** below)—no miner-facing approve/deny routes. **This iteration:** no HTTP admin API for inserting windows (use migration seed, SQL runbook, or internal tooling); follow-up can add operator routes.
- **Global window = admission window (DB-backed):** The **active** upgrade window is the row in **`tee_upgrade_windows`** for which **`upgrade_window_start <= now() <= upgrade_window_end`** (see **Resolving the active window**). If **no** row qualifies, **deny** new **preflight** / **confirm** (**403** / **404**—document the choice). This is independent of **how long** a server stays “in maintenance” after approval:
  - **Already approved:** If the server has an **active slot** (`maintenance_deadline_at > now()`) because it **confirmed** before that row’s `upgrade_window_end`, the miner may **finish** (reboot, boot attestation) **after** `upgrade_window_end`. That is **allowed**—we do not revoke an in-flight slot when the clock passes the end bound.
  - **No new admits after close:** Once no row is “active” for `now`, **no further preflight/confirm** for new cycles until validators **insert** a new window row (or extend an existing row’s end time). Past rows remain for **history and audit**.
- **Why a table instead of env-only:** Environment variables give **no durable history** and encourage **silent overwrites**. A table yields **one row per coordinated target** (`target_measurement_version`), explicit **`upgrade_window_start` / `upgrade_window_end`**, optional **`created_at`**, and a full **audit trail** of past cutovers. **`GET …/policy`** (and internal checks) **read the active row from the DB**, optionally **cached** (short TTL Redis/in-process) to avoid hitting the DB on every preflight; cache must **invalidate or expire** quickly enough that window changes take effect promptly (or invalidate on write when an admin API exists).
- **Rollout identity = `tee_upgrade_windows` row:** **`target_measurement_version`** on the row is the **logical “which upgrade”** string (normalised once at insert). **`id`** distinguishes rows; enforce **`UNIQUE (target_measurement_version)`** so there is **at most one window definition per target** (adjust if you ever need a rare re-run for the same target—then drop uniqueness and key completion by **`id`** only).
- **One successful completion per server per active window (anti-abuse):** On boot success (below), set completion fields from the **pending window snapshot** (see **`servers`** columns). **Preflight** / **confirm** **deny** if **`servers.last_maintenance_completed_window_id`** equals the **active** window’s **`id`** (already finished **this** row’s cutover while that row is still active). When validators **insert** a **new** row (new **`id`**), miners who completed the **previous** row may enter again.
- **Already at or above target (no pointless purge):** **Preflight** / **confirm** **deny** if the server’s **current** attested **`measurement_version`** is **already >= the active row’s `target_measurement_version`** (same **semver compare** as boot completion, e.g. existing `semcomp` usage). **Rationale:** Maintenance exists to move hosts **onto** the mandated image; purging when already compliant would only create churn and scoring noise. **Source of truth:** use the **`Server`** row field(s) updated on **successful boot attestation**; document the exact column in implementation. **If no stored measurement exists yet**, **do not** treat as “already at target.”
- **Per-server slot:** On successful **confirm**, persist `maintenance_deadline_at = now + grace_hours`. **`grace_hours`** may come from the **active `tee_upgrade_windows` row** (preferred—per-rollout knob) or from **`settings`** if the column is null—pick one in implementation and document.
- **Per-miner concurrency (validator capacity knob):** A miner may have at most **`maintenance_max_concurrent_servers_per_miner`** servers in an **active** maintenance state at once (**default `1`**). **Active** means `maintenance_deadline_at IS NOT NULL` and `maintenance_deadline_at > now()` (same notion as “slot not expired”). Before accepting **confirm**, **count** distinct `servers` rows for that `miner_hotkey` meeting that condition; if **count ≥ limit**, return **409** with a clear message (and optionally `current_slots` / `limit` in JSON). An active slot **ends** when **`maintenance_deadline_at` has passed** (**without** setting `last_maintenance_completed_at`—miner failed or abandoned upgrade; **lazy** re-evaluation and/or optional **cron** to null `maintenance_declared_at` / `maintenance_deadline_at` only) or when **successful boot** runs the **completion** path below (**active** fields cleared, **completion** **`window_id` / target** fields set—see schema). **No miner `DELETE`** to clear maintenance. **Operator-only** override (if ever needed) is out of scope here.
- **Two-step flow (preflight + confirm):**
  - **Preflight:** `GET /servers/{server_id}/maintenance/preflight` — read-only check whether **confirm** would succeed **right now**—same auth as confirm (miner + `server_id`). Returns **`eligible: true`** or **`eligible: false`** with structured reasons (no active DB window, **already at or above active target**, **already completed this active window** (`last_maintenance_completed_window_id == active.id`), sole-survivor `chute_id`s, concurrency cap, already **active** slot on this server, not TEE, etc.). **No DB writes**, no purges (reads **active window** from DB or cache).
  - **Confirm:** `PUT /servers/{server_id}/maintenance` — **re-runs** all checks (must match preflight outcome unless state changed between calls), then **commits** maintenance slot and runs **auto-purge**. Idempotent **enter maintenance** semantics are acceptable for **PUT**. If preflight was **eligible** but state changed before confirm, confirm may **409**; clients should **re-preflight**.
- **TEE + ownership:** **Preflight / confirm** only for `Server.is_tee == true` and `server_id` + `HOTKEY_HEADER` passing existing ownership checks (`check_server_ownership`).
- **Sole-survivor rule (fixed policy):** If **any** active instance on that server is the **only** `active` instance globally for its `chute_id`, **`PUT …/maintenance` fails** with **409** and a JSON body listing blocking `{ chute_id, instance_id? }`. **Never** auto-terminate the globally last instance via maintenance: the network keeps that copy until another miner scales up. **Preflight** surfaces the same blocking set. **Auto-bounty on deny** remains **deferred** (follow-up).
- **Instance → server at delete time (forward-compatible):** **Do not** use `instances.host == servers.ip` as the **primary** link. That join is **consistent with today’s one-IP-per-server TEE rule** but will become **ambiguous** when **planned** multi-server-per-IP TEE exists. **Primary** resolution: **`Instance` → `instance_nodes` → `Node` → `Server`** (`nodes.server_id` → `servers.server_id`). **The platform does not support** an instance attached to nodes belonging to **more than one** `server_id`; no cross-server merge or precedence rules are required. For logging, policy, and any “logical server” checks, treat **`(servers.miner_hotkey, servers.name)`** as the stable human-facing identity (unique per miner today).
- **Boot completion → `Server` row:** Use **`BootAttestationArgs.miner_hotkey`** + **`BootAttestationArgs.vm_name`** → **`Server`** where **`servers.name = vm_name`** (and **`servers.miner_hotkey`** matches). **If no row:** first-boot / pre-registration path—**no** maintenance completion. Apply **`last_maintenance_completed_at` / active-slot clear** only on that resolved row.
- **Auto-terminate on confirm (primary path):** After **successful** confirm (DB commit sets maintenance slot), **enumerate all instances** on that server via **Instance → Node → Server** (not host/IP as primary), then for each instance invoke shared **purge** logic (as watchtower does): delete `instances` row, update `instance_audit` with `valid_termination = true`, `deletion_reason` e.g. `tee maintenance`, fail jobs, `notify_deleted`, `invalidate_instance_cache`, etc. **Order:** persist slot **before** purges so concurrent logic can see maintenance. **Performance:** many instances may require **sequential async** purges or a **background task**; document whether confirm **HTTP** waits for all purges to finish or returns after scheduling (prefer **wait** for small N, **task** for large N with idempotent retry on failure).
- **Last-instance / bounty on auto-purge:** Because **confirm** is **denied** when a sole survivor exists on the server, the maintenance purge batch **must not** include a globally last instance for any `chute_id`. For each instance purged in this flow, use **`valid_termination = true`**, maintenance **`deletion_reason`**, and shared **`purge`** machinery; last-instance bounty / multiplier slash **does not apply** to these rows (they are not last-global by construction). Implement via a shared helper or **`purge()`** flags—avoid duplicating `delete_instance` penalty logic.
- **Miner-initiated delete while slot active:** Rare if auto-purge drained the server; if any instance remains (partial failure, race, or future edge case), **`delete_instance`** should still treat **active maintenance slot** (`maintenance_deadline_at` not expired) like today (`valid_termination`, same last-instance policy), using the **active `tee_upgrade_windows` row** (or cache) only if you need to re-check bounds.
- **Successful boot = end of cycle (not “wipe and allow re-confirm”):** After **boot attestation** succeeds, **`resolve_server_for_maintenance_boot_completion`** (see above) must find a **`Server`** row; **if `None`**, skip this entire bullet. Otherwise, when the attested `measurement_version` is **>=** the **active** **`tee_upgrade_windows`** row’s **`target_measurement_version`** read **at boot time** (semver / `semcomp` as elsewhere): set **`last_maintenance_completed_at = now()`**, **`last_maintenance_completed_window_id = maintenance_pending_window_id`**, and **`last_maintenance_completed_target_measurement_version`** from the **pending window row’s** `target_measurement_version`. Then **clear active-slot fields** (`maintenance_declared_at`, `maintenance_deadline_at`, **`maintenance_pending_window_id`**, confirm metadata)—**do not** clear the **completion** fields. **After that:** new preflight/confirm requires a **new** active window row, **not** already at target, and **`last_maintenance_completed_window_id ≠ active.id`**.
- **Explicit non-goals (this spec):** No `scoring_penalty_multiplier` column; no cron adjusting scores for outdated VMs; no `INSTANCES_QUERY` / miner stats query changes for penalties.

### Identity, per-window limits, and abuse model

- **“One maintenance per server per global window”** is only as strong as **server identity**. `server_id` is often **ephemeral** (e.g. new Kubernetes node UID after reprovision). If the miner **deletes** the `servers` row and re-registers, or wipes storage and gets a **new** `server_id` / **new** `name`, the API sees a **new** server: we **cannot** infer they already consumed a slot on a logically “same” machine unless we add **durable tracking** outside `servers`.
- **Rename / reprovision loop:** A miner could enter maintenance, tear down the VM, re-register under a **new** `server_id` (and possibly a **new** `name`), and be eligible again for the **same** configured target. **Why this may be weak abuse:**
  - Each cycle implies **real downtime** and **lost compute / earnings** during reprovision and redeploy.
  - Maintenance protection only affects **how deletes are classified** (thrash + last-instance treatment); it does not mint extra rewards. The “profit” is avoiding scoring penalties on churn, which is bounded by how much they actually delete and redeploy.
- **Residual risk:** A miner could seek **more** `valid_termination`-style deletes than intended by policy if they can cheaply rotate server identities. Mitigation is **economic** (outage cost), **per-miner concurrency**, **`last_maintenance_completed_window_id`** / **target** per completed rollout (same `server_id` / row), **no miner cancel** after purge, plus optional **product** mitigations below.
- **Optional hardening (follow-up, if needed):** Append-only **`server_maintenance_events`** with columns like `(miner_hotkey, server_name, upgrade_window_id, confirmed_at)` and a **unique** constraint on `(upgrade_window_id, miner_hotkey, server_name)` — stops **reuse of the same name** in one rollout after row delete, but **does not** stop a miner who picks a **new name** each time. Stronger binding would need an **immutable** hardware or enrollment identifier (out of scope unless another feature provides it).

### Table `tee_upgrade_windows` (historical record, one row per target)

| Column | Type | Notes |
|--------|------|--------|
| **`id`** | bigint PK (or UUID) | Stable row identity for FKs from **`servers`**. |
| **`upgrade_window_start`** | timestamptz | Admission opens (new **preflight** / **confirm** allowed). |
| **`upgrade_window_end`** | timestamptz | Admission closes for **new** entries; in-flight slots may still finish. |
| **`target_measurement_version`** | text | Minimum attested measurement for this cutover; **normalised** on insert. **`UNIQUE`** recommended (one row per target version). |
| **`grace_hours`** | int nullable | Optional per-row grace after **confirm**; if null, use **`settings`** default. |
| **`created_at`** | timestamptz optional | When the row was inserted (audit). |

**Operational pattern:** `INSERT` a new row when shipping a **new** mandated image line; **UPDATE** `upgrade_window_end` to **now** (or past) to **end** admits for that cutover before opening the next. Old rows **stay** in the table as history.

### Resolving the “active” window

**Definition:** The **active** window is the single row (if any) such that **`upgrade_window_start <= now() <= upgrade_window_end`**. If **multiple** rows overlap (operator error), implementation must pick a **deterministic** rule (e.g. **highest `id`**, or **latest `created_at`**) and **log a warning**; validators should avoid overlaps.

**No active row:** No new **preflight** / **confirm**; feature is “closed” until a new row qualifies.

**Caching:** Load active row via a small helper used by preflight, confirm, policy, and boot completion; cache the result for a **short TTL** (or invalidate on writes) so policy GETs do not hammer the DB.

### Rollout identity, single artifact, and what belongs in maintenance

**Single published VM artifact:** The release pipeline exposes **only the latest** VM image. Once **0.3.1** is published, miners **cannot** fetch **0.3.0**. Any miner who **starts** an upgrade after that point is on the **current** image line.

**What maintenance windows are for (policy):** Use **coordinated admission windows** primarily for **major / minor** (or **breaking / validator-mandated**) TEE image moves. **Patch** releases: miners upgrade **on their own schedule** **without** this API—**no** maintenance-scoped protections for that path in this spec.

**Problem (minor bump mid-window):** A row exists with target **T0**; the pipeline publishes a newer image and the old one is **gone**.

**Operator response (recommended — end and replace, no overlap):**

1. **End** the current row’s admission: set **`upgrade_window_end`** to **now** (or past) on that row so it is no longer “active.”
2. **`INSERT`** a **new** row with **`target_measurement_version = T1`**, new **`upgrade_window_start` / `upgrade_window_end`**, and optional **`grace_hours`** / **`created_at`**.
3. Miners who **completed** the old cutover have **`last_maintenance_completed_window_id`** pointing at the **old** row → they **may** preflight/confirm again because the **active** row is a **new `id`**. Miners **in flight** (confirmed under old row, not yet booted): at boot, **`measurement_version`** must be **>=** the **active** row’s **`target_measurement_version`** read **at boot time** (the **new** floor). **`last_maintenance_completed_*`** must still be filled from the **`maintenance_pending_window_id`** row (the window they **entered** at **confirm**) so you do **not** attribute completion to the **new** row when they only confirmed under the **old** one.

**Overlapping concurrent windows:** **Out of scope** in v1—operators should **not** insert overlapping `[start, end]` ranges; if they do, deterministic resolution + warning applies.

**Implementation note:** **`GET …/policy`** returns the **active** row’s **`id`**, bounds, **`target_measurement_version`**, optional grace, plus **`maintenance_max_concurrent_servers_per_miner`** from settings and the miner’s active slot count.

---

## API Changes

- **New endpoints** (names illustrative; align with existing `/servers` prefix):
  - `GET /servers/{server_id}/maintenance/preflight` — miner auth, **no side effects**. Response e.g. `{ "eligible": bool, "reasons": [...], "blocking_chute_ids": [...], "current_slots": n, "limit": m, ... }`.
  - `PUT /servers/{server_id}/maintenance` — miner auth; body optional `{}` or `{ "ack": true }` if you want an explicit client ack. **Re-validates** all rules, then sets maintenance columns and **auto-purges**. Returns server id, `maintenance_deadline_at`, **list of `instance_id`s purged** (or async job id), echo of window/grace if desired.
  - **No** `DELETE /servers/.../maintenance` for miners.
  - `GET /servers/maintenance/policy` or `GET /servers/maintenance/window` — **read-only** global JSON: **active** window **`id`**, **`upgrade_window_start` / `upgrade_window_end`**, **`target_measurement_version`**, effective **grace** (row or settings fallback), **`maintenance_max_concurrent_servers_per_miner`** (from settings), miner’s **current active slot count** (for UX). Served from DB (via cache). No secrets.
- **Schema changes — new table `tee_upgrade_windows`:** As in the table above; add **`UNIQUE (target_measurement_version)`** if policy is strictly one row per target.
- **Schema changes (`servers` table):** Add nullable columns, e.g.:
  - `maintenance_declared_at` (timestamptz, nullable) — set at **confirm**
  - `maintenance_deadline_at` (timestamptz, nullable) — `now + grace` at confirm
  - **`maintenance_pending_window_id` (FK → `tee_upgrade_windows.id`, nullable)** — set to the **active** window’s **`id`** at **confirm** so boot completion attributes **`last_maintenance_completed_*`** to the row the miner **entered**, even if validators **insert a newer window row** before the VM reboots
  - **`last_maintenance_completed_at` (timestamptz, nullable)** — set **only** on **successful** boot attestation that ends the cycle; **never** cleared by normal flow
  - **`last_maintenance_completed_window_id` (FK nullable)** — set from **`maintenance_pending_window_id`** at boot success; **preflight** denies when this **equals** the **active** window’s **`id`**
  - **`last_maintenance_completed_target_measurement_version` (text, nullable)** — denormalised copy of the **pending window row’s** `target_measurement_version` at boot (for logs and quick string checks without a join)
  - **Optional (audit):** `maintenance_confirmed_at` (timestamptz, nullable) — support/metrics only; **409** if active slot already exists remains the default for duplicate confirm.
- **Migrations:** New timestamped SQL under `api/migrations/` creating **`tee_upgrade_windows`** and altering **`servers`**; keep `api/server/schemas.py` models in sync (this repo holds `Server` in that module, not `orms.py`—follow local convention).

---

## Goal

Success = a miner can **preflight** then **confirm** only when a **DB-backed active `tee_upgrade_windows` row** exists and `now` is inside **`[upgrade_window_start, upgrade_window_end]`**, **not** when **already at or above** that row’s **`target_measurement_version`**, and **not** when **`last_maintenance_completed_window_id`** equals the **active** row’s **`id`**. Subject to **sole-survivor rule** (deny **409**—never purge the globally last instance) and **per-miner concurrent-slot limit** (default **one** server at a time); on successful **confirm** the API **auto-purges** with **`valid_termination`** (no globally last instances in batch—see sole-survivor rule); **successful boot** sets **completion** fields and clears **active** slot fields on the **correct** server row; after **deadline expiry** without successful boot, active fields clear **without** setting completion (retry allowed if a row is still active). Validators add **new table rows** (and end old rows) for each coordinated cutover—**history** remains in **`tee_upgrade_windows`**. **No** miner **DELETE**. **No** scoring / metasync changes for outdated versions in this iteration.

Testable criteria:

- Migration applies cleanly; `Server` ORM matches DB.
- Preflight returns **`eligible: false`** when **no active window row**, **already >= active target**, **`last_maintenance_completed_window_id == active.id`**, outside **`[start, end]`**, **sole-survivor** blocking any instance, or at concurrency cap; confirm returns **403/409** consistently.
- After successful **confirm**, **all** targeted instances are **gone** from `instances` (or async job completes reliably), `instance_audit` shows **`valid_termination`** and maintenance reason, caches invalidated; **no** spurious thrash on miner redeploy after upgrade.
- Optional: miner `delete_instance` under active slot still correct if any instance left.
- Boot attestation success sets **completion** fields and clears **active** slot columns; subsequent preflight returns **ineligible** until the **active** window row is a **new `id`** (new **`INSERT`**) or other gates clear.
- `GET` policy endpoint returns expected shape when window open/closed.
- **Preflight** and **confirm** agree when state is unchanged; after state change, confirm may fail until re-preflight.

---

## Constraints

- Follow [AGENT.md](../../AGENT.md): **no new dependencies**; **window bounds and targets** live in **`tee_upgrade_windows`** (not env); optional **settings** for concurrency limit, default grace, cache TTL; **async** handlers; **Ruff** clean; add **tests** where behavior is non-trivial.
- **Do not** add `scoring_penalty_multiplier`, penalty cron, or `INSTANCES_QUERY` edits in this task.
- **Do not** hardcode window times or target versions in application code—load from the **DB** (active row) or documented migration seeds.
- Keep changes **focused**: prefer small helpers in `api/server/` (e.g. `util.py` or `service.py`) over cross-cutting refactors.

---

## Output Format

1. `api/migrations/YYYYMMDDHHMMSS_server_maintenance.sql` — `CREATE TABLE tee_upgrade_windows (...)`; indexes to resolve **active** row quickly (e.g. on `(upgrade_window_start, upgrade_window_end)` or as justified by queries); `ALTER TABLE servers ADD COLUMN …` / FKs.
2. `api/server/schemas.py` — `TeeUpgradeWindow` (or equivalent) model + new columns / relationships on **`Server`**.
3. `api/config.py` (or settings model) — **`maintenance_max_concurrent_servers_per_miner`**, optional **default `grace_hours`**, **policy cache TTL**; **not** window start/end/target (those are DB rows).
4. `api/server/router.py` — New routes; reuse `get_current_user` / hotkey patterns from existing server routes.
5. `api/server/service.py` (or new helper module) — **`get_active_upgrade_window()`** (DB + cache); `preflight_maintenance` / `confirm_maintenance` (already-at-target, **`last_maintenance_completed_window_id` vs active `id`**, set **`maintenance_pending_window_id`** at confirm); **deadline expiry** nulls active fields only; **`resolve_server_for_maintenance_boot_completion(db, miner_hotkey, vm_name, …)`** using **`BootAttestationArgs`**; extend **`process_boot_attestation`** (or call hook after success) to set **completion** from **pending window row** when **`Server`** exists and measurement OK; **no-op** when **no `Server`** row (**first boot**).
6. `api/instance/util.py` or `watchtower.py` or small `api/instance/maintenance_purge.py` — **Shared** “maintenance purge one instance” used by **confirm** batch; wraps or extends `purge` with **`valid_termination=True`** and maintenance **`deletion_reason`** (batch excludes globally last instances—see sole-survivor rule).
7. `api/instance/router.py` — Keep protected `delete_instance` branch for edge cases (optional if auto-purge is exhaustive).
8. `tests/unit/` (and/or integration) — Preflight **denied** when **no row / outside window / already >= target / completed same window id**; **allowed** after **new `tee_upgrade_windows` row** is active; deadline expiry **without** boot does **not** set completion; **Instance → Node → Server** resolution; **pending `window_id`** vs **new active row** before boot (supersede case).

---

## Failure Conditions

- Maintenance protection applies **outside** the global window or **after** `maintenance_deadline_at`.
- **Confirm** returns **success** when **any** blocking **sole-survivor** `chute_id` exists, when miner already holds **≥ limit** concurrent active slots, or when outside the active DB window (**confirm** must **re-validate** every check; outcomes must match preflight unless state legitimately changed).
- A miner-facing **`DELETE`** exists that **clears** maintenance (must **not** ship).
- Auto-purge uses **`valid_termination = false`** or omits last-instance protection → **thrash** or **wrong scoring** on redeploy.
- Instances **remain routable** after successful **confirm** (purge incomplete / wrong server scope).
- Boot success **omits** completion fields or **wipes** them on every boot (would allow **re-confirm** abuse for the **same** target).
- **Wrongly** allow preflight/confirm when **`last_maintenance_completed_window_id`** equals the **active** window’s **`id`**, or when the server is **already >= active target** (pointless purge).
- **Confirm** or purge **succeeds** while a **globally sole-survivor** instance for any `chute_id` would be terminated (**must** remain **409** / no purge—see fixed sole-survivor rule).
- Boot completion runs on **wrong** `servers` row (must use **`(miner_hotkey, vm_name)` → `servers.name`**, not **IP-only**) or runs without measurement check.
- Maintenance completion runs when **no** `servers` row exists (**first boot**) and incorrectly mutates state (should **no-op**).
- Delete / enumerate path uses **IP-only** as **primary** correlation instead of **Instance → Node → Server** (breaks **planned** shared-IP TEE; weak even today).
- Schema drift: migration applied but `Server` model missing columns (or reverse).
- **Any** dependency added without explicit approval.
- Scoring / metasync penalty code added despite scope.

---

## Rollout Notes

- **Database:** Document **`tee_upgrade_windows`** and the runbook: **`INSERT`** a row to open a cutover (**`target_measurement_version`**, **`upgrade_window_start` / `upgrade_window_end`**, optional **`grace_hours`**, **`created_at`**); **`UPDATE`** `upgrade_window_end` to end admits; **never** delete old rows if you want history (or archive separately). Optionally document in `dev/dev.md`.
- **Settings (optional):** `TEE_MAINTENANCE_MAX_CONCURRENT_SERVERS_PER_MINER`, default grace if row column null, cache TTL—final names follow `settings` naming.
- **Deploy order:** Migrate DB (table + server columns) → deploy API → **`INSERT`** first window row when ready; if **no** row is active for `now`, preflight/confirm deny new entry.
- **Operational:** Miners **`GET …/preflight` → `PUT …/maintenance`**; **`GET …/policy`** reflects the **active DB row** (cached). Validators manage **rows**, not env window clocks.

---

## Follow-ups (not this spec)

- Auto-bounty when confirm is blocked (sole survivor).
- `scoring_penalty_multiplier` + cron + `INSTANCES_QUERY` / miner stats alignment for outdated VMs post-window.
- **`server_maintenance_events`** (or similar) if per-rollout limits must survive server row deletion or **reuse of `server_name`**.
- **Optional:** **`server_id` in `BootAttestationArgs`** if product wants an explicit key beyond **`vm_name`**; **`vm_name` is already required** today ([`BootAttestationArgs`](../../api/server/schemas.py)).

