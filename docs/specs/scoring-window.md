# Feature Spec: Scoring Window Reduction & Audit Export Cleanup

Use the sections **Goal**, **Constraints**, **Output Format**, and **Failure Conditions** as a **Prompt Contract** for this task (see [AGENT.md](../../AGENT.md) at repo root).

**Date**: 2026-04-13  
**Status**: in progress

---

## Context

Miner scoring previously used a 7-day rolling window (`SCORING_INTERVAL = "7 days"`). The `INSTANCES_QUERY` sums `(overlap_seconds * compute_multiplier)` across all instance history segments within this window, normalizes to a distribution, and the result is set as on-chain weights. The same scoring logic is independently implemented in the [chutes-audit](https://github.com/chutesai/chutes-audit) package, which lite validators use to set weights. Both must agree for vtrust to remain tight.

The 7-day window meant new miners needed a week to reach full scoring parity, misbehavior took days to phase out, and the weight-setting cadence drifted (block-counting, no clock alignment) causing minor but unnecessary divergence between validator and auditor weight vectors.

Additionally, the audit exporter (`audit_exporter.py`) was packaging invocations, reports, and jobs CSVs alongside the instance audit data. Since scoring is now based entirely on instance compute units (not invocations), these CSVs are no longer consumed by the auditor and waste bandwidth.

- **Packages affected**: `metasync` (this repo), `audit_exporter.py` (this repo), `chutes-audit` (external repo)
- **Key files**:
  - `metasync/constants.py` -- `SCORING_INTERVAL`, `INVENTORY_INTERVAL`, `INSTANCES_QUERY`, `INVENTORY_HISTORY_QUERY`
  - `metasync/shared.py` -- `get_scoring_data()`
  - `metasync/set_weights_on_metagraph.py` -- `set_weights_periodically()` loop
  - `api/metasync.py` -- `get_inventory_history()` (uses `INVENTORY_INTERVAL`)
  - `audit_exporter.py` -- hourly audit export CronJob
  - `chute_autoscaler.py` -- `simulate_miner_scores()` (imports `SCORING_INTERVAL`)
- **Dependencies**: No new dependencies. `datetime` (stdlib) already available.

---

## Design Decisions

- **1-day window over shorter alternatives**: 4 hours makes the 1-hour deleted-instance eligibility rule significant (25% of window) and increases score volatility. 30 minutes conflicts fundamentally with the 1-hour rule, startup periods, and multiplier blend windows. 1 day keeps all existing eligibility rules negligible while being 7x more responsive.
- **Hourly clock-aligned cadence at :00 UTC**: Both validator and auditor evaluate at the top of each UTC hour, ensuring `now()` in `INSTANCES_QUERY` produces near-identical window boundaries. This aligns naturally with the audit exporter CronJob which runs at minute 1 of each hour. Without alignment, the validator drifts (block-counting) and the auditor fires opportunistically, causing avoidable vtrust divergence.
- **Retain 150-block safety check**: The chain's `weights_rate_limit` (default 100 blocks / ~20 min) is enforced in blocks, not wall-clock time. The `LastUpdate` on-chain check is kept as a skip guard before calling `set_weights`. 1 hour wall-clock comfortably exceeds 100 blocks; no functional change to chain interaction.
- **Decouple `INVENTORY_HISTORY_QUERY`**: Give it its own interval constant (`INVENTORY_INTERVAL`) so the inventory chart retains a 7-day view independent of the scoring window.
- **Remove invocations/reports/jobs from audit export**: Scoring no longer uses invocations. The auditor is being updated to remove synthetics and invocation CSV checking. The validator's `audit_exporter.py` stops generating and uploading these CSVs, and removes `csv_exports` from the JSON payload. The audit JSON now contains only `instance_audit` and `compute_history`.
- **Coordinated deployment**: Both repos must switch within the same hour to avoid vtrust divergence. The auditor's autoupdater handles propagation to lite validators.

---

## API Changes

- **New endpoints**: None
- **Schema changes**: None
- **Migrations**: None
- **Audit JSON payload change**: `csv_exports` key removed from the validator audit report JSON. Auditor must be updated to not expect it.

---

## Goal

Success =
1. `SCORING_INTERVAL` is `"1 day"` in both `metasync/constants.py` and `chutes-audit/audit.py`.
2. Weight setter (`set_weights_periodically`) sleeps until the top of each UTC hour, then checks the 150-block safety gate before setting weights. Retries within a 15-minute budget on failure.
3. `INVENTORY_HISTORY_QUERY` uses its own interval constant (`INVENTORY_INTERVAL = "7 days"`) independent of `SCORING_INTERVAL`.
4. `get_scoring_data()` still produces correct normalized scores with the 1-day window (no query changes needed -- the interval is parameterized).
5. Autoscaler `simulate_miner_scores()` automatically uses the new 1-day window via its import of `SCORING_INTERVAL`.
6. Miner stats endpoint (`/miner/stats`) is unaffected (uses its own hardcoded intervals).
7. `audit_exporter.py` no longer generates or uploads invocations/reports/jobs CSVs. The audit JSON payload contains only `instance_audit` and `compute_history`.

---

## Constraints

- No new dependencies.
- No database migrations.
- No changes to `INSTANCES_QUERY` SQL -- only the interval parameter value changes.
- The `_check_scalable_private` 7-day requirement in `api/instance/router.py` is independent and must NOT change.
- The `/miner/stats` endpoint's hardcoded intervals (`"1 hour"`, `"1 day"`, `"7 days"`) are independent and must NOT change.
- The 150-block `LastUpdate` on-chain check must be retained as a safety gate in the weight-setting loop.
- The `/invocations/exports/` API endpoints remain available (serve existing S3 data), but no new CSVs will be uploaded by the exporter.

---

## Output Format

1. `metasync/constants.py` -- `SCORING_INTERVAL = "1 day"`, `INVENTORY_INTERVAL = "7 days"` (done)
2. `metasync/set_weights_on_metagraph.py` -- hourly UTC clock-aligned `set_weights_periodically()` with 150-block safety check and retry budget (done)
3. `api/metasync.py` -- uses `INVENTORY_INTERVAL` for `get_inventory_history()` (done)
4. `audit_exporter.py` -- remove `INVOCATION_QUERY`, `REPORT_QUERY`, `JOB_QUERY`, `generate_invocation_report_data()`, `get_sha256()`, and `csv_exports` from JSON payload

---

## Failure Conditions

- `SCORING_INTERVAL` is changed but weight-setting loop is not clock-aligned (drift resumes).
- `INVENTORY_HISTORY_QUERY` breaks because it references a removed constant.
- The 150-block safety check is removed, risking `SettingWeightsTooFast` chain rejections.
- `_check_scalable_private` or `/miner/stats` intervals are accidentally changed.
- Auditor repo is not updated to match (scoring interval or audit JSON schema), causing vtrust divergence or parse errors on deploy.
- Audit exporter still attempts to query invocations DB or upload CSVs.

---

## Rollout Notes

- **Validator deploy**: merge and deploy this repo. The weight-setter pod sleeps until top of next UTC hour, then begins setting with `"1 day"` window. Audit exporter stops generating invocation CSVs immediately.
- **Auditor deploy**: merge matching changes to `chutes-audit` (`SCORING_INTERVAL = "1 day"`, hourly clock-aligned `_verify_integrity`, remove synthetics and invocation CSV checking, reduced data retention from 169h to ~25h, updated `compare_miner_metrics` interval). Release so autoupdater propagates to lite validators.
- **Ordering**: deploy validator first, then release auditor. The auditor must tolerate missing `csv_exports` in the audit JSON (or be updated first to not require it). Brief scoring misalignment is tolerable -- vtrust recovers within 1-2 weight-setting cycles once auditors update.
- **Miner impact**: normalized scores for steady-state miners stay roughly the same. New miners (>1 day) immediately reach full parity. Miners with recent downtime see sharper penalties (~4.2% per hour of outage vs ~0.6% under 7 days).

---

## Legacy Trailing Credit (added 2026-04-21)

The 7→1 day window change breaks ramp-up/ramp-down symmetry for nodes that were added under the old regime. Every node added under the 7-day window paid a ramp-up "tax" (diluted contribution for its first 7 days). The symmetric compensation was a 7-day ramp-down "cushion" after removal (the node lingers in the scoring window). Switching to a 1-day window means those nodes would only receive a 1-day cushion — losing ~3 node-days of owed trailing credit.

### Per-Instance Window Design

Instead of a global taper (which only helps nodes removed during the taper period), the fix uses **per-instance window logic** in `INSTANCES_QUERY`:

- **Active instances** (not yet removed): always use the current 1-day window. This keeps steady-state normalized scoring fair for everyone — pre-change and post-change active instances contribute equally after normalization.
- **Deleted instances created BEFORE `SCORING_WINDOW_CHANGE_DATE`** (2026-04-20): use the legacy 7-day window for both the deletion filter (`deleted_at >= now() - 7 days`) and `billing_start` (`GREATEST(created_at, now() - 7 days)`). This gives them the trailing credit they already paid for.
- **Deleted instances created AFTER the change date**: use the current 1-day window (symmetric with their 1-day ramp-up).

This naturally phases out: no new pre-change instances can be created, so once all legacy instances have been removed and their 7-day trail expires, the legacy CASE branches become dead code.

### Auditor Data Requirements

Since auditors must independently compute the same per-instance window logic, they need 7 days of historical data (not just 1 day). The audit entry endpoint (`GET /audit/`) uses `LEGACY_SCORING_WINDOW` (7 days + 1 hour buffer) to ensure auditors receive sufficient history. The instance reconciliation and compute history CSV endpoints already use 7-8 day windows.

### Files Changed

- `metasync/constants.py` — added `LEGACY_SCORING_INTERVAL`, `LEGACY_SCORING_WINDOW`, `SCORING_WINDOW_CHANGE_DATE`; updated `INSTANCES_QUERY` with per-instance CASE logic
- `metasync/shared.py` — passes legacy parameters when formatting `INSTANCES_QUERY`
- `chute_autoscaler.py` — same changes to the duplicated query in `simulate_miner_scores()`
- `api/audit/router.py` — uses `LEGACY_SCORING_WINDOW` instead of `SCORING_WINDOW` for audit entry listing
