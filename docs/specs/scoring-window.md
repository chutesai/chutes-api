# Feature Spec: Scoring Window Reduction (7 Days to 1 Day)

Use the sections **Goal**, **Constraints**, **Output Format**, and **Failure Conditions** as a **Prompt Contract** for this task (see [AGENT.md](../../AGENT.md) at repo root).

**Date**: 2026-04-13  
**Status**: in progress

---

## Context

Miner scoring currently uses a 7-day rolling window (`SCORING_INTERVAL = "7 days"`). The `INSTANCES_QUERY` sums `(overlap_seconds * compute_multiplier)` across all instance history segments within this window, normalizes to a distribution, and the result is set as on-chain weights every ~150 blocks (~30 min). The same scoring logic is independently implemented in the [chutes-audit](https://github.com/chutesai/chutes-audit) package, which lite validators use to set weights. Both must agree for vtrust to remain tight.

The 7-day window means new miners need a week to reach full scoring parity, misbehavior takes days to phase out, and the weight-setting cadence drifts (block-counting, no clock alignment) causing minor but unnecessary divergence between validator and auditor weight vectors.

- **Packages affected**: `metasync` (this repo), `chutes-audit` (external repo)
- **Key files**:
  - `metasync/constants.py` -- `SCORING_INTERVAL`, `INSTANCES_QUERY`, `INVENTORY_HISTORY_QUERY`
  - `metasync/shared.py` -- `get_scoring_data()`
  - `metasync/set_weights_on_metagraph.py` -- `set_weights_periodically()` loop
  - `api/metasync.py` -- `get_inventory_history()` (uses `SCORING_INTERVAL`)
  - `chute_autoscaler.py` -- `simulate_miner_scores()` (imports `SCORING_INTERVAL`)
- **Dependencies**: No new dependencies. `datetime` (stdlib) already available.

---

## Design Decisions

- **1-day window over shorter alternatives**: 4 hours makes the 1-hour deleted-instance eligibility rule significant (25% of window) and increases score volatility. 30 minutes conflicts fundamentally with the 1-hour rule, startup periods, and multiplier blend windows. 1 day keeps all existing eligibility rules negligible while being 7x more responsive.
- **Clock-aligned cadence at :00/:30 UTC**: Both validator and auditor evaluate at the same wall-clock times, ensuring `now()` in `INSTANCES_QUERY` produces near-identical window boundaries. Without alignment, the validator drifts (block-counting) and the auditor fires opportunistically, causing avoidable vtrust divergence.
- **Retain 150-block safety check**: The chain's `weights_rate_limit` (default 100 blocks / ~20 min) is enforced in blocks, not wall-clock time. The `LastUpdate` on-chain check is kept as a skip guard before calling `set_weights`. 30 min wall-clock comfortably exceeds 100 blocks; no functional change to chain interaction.
- **Decouple `INVENTORY_HISTORY_QUERY`**: Give it its own interval constant so the inventory chart can retain a 7-day view independent of the scoring window.
- **Coordinated deployment**: Both repos must switch within the same 30-minute window to avoid vtrust divergence. The auditor's autoupdater handles propagation to lite validators.

---

## API Changes

- **New endpoints**: None
- **Schema changes**: None
- **Migrations**: None

---

## Goal

Success =
1. `SCORING_INTERVAL` is `"1 day"` in both `metasync/constants.py` and `chutes-audit/audit.py`.
2. Weight setter (`set_weights_periodically`) sleeps until the next :00 or :30 UTC boundary, then checks the 150-block safety gate before setting weights.
3. `INVENTORY_HISTORY_QUERY` uses its own interval constant (remains `"7 days"`) independent of `SCORING_INTERVAL`.
4. `get_scoring_data()` still produces correct normalized scores with the 1-day window (no query changes needed -- the interval is parameterized).
5. Autoscaler `simulate_miner_scores()` automatically uses the new 1-day window via its import of `SCORING_INTERVAL`.
6. Miner stats endpoint (`/miner/stats`) is unaffected (uses its own hardcoded intervals).

---

## Constraints

- No new dependencies.
- No database migrations.
- No changes to `INSTANCES_QUERY` SQL -- only the interval parameter value changes.
- The `_check_scalable_private` 7-day requirement in `api/instance/router.py` is independent and must NOT change.
- The `/miner/stats` endpoint's hardcoded intervals (`"1 hour"`, `"1 day"`, `"7 days"`) are independent and must NOT change.
- The 150-block `LastUpdate` on-chain check must be retained as a safety gate in the weight-setting loop.

---

## Output Format

1. `metasync/constants.py` -- change `SCORING_INTERVAL` to `"1 day"`, add `SCORING_CADENCE_MINUTES = 30`, add `INVENTORY_INTERVAL = "7 days"`
2. `metasync/set_weights_on_metagraph.py` -- rewrite `set_weights_periodically()` to sleep until next :00/:30 UTC, retain 150-block safety check
3. `api/metasync.py` -- use `INVENTORY_INTERVAL` instead of `SCORING_INTERVAL` for `get_inventory_history()`

---

## Failure Conditions

- `SCORING_INTERVAL` is changed but weight-setting loop is not clock-aligned (drift resumes).
- `INVENTORY_HISTORY_QUERY` breaks because it still references `SCORING_INTERVAL` after decoupling.
- The 150-block safety check is removed, risking `SettingWeightsTooFast` chain rejections.
- `_check_scalable_private` or `/miner/stats` intervals are accidentally changed.
- Auditor repo is not updated to match, causing vtrust divergence on deploy.

---

## Rollout Notes

- **Validator deploy**: merge and deploy this repo. The new weight-setter pod sleeps until next :00/:30 UTC, then begins setting with `"1 day"` window.
- **Auditor deploy**: merge matching changes to `chutes-audit` (`SCORING_INTERVAL = "1 day"`, clock-aligned `_verify_integrity`, reduced data retention from 169h to ~25h, updated `compare_miner_metrics` interval). Release so autoupdater propagates to lite validators.
- **Ordering**: merge auditor PR first (but don't tag), deploy validator, then tag/release auditor. Brief misalignment is tolerable -- vtrust recovers within 1-2 weight-setting cycles once auditors update.
- **Miner impact**: normalized scores for steady-state miners stay roughly the same. New miners (>1 day) immediately reach full parity. Miners with recent downtime see sharper penalties (~4.2% per hour of outage vs ~0.6% under 7 days).
