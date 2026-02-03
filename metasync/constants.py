SCORING_INTERVAL = "7 days"

# Dynamic bounty boost based on bounty age at claim time (maxes out at 3 hours)
BOUNTY_BOOST_MIN = 1.5
BOUNTY_BOOST_MAX = 2.5
BOUNTY_BOOST_RAMP_MINUTES = 180

# After claiming, instance compute_multiplier gradually adjusts toward
# the current chute target (base * urgency * TEE, etc.) over this many hours.
# The bounty boost component decays to 1.0, but other factors remain.
BOUNTY_BOOST_DECAY_HOURS = 5

# GPU inventory (and unique chute GPU).
INVENTORY_HISTORY_QUERY = """
WITH time_series AS (
  SELECT generate_series(
    date_trunc('hour', now() - INTERVAL '{interval}'),
    date_trunc('hour', now()),
    INTERVAL '1 hour'
  ) AS time_point
),
-- Get the latest gpu_count per chute (most recent entry only)
latest_chute_config AS (
  SELECT DISTINCT ON (chute_id)
    chute_id,
    (node_selector->>'gpu_count')::integer AS gpu_count
  FROM chute_history
  ORDER BY chute_id, created_at DESC
),
-- ALL active instances with GPU counts
active_instances_with_gpu AS (
  SELECT
    ts.time_point,
    ia.instance_id,
    ia.chute_id,
    ia.miner_hotkey,
    COALESCE(lcc.gpu_count, 1) AS gpu_count
  FROM time_series ts
  JOIN instance_audit ia
    ON ia.activated_at <= ts.time_point
   AND (ia.deleted_at IS NULL OR ia.deleted_at >= ts.time_point)
   AND ia.activated_at IS NOT NULL
   AND (
        ia.billed_to IS NOT NULL
        OR (COALESCE(ia.deleted_at, ts.time_point) - ia.activated_at >= interval '1 hour')
   )
  LEFT JOIN latest_chute_config lcc
    ON ia.chute_id = lcc.chute_id
),
-- Get all miners who had any activity in the interval
all_miners AS (
  SELECT DISTINCT miner_hotkey
  FROM active_instances_with_gpu
),
-- Cross join to get every miner x every hour
miner_time_series AS (
  SELECT
    ts.time_point,
    am.miner_hotkey
  FROM time_series ts
  CROSS JOIN all_miners am
),
-- Calculate metrics per timepoint
metrics_per_timepoint AS (
  SELECT
    time_point,
    miner_hotkey,
    -- For breadth: unique chutes with GPU weighting
    (SELECT SUM(gpu_count) FROM (
      SELECT DISTINCT ON (chute_id) chute_id, gpu_count
      FROM active_instances_with_gpu aig2
      WHERE aig2.time_point = aig.time_point
        AND aig2.miner_hotkey = aig.miner_hotkey
    ) unique_chutes) AS gpu_weighted_unique_chutes,
    -- For stability: total GPUs across all instances
    SUM(gpu_count) AS total_active_gpus
  FROM active_instances_with_gpu aig
  GROUP BY time_point, miner_hotkey
)
-- Return the history for both metrics (with zeros for missing hours)
SELECT
  mts.time_point::text,
  mts.miner_hotkey,
  COALESCE(mpt.gpu_weighted_unique_chutes, 0) AS unique_chute_gpus,
  COALESCE(mpt.total_active_gpus, 0) AS total_active_gpus
FROM miner_time_series mts
LEFT JOIN metrics_per_timepoint mpt
  ON mts.time_point = mpt.time_point
  AND mts.miner_hotkey = mpt.miner_hotkey
ORDER BY mts.miner_hotkey, mts.time_point
"""

# Instances lifetime/compute units queries - this is the entire basis for scoring!
# Uses instance_compute_history for accurate time-weighted multipliers.
# The history table includes the startup period (created_at to activated_at) at 0.3x rate,
# so billing_start uses created_at to capture this.
# All bonuses (bounty, urgency, TEE, private) are baked into compute_multiplier.
INSTANCES_QUERY = """
WITH billed_instances AS (
    SELECT
        ia.miner_hotkey,
        ia.instance_id,
        ia.chute_id,
        ia.created_at,
        ia.activated_at,
        ia.deleted_at,
        ia.stop_billing_at,
        ia.compute_multiplier,
        ia.bounty,
        -- Start from created_at to include startup period (history has 0.3x rate for this period)
        GREATEST(ia.created_at, now() - interval '{interval}') as billing_start,
        LEAST(
            COALESCE(ia.stop_billing_at, now()),
            COALESCE(ia.deleted_at, now()),
            now()
        ) as billing_end
    FROM instance_audit ia
    WHERE ia.activated_at IS NOT NULL
      AND (
          (
            ia.billed_to IS NULL
            AND ia.deleted_at IS NOT NULL
            AND ia.deleted_at - ia.activated_at >= INTERVAL '1 hour'
          )
          OR ia.valid_termination IS TRUE
          OR ia.deletion_reason in (
              'job has been terminated due to insufficient user balance',
              'user-defined/private chute instance has not been used since shutdown_after_seconds',
              'user has zero/negative balance (private chute)'
          )
          OR ia.deletion_reason LIKE '%has an old version%'
          OR ia.deleted_at IS NULL
      )
      AND (ia.deleted_at IS NULL OR ia.deleted_at >= now() - interval '{interval}')
),

-- Calculate time-weighted compute units using history table.
-- For each instance, sum (overlap_seconds * multiplier) across all history intervals.
instance_weighted_compute AS (
    SELECT
        bi.instance_id,
        bi.miner_hotkey,
        bi.billing_start,
        bi.billing_end,
        bi.bounty,
        bi.compute_multiplier as fallback_multiplier,
        COALESCE(
            SUM(
                EXTRACT(EPOCH FROM (
                    LEAST(COALESCE(ich.ended_at, now()), bi.billing_end)
                    - GREATEST(ich.started_at, bi.billing_start)
                )) * ich.compute_multiplier
            ),
            -- Fallback to instance_audit.compute_multiplier if no history exists
            EXTRACT(EPOCH FROM (bi.billing_end - bi.billing_start)) * COALESCE(bi.compute_multiplier, 1.0)
        ) AS weighted_compute_units
    FROM billed_instances bi
    LEFT JOIN instance_compute_history ich
           ON ich.instance_id = bi.instance_id
          AND ich.started_at < bi.billing_end
          AND (ich.ended_at IS NULL OR ich.ended_at > bi.billing_start)
    WHERE bi.billing_end > bi.billing_start
    GROUP BY bi.instance_id, bi.miner_hotkey, bi.billing_start, bi.billing_end, bi.bounty, bi.compute_multiplier
),

-- Aggregate compute units by miner
miner_compute_units AS (
    SELECT
        iwc.miner_hotkey,
        COUNT(*) AS total_instances,
        COUNT(CASE WHEN iwc.bounty IS TRUE THEN 1 END) AS bounty_score,
        SUM(EXTRACT(EPOCH FROM (iwc.billing_end - iwc.billing_start))) AS compute_seconds,
        SUM(iwc.weighted_compute_units) AS compute_units
    FROM instance_weighted_compute iwc
    GROUP BY iwc.miner_hotkey
)

SELECT
    miner_hotkey,
    total_instances,
    bounty_score,
    COALESCE(compute_seconds, 0) AS compute_seconds,
    COALESCE(compute_units, 0) AS compute_units
FROM miner_compute_units
ORDER BY compute_units DESC
"""
