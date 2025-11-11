SCORING_INTERVAL = "7 days"

# Bonuses applied to base score, where base score is simply compute units * instance lifetime where termination reason is valid.
BONUS = {
    "demand": 0.35,  # Miner generally meets the platform demands, i.e. chutes with high utilization are deployed more frequently.
    "bounty": 0.35,  # Claimed bounties, i.e. when there was platform demand for a chute, they launched it.
    "breadth": 0.3,  # Non-selectivity of the miner, i.e. deploying all chutes with equal weight.
}
DEMAND_COMPUTE_WEIGHT = 0.75
DEMAND_COUNT_WEIGHT = 0.25
BONUS_WEIGHT = 0.15
BONUS_EXP = 2.0

# Prevent bounty spamming.
BOUNTY_DECAY = 0.8
BOUNTY_RHO = 0.5

# Query to fetch raw request counts and compute units per chute (to calculate 'demand' bonus).
NORMALIZED_COMPUTE_QUERY = """
SELECT
    i.miner_hotkey,
    COUNT(CASE WHEN i.error_message IS NULL THEN 1 END) AS successful_count,
    SUM(
        i.compute_multiplier *
        CASE
            -- For token-based computations (nc = normalized compute, handles prompt & completion tokens).
            WHEN i.metrics->>'nc' IS NOT NULL
                AND (i.metrics->>'nc')::float > 0
            THEN (i.metrics->>'nc')::float

            -- For step-based computations
            WHEN i.metrics->>'steps' IS NOT NULL
                AND (i.metrics->>'steps')::float > 0
                AND i.metrics->>'masps' IS NOT NULL
            THEN (i.metrics->>'steps')::float * (i.metrics->>'masps')::float

            -- Legacy token-based calculation if 'nc' not available.
            WHEN i.metrics->>'it' IS NOT NULL
                AND i.metrics->>'ot' IS NOT NULL
                AND (i.metrics->>'it')::float > 0
                AND (i.metrics->>'ot')::float > 0
                AND i.metrics->>'maspt' IS NOT NULL
            THEN ((i.metrics->>'it')::float + (i.metrics->>'ot')::float) * (i.metrics->>'maspt')::float

            -- Fallback to actual elapsed time
            ELSE EXTRACT(EPOCH FROM (i.completed_at - i.started_at))
        END
    ) AS compute_units
FROM invocations i
WHERE i.started_at > NOW() - INTERVAL '{interval}'
AND NOT EXISTS (
    SELECT 1
    FROM reports
    WHERE invocation_id = i.parent_invocation_id
    AND confirmed_at IS NOT NULL
)
GROUP BY i.miner_hotkey ORDER BY compute_units desc
"""

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
-- Return the history for both metrics
SELECT
  time_point::text,
  miner_hotkey,
  COALESCE(gpu_weighted_unique_chutes, 0) AS unique_chute_gpus,
  COALESCE(total_active_gpus, 0) AS total_active_gpus
FROM metrics_per_timepoint
ORDER BY miner_hotkey, time_point
"""
INVENTORY_QUERY = (
    """
SELECT
  miner_hotkey,
  AVG(unique_chute_gpus)::integer AS avg_unique_chute_gpus,
  AVG(total_active_gpus)::integer AS avg_total_active_gpus
FROM ("""
    + INVENTORY_HISTORY_QUERY
    + """) AS history_data
GROUP BY miner_hotkey
ORDER BY avg_unique_chute_gpus DESC
"""
)

# Instances lifetime/compute units queries - this is the entire basis for scoring!
INSTANCES_QUERY = """
WITH billed_instances AS (
    SELECT
        ia.miner_hotkey,
        ia.instance_id,
        ia.chute_id,
        ia.activated_at,
        ia.deleted_at,
        ia.stop_billing_at,
        ia.compute_multiplier,
        ia.bounty,
        GREATEST(ia.activated_at, now() - interval '{interval}') as billing_start,
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

-- Count total bounties per chute in the interval
chute_bounty_totals AS (
    SELECT
        bi.chute_id,
        COUNT(*)::bigint AS n_total
    FROM billed_instances bi
    WHERE bi.bounty IS TRUE
      AND bi.billing_end > bi.billing_start
    GROUP BY bi.chute_id
),

-- Count bounties per miner per chute in the interval
miner_chute_bounty_counts AS (
    SELECT
        bi.miner_hotkey,
        bi.chute_id,
        COUNT(*)::bigint AS n_miner_chute
    FROM billed_instances bi
    WHERE bi.bounty IS TRUE
      AND bi.billing_end > bi.billing_start
    GROUP BY bi.miner_hotkey, bi.chute_id
),

-- Convert counts to an "effective" bounty score with:
--   per-miner geometric diminishing + global chute dampening
miner_bounty_effective AS (
    SELECT
        mcbc.miner_hotkey,
        SUM(
            (1.0 - POWER({bounty_decay}, mcbc.n_miner_chute::double precision))
            / (1.0 - {bounty_decay})
            *
            POWER(GREATEST(cbt.n_total, 1)::double precision, {bounty_rho} - 1.0)
        ) AS bounty_score
    FROM miner_chute_bounty_counts mcbc
    JOIN chute_bounty_totals cbt USING (chute_id)
    GROUP BY mcbc.miner_hotkey
),

-- Aggregate compute units by miner (and pull in the effective bounty score)
miner_compute_units AS (
    SELECT
        bi.miner_hotkey,
        COUNT(*) AS total_instances,
        COALESCE(mbe.bounty_score, 0.0) AS bounty_score,
        SUM(EXTRACT(EPOCH FROM (bi.billing_end - bi.billing_start))) AS compute_seconds,
        SUM(EXTRACT(EPOCH FROM (bi.billing_end - bi.billing_start)) * bi.compute_multiplier) AS compute_units
    FROM billed_instances bi
    LEFT JOIN miner_bounty_effective mbe
           ON mbe.miner_hotkey = bi.miner_hotkey
    WHERE bi.billing_end > bi.billing_start
    GROUP BY bi.miner_hotkey, mbe.bounty_score
)

SELECT
    miner_hotkey,
    total_instances,
    bounty_score,
    COALESCE(compute_seconds, 0) AS compute_seconds,
    COALESCE(compute_units, 0)  AS compute_units
FROM miner_compute_units
ORDER BY compute_units DESC
"""
