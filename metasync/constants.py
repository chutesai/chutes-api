# Proportion of weights to assign to each metric.
FEATURE_WEIGHTS = {
    "compute_units": 0.52,  # Total amount of compute time (compute multiplier * total time).
    "invocation_count": 0.20,  # Total number of invocations.
    "unique_chute_count": 0.20,  # Average instantaneous unique chutes (gpu scaled) over the scoring period.
    "bounty_count": 0.08,  # Number of bounties received (not bounty values, just counts).
}
# Time slice to calculate the incentives from.
SCORING_INTERVAL = "7 days"
# Query to fetch raw metrics for compute_units and bounties.
NORMALIZED_COMPUTE_QUERY = """
SELECT
    mn.hotkey,
    COUNT(CASE WHEN (i.metrics->>'p')::bool IS NOT TRUE THEN 1 END) as invocation_count,
    COUNT(CASE WHEN i.bounty > 0 AND (i.metrics->>'p')::bool IS NOT TRUE THEN 1 END) AS bounty_count,
    sum(
        i.bounty +
        i.compute_multiplier *
        CASE
            -- Private chutes/jobs/etc are accounted for by instance data instead of here.
            WHEN (i.metrics->>'p')::bool IS TRUE THEN 0::float

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
JOIN metagraph_nodes mn ON i.miner_hotkey = mn.hotkey AND mn.netuid = 64
WHERE i.started_at > NOW() - INTERVAL '{interval}'
AND i.error_message IS NULL
AND i.miner_uid >= 0
AND i.completed_at IS NOT NULL
AND NOT EXISTS (
    SELECT 1
    FROM reports
    WHERE invocation_id = i.parent_invocation_id
    AND confirmed_at IS NOT NULL
)
GROUP BY mn.hotkey;
"""
# Query to calculate the average number of unique chutes active at any single point in time, i.e. unique_chute_count.
UNIQUE_CHUTE_BASE = """
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
active_chutes AS (
  SELECT DISTINCT ON (ts.time_point, ia.chute_id, ia.miner_hotkey)
    ts.time_point,
    ia.chute_id,
    ia.miner_hotkey
  FROM time_series ts
  JOIN instance_audit ia
    ON ia.activated_at <= ts.time_point
   AND (ia.deleted_at IS NULL OR ia.deleted_at >= ts.time_point)
   AND ia.activated_at IS NOT NULL
   AND (
        ia.billed_to IS NOT NULL
        OR (COALESCE(ia.deleted_at, ts.time_point) - ia.activated_at >= interval '1 hour')
   )
)
"""
UNIQUE_CHUTE_AVERAGE_QUERY = (
    UNIQUE_CHUTE_BASE
    + """,
-- Calculate GPU-weighted chutes per miner per time point
gpu_weighted_per_timepoint AS (
  SELECT
    ac.time_point,
    ac.miner_hotkey,
    SUM(COALESCE(lcc.gpu_count, 1)) AS gpu_weighted_chutes
  FROM active_chutes ac
  LEFT JOIN latest_chute_config lcc
    ON ac.chute_id = lcc.chute_id
  GROUP BY ac.time_point, ac.miner_hotkey
)
-- Calculate the average across all time points
SELECT
  miner_hotkey,
  AVG(gpu_weighted_chutes)::integer AS avg_gpu_weighted_chutes
FROM gpu_weighted_per_timepoint
GROUP BY miner_hotkey
ORDER BY avg_gpu_weighted_chutes DESC;
"""
)
UNIQUE_CHUTE_HISTORY_QUERY = (
    UNIQUE_CHUTE_BASE
    + """
SELECT
  ac.time_point::text,
  ac.miner_hotkey,
  SUM(COALESCE(lcc.gpu_count, 1)) AS avg_gpu_weighted_chutes
FROM active_chutes ac
LEFT JOIN latest_chute_config lcc
  ON ac.chute_id = lcc.chute_id
GROUP BY ac.time_point, ac.miner_hotkey;
"""
)

# Private instances, including jobs.
PRIVATE_INSTANCES_QUERY = """
WITH billed_instances AS (
    SELECT
        ia.miner_hotkey,
        ia.instance_id,
        ia.activated_at,
        ia.stop_billing_at,
        ia.compute_multiplier,
        GREATEST(ia.activated_at, now() - interval '{interval}') as billing_start,
        LEAST(
            COALESCE(ia.stop_billing_at, now()),
            COALESCE(ia.deleted_at, now()),
            now()
        ) as billing_end
    FROM instance_audit ia
    WHERE ia.billed_to IS NOT NULL
      AND ia.activated_at IS NOT NULL
      AND ia.deletion_reason != 'miner initialized'
      AND (ia.stop_billing_at IS NULL OR ia.stop_billing_at >= now() - interval '{interval}')
),

-- Aggregate compute units by miner
miner_compute_units AS (
    SELECT
        miner_hotkey,
        COUNT(*) as total_instances,
        SUM(EXTRACT(EPOCH FROM (billing_end - billing_start))) as compute_seconds,
        SUM(EXTRACT(EPOCH FROM (billing_end - billing_start)) * compute_multiplier) as compute_units
    FROM billed_instances
    WHERE billing_end > billing_start
    GROUP BY miner_hotkey
)
SELECT
    miner_hotkey,
    total_instances,
    COALESCE(compute_seconds, 0) as compute_seconds,
    COALESCE(compute_units, 0) as compute_units
FROM miner_compute_units;
"""
