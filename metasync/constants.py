# Proportion of weights to assign to each metric.
FEATURE_WEIGHTS = {
    "compute_units": 0.72,  # Total amount of compute time (compute multiplier * total time).
    "unique_chute_count": 0.20,  # Average instantaneous unique chutes (gpu scaled) over the scoring period.
    "bounty_count": 0.08,  # Number of bounties received (not bounty values, just counts).
}
# Time slice to calculate the incentives from.
SCORING_INTERVAL = "7 days"
# Query to fetch raw metrics for compute_units and bounties.
NORMALIZED_COMPUTE_QUERY = """
SELECT
    mn.hotkey,
    COUNT(CASE WHEN i.bounty > 0 THEN 1 END) AS bounty_count,
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

            -- For token-based computations (nc = normalized compute, handles prompt & completion tokens).
            WHEN i.metrics->>'nc' IS NOT NULL
                AND (i.metrics->>'nc')::float > 0
            THEN (i.metrics->>'nc')::float

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
GROUP BY mn.hotkey
ORDER BY compute_units DESC;
"""
# Query to calculate the average number of unique chutes active at any single point in time, i.e. unique_chute_count.
UNIQUE_CHUTE_AVERAGE_QUERY = """
WITH time_series AS (
  SELECT
    generate_series(
      date_trunc('hour', now() - INTERVAL '{interval}'),
      date_trunc('hour', now()),
      INTERVAL '1 hour'
    ) AS time_point
),
-- Get all instances that had at least one successful invocation (ever) while the instance was alive.
instances_with_success AS (
  SELECT DISTINCT
    instance_id
  FROM invocations ii
  WHERE
    error_message IS NULL
    AND completed_at IS NOT NULL
    AND miner_uid >= 0
    AND NOT EXISTS (
        SELECT 1
        FROM reports
        WHERE invocation_id = ii.parent_invocation_id
        AND confirmed_at IS NOT NULL
    )
),
-- Get all unique miner_hotkeys from instance_audit relevant to the network
all_miners AS (
  SELECT DISTINCT ia.miner_uid, ia.miner_hotkey
  FROM instance_audit ia
  JOIN metagraph_nodes mn ON ia.miner_hotkey = mn.hotkey
  WHERE mn.netuid = 64 AND mn.node_id >= 0
),
-- For each time point, find active instances that have had successful invocations
active_instances_per_timepoint AS (
  SELECT
    ts.time_point,
    ia.instance_id,
    ia.chute_id,
    ia.miner_hotkey
  FROM time_series ts
  JOIN instance_audit ia ON
    ia.verified_at <= ts.time_point AND
    (ia.deleted_at IS NULL OR ia.deleted_at >= ts.time_point)
  JOIN metagraph_nodes mn ON ia.miner_hotkey = mn.hotkey
  JOIN instances_with_success iws ON
    ia.instance_id = iws.instance_id
  WHERE mn.netuid = 64 AND mn.node_id >= 0
),
-- Pre-calculate the most recent GPU count for each chute from chute_history
chute_latest_gpu_history AS (
  SELECT DISTINCT ON (ch.chute_id) -- Get only the latest record per chute_id
      ch.chute_id,
      (ch.node_selector->>'gpu_count')::integer AS latest_gpu_count
  FROM chute_history ch
  WHERE ch.node_selector ? 'gpu_count'
    AND jsonb_typeof(ch.node_selector->'gpu_count') = 'number'
  ORDER BY ch.chute_id, ch.created_at DESC
),
-- Calculate GPU-weighted chute count per miner per time point using historical max GPU count
active_chutes_per_timepoint AS (
  SELECT
    aipt.time_point,
    aipt.miner_hotkey,
    -- Sum the latest_gpu_count, defaulting to 1 if no valid history exists for the chute
    SUM(COALESCE(clgh.latest_gpu_count, 1)) AS gpu_weighted_chutes
  FROM (
    -- Get distinct chute_ids per time point and miner
    SELECT DISTINCT
      time_point,
      miner_hotkey,
      chute_id
    FROM active_instances_per_timepoint
  ) aipt
  -- LEFT JOIN with the pre-calculated *latest* GPU count per chute from history
  LEFT JOIN chute_latest_gpu_history clgh ON -- Join with the new CTE
    aipt.chute_id = clgh.chute_id
  GROUP BY aipt.time_point, aipt.miner_hotkey
),
-- Create a cross join of all time points with all miners
all_timepoints_for_all_miners AS (
  SELECT
    ts.time_point,
    am.miner_hotkey
  FROM time_series ts
  CROSS JOIN all_miners am
),
-- Join with active_chutes to get complete dataset with zeros
complete_dataset AS (
  SELECT
    atm.miner_hotkey,
    atm.time_point,
    COALESCE(acpt.gpu_weighted_chutes, 0) AS gpu_weighted_chutes
  FROM all_timepoints_for_all_miners atm
  LEFT JOIN active_chutes_per_timepoint acpt ON
    atm.time_point = acpt.time_point AND
    atm.miner_hotkey = acpt.miner_hotkey
)
-- Calculate average GPU-weighted chutes per miner across all time points
SELECT miner_hotkey, AVG(gpu_weighted_chutes)::integer AS avg_gpu_weighted_chutes
FROM complete_dataset
GROUP BY miner_hotkey
ORDER BY avg_gpu_weighted_chutes DESC;
"""

# Private instances, including jobs.
PRIVATE_INSTANCES_QUERY = """
WITH billed_instances AS (
    SELECT
        ia.miner_hotkey,
        ia.instance_id,
        ia.activated_at,
        ia.stop_billing_at,
        i.compute_multiplier,
        GREATEST(ia.activated_at, now() - interval '{interval}') as billing_start,
        LEAST(COALESCE(ia.stop_billing_at, now()), now()) as billing_end
    FROM instance_audit ia
    JOIN instances i ON i.instance_id = ia.instance_id
    WHERE ia.billed_to IS NOT NULL
      AND (
        (ia.activated_at >= now() - interval '{interval}' OR ia.activated_at < now() - interval '{interval}')
        AND (ia.stop_billing_at IS NULL OR ia.stop_billing_at >= now() - interval '{interval}')
      )
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
FROM miner_compute_units
ORDER BY compute_units DESC;
"""

# Unique chute history.
UNIQUE_CHUTE_HISTORY_QUERY = (
    UNIQUE_CHUTE_AVERAGE_QUERY.replace(
        "SELECT miner_hotkey, AVG", "SELECT miner_hotkey, time_point::text, AVG"
    )
    .replace("GROUP BY miner_hotkey", "GROUP BY miner_hotkey, time_point")
    .replace("ORDER BY avg_gpu_weighted_chutes DESC", "ORDER BY miner_hotkey ASC, time_point DESC")
    .replace(
        "FROM complete_dataset",
        "FROM complete_dataset WHERE miner_hotkey IN (SELECT hotkey FROM metagraph_nodes WHERE netuid = 64)",
    )
)

# Utilization ratio for busiest chutes.
UTILIZATION_THRESHOLD = 0.02
UTILIZATION_RATIO_QUERY = """
WITH instance_spans AS (
  SELECT
    miner_hotkey, instance_id,
    MAX(completed_at) - MIN(started_at) as total_active_time,
    SUM(completed_at - started_at) AS total_processing_time
  FROM invocations
  WHERE started_at >= now() - INTERVAL '{interval}'
  AND error_message IS NULL AND completed_at IS NOT NULL
  GROUP BY miner_hotkey, instance_id
),
instance_metrics AS (
  SELECT
    miner_hotkey, instance_id,
    EXTRACT(EPOCH FROM total_active_time) AS total_active_seconds,
    EXTRACT(EPOCH FROM total_processing_time) AS total_processing_seconds,
    CASE
      WHEN EXTRACT(EPOCH FROM total_active_time) > 0
      THEN ROUND(
        (EXTRACT(EPOCH FROM total_processing_time) /
         EXTRACT(EPOCH FROM total_active_time))::numeric,
        2
      )
      ELSE 0
    END AS busy_ratio
  FROM instance_spans
  JOIN metagraph_nodes mn ON instance_spans.miner_hotkey = mn.hotkey
),
ranked_instances AS (
  SELECT
    miner_hotkey, instance_id,
    total_active_seconds, total_processing_seconds, busy_ratio,
    ROW_NUMBER() OVER (PARTITION BY miner_hotkey ORDER BY busy_ratio DESC) AS rank
  FROM instance_metrics WHERE total_active_seconds >= 3600
),
top_instances AS (
  SELECT
    miner_hotkey, instance_id,
    total_active_seconds, total_processing_seconds, busy_ratio
  FROM ranked_instances
  WHERE rank <= 3
),
instance_counts AS (
  SELECT
    miner_hotkey,
    COUNT(*) AS instance_count
  FROM top_instances
  GROUP BY miner_hotkey
)
SELECT
  mn.hotkey AS miner_hotkey,
  CASE
    WHEN ic.instance_count >= 3 THEN ROUND(MIN(ti.busy_ratio)::numeric, 6)
    ELSE 0
  END AS min_top_busy_ratio
FROM metagraph_nodes mn
LEFT JOIN top_instances ti ON mn.hotkey = ti.miner_hotkey
LEFT JOIN instance_counts ic ON mn.hotkey = ic.miner_hotkey
WHERE mn.netuid = 64
GROUP BY mn.hotkey, ic.instance_count
ORDER BY min_top_busy_ratio DESC;
"""
