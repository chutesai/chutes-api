# Proportion of weights to assign to each metric.
FEATURE_WEIGHTS = {
    "compute_units": 0.45,  # Total amount of compute time (compute muliplier * total time).
    "invocation_count": 0.25,  # Total number of invocations.
    "unique_chute_count": 0.2,  # Average instantaneous unique chutes over the scoring period.
    "bounty_count": 0.1,  # Number of bounties received (not bounty values, just counts).
}
# Time slice to calculate the incentives from.
SCORING_INTERVAL = "7 days"
# Query to fetch raw metrics for compute_units, invocation_count, and bounty_count.
NORMALIZED_COMPUTE_QUERY = """
WITH computation_rates AS (
    SELECT
        chute_id,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch from completed_at - started_at) / (metrics->>'steps')::float) as median_step_time,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch from completed_at - started_at) / ((metrics->>'it')::float + (metrics->>'ot')::float)) as median_token_time
    FROM invocations
    WHERE ((metrics->>'steps' IS NOT NULL and (metrics->>'steps')::float > 0) OR (metrics->>'it' IS NOT NULL AND metrics->>'ot' IS NOT NULL AND (metrics->>'ot')::float > 0 AND (metrics->>'it')::float > 0))
      AND started_at >= NOW() - INTERVAL '2 days'
    GROUP BY chute_id
)
SELECT
    mn.hotkey,
    COUNT(*) as invocation_count,
    COUNT(CASE WHEN i.bounty > 0 THEN 1 END) AS bounty_count,
    sum(
        i.bounty +
        i.compute_multiplier *
        CASE
            WHEN i.metrics->>'steps' IS NOT NULL
                AND r.median_step_time IS NOT NULL
            THEN (i.metrics->>'steps')::float * r.median_step_time
            WHEN i.metrics->>'it' IS NOT NULL
                AND i.metrics->>'ot' IS NOT NULL
                AND r.median_token_time IS NOT NULL
            THEN ((i.metrics->>'it')::float + (i.metrics->>'ot')::float) * r.median_token_time
            ELSE EXTRACT(EPOCH FROM (i.completed_at - i.started_at))
        END
    ) AS compute_units
FROM invocations i
JOIN metagraph_nodes mn ON i.miner_hotkey = mn.hotkey AND mn.netuid = 64
LEFT JOIN computation_rates r ON i.chute_id = r.chute_id
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
# Query to calculate the average number of unique chutes active at any single point in time, i.e. unique_count_count.
UNIQUE_CHUTE_AVERAGE_QUERY = """
WITH time_series AS (
  SELECT
    generate_series(
      date_trunc('hour', now() - INTERVAL '7 days'),
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
-- Pre-calculate the maximum historical GPU count for each chute from chute_history
chute_max_gpu_history AS (
  SELECT
    ch.chute_id,
    -- Find the maximum gpu_count recorded in history for this chute
    -- Ensure the value extracted is actually a number before casting
    MAX(CASE
          WHEN jsonb_typeof(ch.node_selector->'gpu_count') = 'number'
          THEN (ch.node_selector->>'gpu_count')::integer
          ELSE NULL -- Ignore non-numeric values
        END) AS max_gpu_count
  FROM chute_history ch
  WHERE ch.node_selector ? 'gpu_count'
  GROUP BY ch.chute_id
),
-- Calculate GPU-weighted chute count per miner per time point using historical max GPU count
active_chutes_per_timepoint AS (
  SELECT
    aipt.time_point,
    aipt.miner_hotkey,
    SUM(COALESCE(cmgh.max_gpu_count, 1)) AS gpu_weighted_chutes
  FROM (
    -- Get distinct chute_ids per time point and miner
    SELECT DISTINCT
      time_point,
      miner_hotkey,
      chute_id
    FROM active_instances_per_timepoint
  ) aipt
  -- LEFT JOIN with the pre-calculated max GPU count per chute from history
  LEFT JOIN chute_max_gpu_history cmgh ON
    aipt.chute_id = cmgh.chute_id
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
