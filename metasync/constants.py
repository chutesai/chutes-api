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
    i.miner_hotkey,
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
LEFT JOIN computation_rates r ON i.chute_id = r.chute_id
WHERE i.started_at > NOW() - INTERVAL '{interval}'
AND i.error_message IS NULL
AND i.miner_uid > 0
AND i.completed_at IS NOT NULL
GROUP BY i.miner_hotkey
ORDER BY compute_units DESC;
"""
# Query to calculate the average number of unique chutes active at any single point in time, i.e. unique_count_count.
UNIQUE_CHUTE_AVERAGE_QUERY = """
WITH time_series AS (
  SELECT
    generate_series(
      date_trunc('hour', now() - INTERVAL '{interval}'),
      date_trunc('hour', now()),
      INTERVAL '10 minutes'
    ) AS time_point
),
chute_timeframes AS (
  SELECT
    chute_id,
    miner_hotkey,
    MIN(started_at) AS first_invocation,
    MAX(started_at) AS last_invocation
  FROM invocations
  WHERE
    started_at >= now() - INTERVAL '{interval}'
    AND error_message IS NULL
    AND completed_at IS NOT NULL
  GROUP BY chute_id, miner_hotkey
),
ten_minute_active_chutes AS (
  SELECT
    t.time_point,
    ct.miner_hotkey,
    COUNT(DISTINCT ct.chute_id) AS active_chutes
  FROM time_series t
  LEFT JOIN chute_timeframes ct ON
    t.time_point >= ct.first_invocation AND
    t.time_point <= ct.last_invocation
  GROUP BY t.time_point, ct.miner_hotkey
)
SELECT
  miner_hotkey,
  AVG(active_chutes)::integer AS avg_active_chutes
FROM ten_minute_active_chutes
GROUP BY miner_hotkey
ORDER BY avg_active_chutes DESC;
"""
