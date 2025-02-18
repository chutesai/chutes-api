# Proportion of weights to assign to each metric.
FEATURE_WEIGHTS = {
    "compute_units": 0.45,  # Total amount of compute time (compute muliplier * total time).
    "invocation_count": 0.25,  # Total number of invocations.
    "unique_chute_count": 0.20,  # Number of unique chutes over the scoring period.
    "bounty_count": 0.1,  # Number of bounties received (not bounty values, just counts).
}
SCORING_INTERVAL = "7 days"
NORMALIZED_COMPUTE_QUERY = """
WITH computation_rates AS (
    SELECT
        chute_id,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch from completed_at - started_at) / (metrics->>'steps')::float) as median_step_time,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch from completed_at - started_at) / (metrics->>'tokens')::float) as median_token_time
    FROM invocations
    WHERE ((metrics->>'steps' IS NOT NULL and (metrics->>'steps')::float > 0) OR (metrics->>'tokens' IS NOT NULL and (metrics->>'tokens')::float > 0))
      AND started_at >= NOW() - INTERVAL '2 days'
    GROUP BY chute_id
)
SELECT
    i.miner_hotkey,
    COUNT(*) as invocation_count,
    COUNT(DISTINCT(i.chute_id)) AS unique_chute_count,
    COUNT(CASE WHEN i.bounty > 0 THEN 1 END) AS bounty_count,
    sum(
        i.bounty +
        i.compute_multiplier *
        CASE
            WHEN i.metrics->>'steps' IS NOT NULL
                AND r.median_step_time IS NOT NULL
                AND EXTRACT(EPOCH FROM (i.completed_at - i.started_at)) > ((i.metrics->>'steps')::float * r.median_step_time)
            THEN (i.metrics->>'steps')::float * r.median_step_time
            WHEN i.metrics->>'tokens' IS NOT NULL
                AND r.median_token_time IS NOT NULL
                AND EXTRACT(EPOCH FROM (i.completed_at - i.started_at)) > ((i.metrics->>'tokens')::float * r.median_token_time)
            THEN (i.metrics->>'tokens')::float * r.median_token_time
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
