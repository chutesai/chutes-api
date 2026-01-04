"""
Track general invocation metrics in Prometheus.
"""

from prometheus_client import Counter


usage_usd = Counter(
    "usage_usd_total",
    "Total USD usage charged to users",
    ["chute_id"],
)
compute_seconds = Counter(
    "compute_seconds_total",
    "Total compute seconds across all invocations",
    ["chute_id"],
)


def track_invocation_usage(chute_id: str, balance_used: float, compute_time: float):
    """
    Track USD usage and compute seconds per chute for miner metrics.
    """
    if balance_used > 0:
        usage_usd.labels(chute_id=chute_id).inc(balance_used)
    if compute_time > 0:
        compute_seconds.labels(chute_id=chute_id).inc(compute_time)
