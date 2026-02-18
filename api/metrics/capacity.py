"""
Track capacity (connections per instance vs chute concurrency).
"""

import asyncio
from prometheus_client import Gauge, Counter

mean_connections = Gauge(
    "mean_connections",
    "Average number of concurrent connections to backend nodes",
    ["chute_id"],
    multiprocess_mode="livemostrecent",
)
concurrency = Gauge(
    "concurrency",
    "Maximum concurrency capacity for the model",
    ["chute_id"],
    multiprocess_mode="livemostrecent",
)
utilization = Gauge(
    "utilization",
    "Ratio of mean connections to concurrency (0-1)",
    ["chute_id"],
    multiprocess_mode="livemostrecent",
)
requests_completed = Counter(
    "requests_completed_total",
    "Total number of completed requests",
    ["chute_id"],
)
requests_rate_limited = Counter(
    "requests_rate_limited_total",
    "Total number of rate limited requests",
    ["chute_id"],
)


def _track_capacity(
    chute_id: str,
    mean_conn: float,
    chute_concurrency: int,
    instance_utilization: float = None,
):
    """
    Track connection capacity metrics per chute.
    If instance_utilization is provided (from X-Chutes-Conn-Utilization header),
    use it directly instead of calculating from mean_conn/concurrency.
    """
    mean_connections.labels(chute_id=chute_id).set(mean_conn)
    concurrency.labels(chute_id=chute_id).set(chute_concurrency)
    if instance_utilization is not None:
        utilization.labels(chute_id=chute_id).set(instance_utilization)
    elif chute_concurrency > 0:
        util_ratio = mean_conn / chute_concurrency
        utilization.labels(chute_id=chute_id).set(util_ratio)
    else:
        utilization.labels(chute_id=chute_id).set(0.0)


async def track_capacity(
    chute_id: str,
    mean_conn: float,
    chute_concurrency: int,
    instance_utilization: float = None,
) -> None:
    """
    Async wrapper around the gauge updates for capacity.
    """
    await asyncio.to_thread(
        _track_capacity,
        chute_id=chute_id,
        mean_conn=mean_conn,
        chute_concurrency=chute_concurrency,
        instance_utilization=instance_utilization,
    )


def track_request_completed(chute_id: str):
    """
    Track a completed request.
    """
    requests_completed.labels(chute_id=chute_id).inc()


def track_request_rate_limited(chute_id: str):
    """
    Track a rate limited request.
    """
    requests_rate_limited.labels(chute_id=chute_id).inc()
