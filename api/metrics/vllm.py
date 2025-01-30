"""
Track LLM usage metrics in prometheus (in addition to DB).
"""

from prometheus_client import Counter, Histogram, Gauge


total_requests = Counter("requests", "Total requests", ["chute_id", "error", "miner_hotkey"])
request_duration = Histogram(
    "duration",
    "Duration of each request in seconds",
    ["chute_id", "error", "miner_hotkey"],
)
tokens_per_second = Gauge(
    "tps", "Average tokens processed per second", ["chute_id", "miner_hotkey"]
)
prompt_tokens = Counter(
    "prompt_tokens_total", "Total number of prompt tokens processed", ["chute_id", "miner_hotkey"]
)
completion_tokens = Counter(
    "completion_tokens_total",
    "Total number of completion tokens generated",
    ["chute_id", "miner_hotkey"],
)
time_to_first_token = Gauge(
    "time_to_first_token_seconds",
    "Time taken to receive first token from start of request",
    ["chute_id", "miner_hotkey"],
)


def track_usage(chute_id: str, miner_hotkey: str, duration: float, metrics: dict = {}):
    if not metrics or not metrics.get("it") or not metrics.get("tps"):
        return

    tokens_per_second.labels(chute_id=chute_id, miner_hotkey=miner_hotkey).set(metrics["tps"])
    time_to_first_token.labels(chute_id=chute_id, miner_hotkey=miner_hotkey).set(
        metrics.get("ttft", 0.0)
    )
    prompt_tokens.labels(chute_id=chute_id, miner_hotkey=miner_hotkey).inc(metrics["it"])
    completion_tokens.labels(chute_id=chute_id, miner_hotkey=miner_hotkey).inc(metrics["ot"])
