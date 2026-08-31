"""Low-cardinality operational metrics for externally executed work."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


upstream_requests = Counter(
    "external_upstream_requests_total",
    "Outbound external requests by lifecycle phase and outcome.",
    ("phase", "outcome"),
)
upstream_latency = Histogram(
    "external_upstream_request_duration_seconds",
    "Outbound external request latency by lifecycle phase.",
    ("phase",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 180, 600),
)
operation_admissions = Counter(
    "external_operation_admissions_total",
    "External operation admissions by mode and outcome.",
    ("mode", "outcome"),
)
admission_rejections = Counter(
    "external_admission_rejections_total",
    "External operation admissions rejected by governance policy.",
    ("reason",),
)
settlement_attempts = Counter(
    "external_settlement_attempts_total",
    "External settlement attempts by outcome.",
    ("outcome",),
)
billing_delivery_attempts = Counter(
    "external_billing_delivery_attempts_total",
    "Durable external usage events applied to platform billing by outcome.",
    ("outcome",),
)
settlement_backlog = Gauge(
    "external_settlement_backlog",
    "Terminal external operations awaiting successful settlement.",
    ("status",),
    multiprocess_mode="livemostrecent",
)
operation_queue_depth = Gauge(
    "external_operation_queue_depth",
    "External operations awaiting task polling by status.",
    ("status",),
    multiprocess_mode="livemostrecent",
)
oldest_poll_lag = Gauge(
    "external_operation_oldest_poll_lag_seconds",
    "Seconds the oldest due external operation has waited for polling.",
    multiprocess_mode="livemostrecent",
)
retention_deletions = Counter(
    "external_operation_retention_deletions_total",
    "External operation rows removed after their retention period.",
)
governance_bucket_deletions = Counter(
    "external_governance_bucket_deletions_total",
    "Expired external governance rollup buckets removed by maintenance.",
)
artifact_requests = Counter(
    "external_artifact_relay_requests_total",
    "Artifact relay attempts by outcome.",
    ("outcome",),
)
artifact_bytes = Counter(
    "external_artifact_relay_bytes_total",
    "Bytes relayed from retained external artifacts.",
)
circuit_events = Counter(
    "external_circuit_events_total",
    "External account circuit-breaker events by reason and action.",
    ("reason", "action"),
)


def status_class(status_code: int) -> str:
    if 100 <= status_code <= 599:
        return f"{status_code // 100}xx"
    return "invalid"


__all__ = [
    "admission_rejections",
    "artifact_bytes",
    "artifact_requests",
    "billing_delivery_attempts",
    "circuit_events",
    "governance_bucket_deletions",
    "oldest_poll_lag",
    "operation_admissions",
    "operation_queue_depth",
    "retention_deletions",
    "settlement_attempts",
    "settlement_backlog",
    "status_class",
    "upstream_latency",
    "upstream_requests",
]
