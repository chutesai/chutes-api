"""
Track the launch-config (deployment attempt) funnel in Prometheus.

Every deployment attempt is a launch_config row that moves through:
    created -> retrieved -> verified  (success)
                         \\-> failed   (with a verification_error)

The DB keeps these rows, but launch_configs.chute_id is ON DELETE CASCADE, and
failed_chute_cleanup.py deletes chutes that failed every deployment attempt -- which
cascade-wipes exactly the most failure-prone chutes' attempt history. So the attempt/failure
*trend* is exported to Prometheus/GMP (24mo retention) where it survives chute deletion. The DB
still holds recent per-row detail for drill-down within the cleanup window.

verification_error is a free-form string set at ~20 call sites, so it cannot be a Prometheus
label directly (unbounded cardinality). classify_verification_error() buckets it into a fixed set.
"""

from enum import Enum

from prometheus_client import Counter


class FailureReason(str, Enum):
    """Bounded set of deployment-failure reason classes (the `reason` label on the failures counter).

    The raw verification_error strings are free-form and unbounded; classify_verification_error()
    maps each into one of these so the Prometheus label cardinality stays fixed.
    """

    UNVERIFIABLE = "unverifiable"
    INVALID_COMMAND = "invalid_command"
    K8S_ENV_CHECK = "k8s_env_check"
    INSPECTO_VERIFICATION = "inspecto_verification"
    FS_VERIFICATION = "fs_verification"
    ENV_TAMPERING = "env_tampering"
    AEGIS_VALIDATION = "aegis_validation"
    RINT_MISSING = "rint_missing"
    RINT_INVALID = "rint_invalid"
    JOB_CLAIMED = "job_claimed"
    V4_REQUIREMENT = "v4_requirement"
    TLS_VALIDATION = "tls_validation"
    GPU_CONFIG = "gpu_config"
    INSTANCE_DELETED = "instance_deleted"
    OTHER = "other"


attempts = Counter(
    "chute_launch_config_attempts_total",
    "Total deployment attempts (launch configs created)",
    ["chute_id"],
)
retrieved = Counter(
    "chute_launch_config_retrieved_total",
    "Launch configs retrieved by a miner (deployment actually started)",
    ["chute_id"],
)
verified = Counter(
    "chute_launch_config_verified_total",
    "Launch configs that passed verification",
    ["chute_id"],
)
failures = Counter(
    "chute_launch_config_failures_total",
    "Launch config verification failures, bucketed by reason class",
    ["chute_id", "reason"],
)


# Ordered (substring, reason-class) rules. First case-insensitive match wins. Keep the set
# bounded -- new free-form error strings fall through to OTHER rather than inflating cardinality.
_REASON_RULES = (
    ("unable to verify", FailureReason.UNVERIFIABLE),
    ("invalid command", FailureReason.INVALID_COMMAND),
    ("kubernetes environment check", FailureReason.K8S_ENV_CHECK),
    ("inspecto environment/lib", FailureReason.INSPECTO_VERIFICATION),
    ("file system verification", FailureReason.FS_VERIFICATION),
    ("mismatched hash", FailureReason.FS_VERIFICATION),
    ("env tampering", FailureReason.ENV_TAMPERING),
    ("aegis", FailureReason.AEGIS_VALIDATION),
    ("missing runtime integrity", FailureReason.RINT_MISSING),
    ("invalid runtime integrity", FailureReason.RINT_INVALID),
    ("already claimed by another miner", FailureReason.JOB_CLAIMED),
    ("must provide v4 rint_commitment", FailureReason.V4_REQUIREMENT),
    ("must provide tls certificate", FailureReason.V4_REQUIREMENT),
    ("tls certificate signature", FailureReason.TLS_VALIDATION),
    ("gpu/nodes configuration", FailureReason.GPU_CONFIG),
    ("instance was deleted", FailureReason.INSTANCE_DELETED),
)


def classify_verification_error(error) -> FailureReason:
    """
    Map a free-form verification_error string to a bounded reason class for Prometheus labels.
    """
    if not error:
        return FailureReason.OTHER
    text = str(error).lower()
    for needle, reason in _REASON_RULES:
        if needle in text:
            return reason
    return FailureReason.OTHER


def track_attempt(chute_id: str) -> None:
    attempts.labels(chute_id=chute_id).inc()


def track_retrieved(chute_id: str) -> None:
    retrieved.labels(chute_id=chute_id).inc()


def track_verified(chute_id: str) -> None:
    verified.labels(chute_id=chute_id).inc()


def track_failure(chute_id: str, error) -> None:
    failures.labels(chute_id=chute_id, reason=classify_verification_error(error).value).inc()
