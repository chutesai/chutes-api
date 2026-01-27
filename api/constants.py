from enum import Enum


class NoncePurpose(str, Enum):
    """Purpose values for attestation nonces to prevent cross-purpose reuse."""
    
    BOOT = "boot"
    RUNTIME = "runtime"
    INSTANCE_VERIFICATION = "instance_verification"


ZERO_ADDRESS_HOTKEY = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"  # Public key is 0x00000...
HOTKEY_HEADER = "X-Chutes-Hotkey"
COLDKEY_HEADER = "X-Chutes-Coldkey"
SIGNATURE_HEADER = "X-Chutes-Signature"
NONCE_HEADER = "X-Chutes-Nonce"
AUTHORIZATION_HEADER = "Authorization"
PURPOSE_HEADER = "X-Chutes-Purpose"
MINER_HEADER = "X-Chutes-Miner"
VALIDATOR_HEADER = "X-Chutes-Validator"
ENCRYPTED_HEADER = "X-Chutes-Encrypted"

# Min balance to register via the CLI (tao units)
MIN_REG_BALANCE = 0.25

# Price multiplier to convert compute unit pricing to per-million token pricing.
# This is a bit tricky, since we allow different node selectors potentially for
# any particular model, e.g. you could run a llama 8b on 1 node or 8, so the price
# per million really can change depending on the node selector.
# For example:
#  llama-3-8b with node selector requiring minimally an h100
#  Example h100 hourly price (subject to change): $1.5
#  $/million = $1.5 * 0.01358695 = $0.02/million input
#            = $1.5 * 0.05434782 = $0.08/million output
# Deepseek example, 8x h200:
#  $2.3 * 8 * 0.01358695 = $0.25/million input
#  $2.3 * 8 * 0.05434782 = $1.00/million output
# NOTE: there is also a multiplier when the chute's concurrency is < 16,
# because for example the concurrency may be reduced to accomodate more
# total concurrent tokens in KV cache, such as GLM-4.5-FP8 at full context
# has concurrency 12, so:
#  $2.3 * 8 * 16/12 * 0.01358695 = $0.33/million input
#  $2.3 * 8 * 16/12 * 0.05434782 = $1.33/million output
# Kimi-K2 example (8xb200)
#  $3.5 * 8 * 0.01358695 = $0.38
#  $3.5 * 8 * 0.05434782 = $1.52
LLM_PRICE_MULT_PER_MILLION_IN = 0.01358695
LLM_PRICE_MULT_PER_MILLION_OUT = 0.05434782
LLM_MIN_PRICE_IN = 0.01
LLM_MIN_PRICE_OUT = 0.01

# Likewise, for diffusion models, we allow different node selectors and step
# counts, so we can't really have a fixed "per image" pricing, just a price
# that varies based on the node selector and the number of steps requested.
DIFFUSION_PRICE_MULT_PER_STEP = 0.005

# Minimum utilization of a chute before additional instances can be added.
UTILIZATION_SCALE_UP = 0.5

# Utilization threshold below which scale-down is considered.
# Gap between SCALE_DOWN and SCALE_UP creates a "stable zone".
UTILIZATION_SCALE_DOWN = 0.2

# Cap on number of instances for an underutilized public chute.
UNDERUTILIZED_CAP = 2

# Percentage of requests being rate limited to allow scaling up.
RATE_LIMIT_SCALE_UP = 0.03

# Scale-down moving average parameters.
# How far back to look in capacity_log for trend analysis.
SCALE_DOWN_LOOKBACK_MINUTES = 90
# Can't drop more than this ratio below the rolling average target count.
SCALE_DOWN_MAX_DROP_RATIO = 0.6

# Maximum size of VLM asset (video/image).
VLM_MAX_SIZE = 100 * 1024 * 1024

# Private instance compute multiplier bonus.
PRIVATE_INSTANCE_BONUS = 2
INTEGRATED_SUBNET_BONUS = 3

# TEE bonus.
TEE_BONUS = 2.25

# Duration for instance disablement when consecutive errors are hit (increases linearly until max).
INSTANCE_DISABLE_BASE_TIMEOUT = 90

# Number of times an instance can be disabled before deleting (after consecutive failures).
MAX_INSTANCE_DISABLES = 2

# Cascade failure detection: if more than this many instances are pending deletion
# within the detection window, assume network outage and skip deletions.
CASCADE_FAILURE_THRESHOLD = 50

# How long to wait before checking for cascade failures (seconds).
CASCADE_DETECTION_DELAY = 45

# TTL for pending deletion markers (seconds).
CASCADE_PENDING_TTL = 75

# IDP/OAuth2 style login constants.
MAX_REFRESH_TOKEN_LIFETIME_DAYS = 30
DEFAULT_REFRESH_TOKEN_LIFETIME_DAYS = 30
ACCESS_TOKEN_EXPIRY_SECONDS = 3600
AUTH_CODE_EXPIRY_SECONDS = 600
LOGIN_NONCE_EXPIRY_SECONDS = 300

# Subnet integrations.
INTEGRATED_SUBNETS = {
    "affine": {
        "netuid": 120,
        "model_substring": "affine",
    },
    "babelbit": {
        "netuid": 59,
        "model_substring": "babelbit",
    },
    "score": {
        "netuid": 44,
        "model_substring": "turbovision",
    },
}

# Chute utilization query.
CHUTE_UTILIZATION_QUERY = """
WITH chute_details AS (
    SELECT
        c.chute_id,
        CASE WHEN c.public IS true THEN c.name ELSE '[private chute]' END AS name,
        COUNT(i.instance_id) AS total_instance_count,
        COUNT(i.instance_id) FILTER (WHERE i.active IS true) AS active_instance_count
    FROM chutes c
    LEFT JOIN instances i ON c.chute_id = i.chute_id
    LEFT JOIN rolling_updates ru ON c.chute_id = ru.chute_id
    GROUP BY c.chute_id, c.name, c.public
),
latest_logs AS (
    SELECT
        cd.chute_id,
        ll.timestamp,
        ll.utilization_current,
        ll.utilization_5m,
        ll.utilization_15m,
        ll.utilization_1h,
        ll.rate_limit_ratio_5m,
        ll.rate_limit_ratio_15m,
        ll.rate_limit_ratio_1h,
        ll.total_requests_5m,
        ll.total_requests_15m,
        ll.total_requests_1h,
        ll.completed_requests_5m,
        ll.completed_requests_15m,
        ll.completed_requests_1h,
        ll.rate_limited_requests_5m,
        ll.rate_limited_requests_15m,
        ll.rate_limited_requests_1h,
        ll.instance_count,
        ll.action_taken,
        ll.target_count,
        ll.effective_multiplier
    FROM chute_details cd
    CROSS JOIN LATERAL (
        SELECT
            timestamp,
            utilization_current,
            utilization_5m,
            utilization_15m,
            utilization_1h,
            rate_limit_ratio_5m,
            rate_limit_ratio_15m,
            rate_limit_ratio_1h,
            total_requests_5m,
            total_requests_15m,
            total_requests_1h,
            completed_requests_5m,
            completed_requests_15m,
            completed_requests_1h,
            rate_limited_requests_5m,
            rate_limited_requests_15m,
            rate_limited_requests_1h,
            instance_count,
            action_taken,
            target_count,
            effective_multiplier
        FROM capacity_log cl
        WHERE cl.chute_id = cd.chute_id
        ORDER BY cl.timestamp DESC
        LIMIT 1
    ) ll
)
SELECT
    cd.chute_id,
    cd.name,
    ll.timestamp,
    ll.utilization_current,
    ll.utilization_5m,
    ll.utilization_15m,
    ll.utilization_1h,
    ll.rate_limit_ratio_5m,
    ll.rate_limit_ratio_15m,
    ll.rate_limit_ratio_1h,
    ll.total_requests_5m,
    ll.total_requests_15m,
    ll.total_requests_1h,
    ll.completed_requests_5m,
    ll.completed_requests_15m,
    ll.completed_requests_1h,
    ll.rate_limited_requests_5m,
    ll.rate_limited_requests_15m,
    ll.rate_limited_requests_1h,
    ll.instance_count,
    ll.action_taken,
    ll.target_count,
    ll.effective_multiplier,
    cd.total_instance_count,
    cd.active_instance_count
FROM chute_details cd
JOIN latest_logs ll ON cd.chute_id = ll.chute_id
ORDER BY ll.total_requests_1h DESC;
"""
