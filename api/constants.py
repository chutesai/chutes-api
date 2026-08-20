from enum import Enum


class NoncePurpose(str, Enum):
    """Purpose values for attestation nonces to prevent cross-purpose reuse."""

    BOOT = "boot"
    RUNTIME = "runtime"
    INSTANCE_VERIFICATION = "instance_verification"


class ServerHealthStatus(str, Enum):
    """TEE server liveness, derived live from servers.last_health_at by Server.health_status."""

    HEALTHY = "healthy"  # last successful probe within the degraded threshold
    DEGRADED = "degraded"  # no comms past the degraded threshold (e.g. >12h)
    OFFLINE = "offline"  # no comms past the offline threshold (e.g. >72h)
    UNKNOWN = "unknown"  # last_health_at IS NULL — never seen healthy


# Attestation-proxy health endpoint (HTTPS, self-signed cert: CN=attestation-service).
ATTESTATION_PROXY_PORT = 30443
ATTESTATION_PROXY_HEALTH_PATH = "/health"


ZERO_ADDRESS_HOTKEY = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"  # Public key is 0x00000...
HOTKEY_HEADER = "X-Chutes-Hotkey"
COLDKEY_HEADER = "X-Chutes-Coldkey"
SIGNATURE_HEADER = "X-Chutes-Signature"
NONCE_HEADER = "X-Chutes-Nonce"
# RSA operator signature over the quote nonce, used only by the initramfs (signed) attestation
# path where sr25519 is unavailable. Distinct from SIGNATURE_HEADER so the rc gate can tell the
# signed proof from the sr25519 request signature by header presence alone (see
# extract_attestation_auth / authorize_rc_measurement).
OPERATOR_SIGNATURE_HEADER = "X-Operator-Signature"
AUTHORIZATION_HEADER = "Authorization"
PURPOSE_HEADER = "X-Chutes-Purpose"
MINER_HEADER = "X-Chutes-Miner"
VALIDATOR_HEADER = "X-Chutes-Validator"
ENCRYPTED_HEADER = "X-Chutes-Encrypted"
ATTESTATION_SIGNATURE_HEADER = "X-Signature"
# Injected by the attestation nginx proxy (tdx-attestation.chutes.ai), carrying
# ATTESTATION_PROXY_SECRET; proves an attestation request arrived via that proxy. Used by legacy
# 1.3.x VMs (see require_attestation_proxy).
ATTESTATION_PROXY_AUTH_HEADER = "X-Attestation-Proxy-Auth"
# Injected by the cvm nginx proxy (cvm.chutes.ai), carrying CVM_PROXY_SECRET; proves a request
# arrived via the full-mTLS CVM proxy used by 1.4.0+ VMs. A match marks the request as
# mTLS-verified (see require_attestation_proxy).
CVM_PROXY_AUTH_HEADER = "X-Cvm-Proxy-Auth"
# Injected by the registry-proxy nginx frontend, carrying REGISTRY_PROXY_SECRET; proves a
# /registry/auth subrequest arrived via the registry proxy (see require_registry_proxy_secret).
REGISTRY_PROXY_AUTH_HEADER = "X-Registry-Proxy-Auth"

# LUKS volume names allowed in GET/POST (extendable)
SUPPORTED_LUKS_VOLUMES = ("storage", "tdx-cache")

# The storage volume's first-boot state determines whether a new k3s encryption
# key must be generated (luksFormat on a raw device vs. luksOpen on existing LUKS).
LUKS_STORAGE_VOLUME = "storage"

# Minimum VM image version that supports root volume LUKS passphrase rotation.
# VMs on versions below this receive root_next=null in boot attestation responses.
MIN_ROOT_ROTATION_VERSION = "1.4.0"

# Minimum VM image version whose firmware registers the per-VM ephemeral auth key (vm_auth_ss58,
# returned by boot attestation) as an allowed signer for validator->VM calls. Older firmware
# (1.3.x) only trusts the validator key, so the validator MUST sign those calls with its own
# keypair -- signing with the ephemeral key 401s on the VM. Gates both key generation (boot) and
# key usage (TeeServerClient.create).
MIN_VM_AUTH_KEY_VERSION = "1.4.0"

# Host profile submissions (POST /servers/tdx/host_profiles). discover-profile.sh output is a few
# KB (pci_topology, the lspci tree, dominates), so the cap is generous but still bounds what a
# single miner can push into the bucket. Both rate limits are counted only AFTER the signature
# verifies, so a forged hotkey header can't burn a real miner's quota.
HOST_PROFILE_MAX_BYTES = 256 * 1024

# Bounds on the modeled fields of a submitted profile. discover-profile.sh produces
# machine-generated values (PCI device ids are the 4 hex chars lspci prints, the processor id is
# 16 hex chars of CPUID leaf-1, cpu_args is "host" or "host,-avx10"), but a submission is
# attacker-controlled data from a miner: these fields end up in object metadata -- i.e. HTTP
# request headers -- in log lines, and in front of an offline generation job running with more
# privilege than this API. None of them may be unbounded or free-form. Ranges are generous enough
# that plausible future hardware still validates.
HOST_PROFILE_MAX_GPUS = 64
HOST_PROFILE_MAX_NUMA_NODES = 64
HOST_PROFILE_MAX_SOCKETS = 64
HOST_PROFILE_MAX_CPUS = 8192
HOST_PROFILE_MAX_THREADS_PER_CORE = 16
HOST_PROFILE_MAX_RAM_GB = 262144
HOST_PROFILE_MAX_VRAM_GB = 65536
HOST_PROFILE_MAX_BAR_MB = 16 * 1024 * 1024
HOST_PROFILE_MAX_NICS = 256
# The lspci -tv tree; a few KB on a large box, bounded well under the whole-body cap.
HOST_PROFILE_MAX_TOPOLOGY_CHARS = 64 * 1024
HOST_PROFILE_SUBMISSIONS_PER_HOTKEY = 10
HOST_PROFILE_SUBMISSIONS_GLOBAL = 120
HOST_PROFILE_WINDOW_SECONDS = 3600

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

# Default discount for cached prompt tokens (90% off).
DEFAULT_CACHE_DISCOUNT = 0.9

# Likewise, for diffusion models, we allow different node selectors and step
# counts, so we can't really have a fixed "per image" pricing, just a price
# that varies based on the node selector and the number of steps requested.
DIFFUSION_PRICE_MULT_PER_STEP = 0.002

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

# Cooldown between bounty creations per chute to prevent race conditions.
BOUNTY_COOLDOWN_SECONDS = 600

# How long a bounty (and the matching warmup demand window) stays open, per chute type. Public
# chutes keep a long window; private non-legacy chutes get a short one so users aren't billed for
# idle time (affine chutes get a slightly longer window). The warmup request->hot correlation key
# uses the same value so the two never drift -- see bounty_lifetime_for().
BOUNTY_LIFETIME_PUBLIC = 86400
BOUNTY_LIFETIME_PRIVATE = 3600
BOUNTY_LIFETIME_AFFINE = 7200

# Maximum size of VLM asset (video/image).
VLM_MAX_SIZE = 100 * 1024 * 1024

# Private instance compute multiplier bonus.
PRIVATE_INSTANCE_BONUS = 2
INTEGRATED_SUBNET_BONUS = 3
TEE_PRIVATE_INSTANCE_BONUS = 1.3

# TEE bonus.
TEE_BONUS = 2.25

# Duration for instance disablement when consecutive errors are hit (increases linearly until max).
INSTANCE_DISABLE_BASE_TIMEOUT = 90

# Number of times an instance can be disabled within a sliding 1-hour window before deletion.
MAX_INSTANCE_DISABLES = 5

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
        "max_public_chutes": 3,
    },
    "babelbit": {
        "netuid": 59,
        "model_substring": "babelbit",
        "max_public_chutes": 3,
    },
    "chronoseek": {
        "netuid": 20,
        "model_substring": "chronoseek",
        "max_public_chutes": 3,
        "source_public": False,
    },
    "glyph": {
        "netuid": 117,
        "model_substring": "glyph",
        "max_public_chutes": 3,
    },
    "leoma": {
        "netuid": 99,
        "model_substring": "leoma",
        "max_public_chutes": 3,
    },
    "prometheon": {
        "netuid": 108,
        "model_substring": "prometheon",
        "max_public_chutes": 3,
    },
    "score": {
        "netuid": 44,
        "model_substring": "turbovision",
        "max_public_chutes": 3,
    },
    "vocence": {
        "netuid": 78,
        "model_substring": "vocence",
        "max_public_chutes": 3,
    },
}


def is_chute_source_public(name: str) -> bool:
    """Return whether a chute name maps to publicly visible source code."""
    normalized_name = (name or "").lower()
    for config in INTEGRATED_SUBNETS.values():
        if (
            config["model_substring"] in normalized_name
            and config.get("source_public", True) is False
        ):
            return False
    return True


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
