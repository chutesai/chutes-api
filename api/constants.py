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

# Price multiplier to convert compute unit pricing to per-million token pricing.
# This is a bit tricky, since we allow different node selectors potentially for
# any particular model, e.g. you could run a llama 8b on 1 node or 8, so the price
# per million really can change depending on the node selector.
# For example:
#  llama-3-8b with node selector requiring minimally an h100
#  Example h100 hourly price (subject to change): $1.15
#  $/million = $1.15 * 0.016447 = $0.01891405/million input
#            = $1.15 * 0.065785 = $0.07565275/million output
# Deepseek example, 8x h200:
#  $1.9 * 8 * 0.016447 = $0.25/million input
#  $1.9 * 8 * 0.065785 = $1.00/million output
LLM_PRICE_MULT_PER_MILLION_IN = 0.016447
LLM_PRICE_MULT_PER_MILLION_OUT = 0.065785
LLM_MIN_PRICE_IN = 0.01
LLM_MIN_PRICE_OUT = 0.01

# Likewise, for diffusion models, we allow different node selectors and step
# counts, so we can't really have a fixed "per image" pricing, just a price
# that varies based on the node selector and the number of steps requested.
DIFFUSION_PRICE_MULT_PER_STEP = 0.005

# Minimum utilization of a chute before additional instances can be added.
UTILIZATION_SCALE_UP = 0.35
# XXX Legacy value.
EXPANSION_UTILIZATION_THRESHOLD = 0.02

# Cap on number of instances for an underutilized chute.
UNDERUTILIZED_CAP = 5

# Percentage of requests being rate limited to allow scaling up.
RATE_LIMIT_SCALE_UP = 0.02

# Maximum size of VLM asset (video/image).
VLM_MAX_SIZE = 100 * 1024 * 1024
