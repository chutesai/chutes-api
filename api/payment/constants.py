"""
Payment related constants.
"""

# Payout structure, where:
#  take: ratio of usage-based payouts to allocate to that role
#  min_payout: minimum amount (tao) before a transfer can occur,
#    otherwise stored as pending
#  limit: for pooled/delayed payment types, e.g. contributions
#    are only paid when they happen, the max to keep in the wallet,
#    beyond which the payments are just evenly distributed to
#    other recipients.
PAYOUT_STRUCTURE = {
    "miner": {
        "take": 0.73,
        "min_payout": 0.5,
    },
    "validator": {
        "take": 0.1,
        "min_payout": 1.0,
    },
    "subnet": {
        "take": 0.1,
        "min_payout": 1.0,
    },
    "moderator": {
        "take": 0.02,
        "min_payout": 0.02,
    },
    "contributions": {
        "take": 0.03,
        "min_payout": 1.0,
        "limit": 100.0,
    },
    "image_creator": {
        "take": 0.01,
        "min_payout": 0.25,
    },
    "chute_creator": {
        "take": 0.01,
        "min_payout": 0.25,
    },
}

# The most expensive compute unit hourly price (h100 sxm5) (in USD).
COMPUTE_UNIT_PRICE_BASIS = 2.0
