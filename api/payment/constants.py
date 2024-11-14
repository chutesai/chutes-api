"""
Payment related constants.
"""

from api.config import settings

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
        "take": settings.miner_take,
        "min_payout": 0.5,
        "limit": None,
    },
    "maintainer": {
        "take": settings.maintainer_take,
        "min_payout": 1.0,
        "addresses": settings.maintainer_payout_addresses,
        "limit": None,
    },
    "contributor": {
        "take": settings.contributor_take,
        "min_payout": 1.0,
        "limit": 100.0,
    },
    "moderator": {
        "take": settings.moderator_take,
        "min_payout": 0.25,
        "limit": None,
    },
    "image_creator": {
        "take": settings.image_creator_take,
        "min_payout": 0.25,
        "limit": None,
    },
    "chute_creator": {
        "take": settings.chute_creator_take,
        "min_payout": 0.25,
        "limit": None,
    },
}
assert sum([item["take"] for item in PAYOUT_STRUCTURE.values()]) == 1.0

# The most expensive compute unit hourly price (h100 sxm5) (in USD).
COMPUTE_UNIT_PRICE_BASIS = 4.0
