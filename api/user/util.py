import re
from typing import Any, Tuple
from substrateinterface import Keypair
from api.payment.util import encrypt_wallet_secret


async def generate_payment_address() -> Tuple[str, str]:
    """
    Generate a new payment address for the user.
    """
    mnemonic = Keypair.generate_mnemonic(words=24)
    keypair = Keypair.create_from_mnemonic(mnemonic)
    payment_address = keypair.ss58_address
    wallet_secret = await encrypt_wallet_secret(mnemonic)
    return payment_address, wallet_secret


def validate_the_username(value: Any) -> str:
    """
    Simple username validation.
    """
    if not isinstance(value, str):
        raise ValueError("Username must be a string")
    if not re.match(r"^[a-zA-Z0-9_]{3,15}$", value):
        raise ValueError(
            "Username must be 3-15 characters and contain only alphanumeric/underscore characters"
        )
    return value
