"""
User event listeners.
"""

import uuid
from sqlalchemy import event
from substrateinterface import Keypair
from api.user.schemas import User
from api.payment.util import encrypt_wallet_secret


def generate_uid(_, __, user: User):
    """
    Set the user_id based on hotkey.
    Runs after fingerprint is generated.
    """
    user.user_id = str(uuid.uuid5(uuid.NAMESPACE_OID, user.fingerprint_hash))


def generate_payment_address(_, __, user: User):
    """
    Generate a new payment address for the user.
    Runs AFTER user_id is generated.
    """
    mnemonic = Keypair.generate_mnemonic(words=24)
    keypair = Keypair.create_from_mnemonic(mnemonic)
    user.payment_address = keypair.ss58_address
    user.wallet_secret = encrypt_wallet_secret(mnemonic)


def ensure_hotkey(_, __, user: User):
    """
    Sets hotkey to zero address if not provided.
    """
    if not user.hotkey:
        user.hotkey = None


def ensure_coldkey(_, __, user: User):
    """
    Sets coldkey to payment address if not provided.
    Runs AFTER payment address is generated.
    """
    if not user.coldkey:
        user.coldkey = user.payment_address


@event.listens_for(User, "before_insert")
def handle_user_insert(_, __, user: User):
    """
    Handle user insert.
    """
    # Order is important here
    generate_uid(_, __, user)
    generate_payment_address(_, __, user)
    ensure_coldkey(_, __, user)
    ensure_hotkey(_, __, user)  # Order not important for this one
