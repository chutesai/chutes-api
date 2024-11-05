"""
User event listeners.
"""

import uuid
from sqlalchemy import event
from substrateinterface import Keypair
from run_api.user.schemas import User
from run_api.config import settings


@event.listens_for(User, "before_insert")
def generate_uid(_, __, user):
    """
    Set the user_id based on hotkey.
    """
    user.user_id = str(uuid.uuid5(uuid.NAMESPACE_OID, user.hotkey))


@event.listens_for(User, "before_insert")
def generate_payment_address(_, __, user):
    """
    Generate a new payment address for the user.
    """
    mnemonic = Keypair.generate_mnemonic(words=24)
    keypair = Keypair.create_from_mnemonic(mnemonic)
    user.payment_address = keypair.ss58_address
    settings.vault_client.secrets.kv.v2.create_or_update_secret(
        path=f"payments/tao/{user.user_id}",
        secret=dict(mnemonic=mnemonic, address=user.payment_address),
    )
