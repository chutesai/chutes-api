import re
from typing import Any, Tuple
from sqlalchemy import select
from substrateinterface import Keypair, SubstrateInterface
from loguru import logger
from api.config import settings
from api.database import get_session
from api.permissions import Permissioning
from api.payment.util import encrypt_wallet_secret, decrypt_wallet_secret


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


async def refund_deposit(user_id: str, destination: str):
    """
    Return the developer deposit.
    """
    from api.user.schemas import User

    async with get_session() as session:
        user = (
            await session.execute(select(User).where(User.user_id == user_id))
        ).scalar_one_or_none()

        # Discover the balance - we're returning all of it, whatever they sent.
        substrate = SubstrateInterface(url=settings.subtensor)
        result = substrate.query(
            module="System",
            storage_function="Account",
            params=[user.developer_payment_address],
        )
        balance = 0.0
        if result:
            balance = result.value["data"]["free"]
        if not balance:
            message = f"Wallet {user.developer_payment_address} does not have any free balance!"
            logger.warning(message)
            return False, message

        keypair = Keypair.create_from_mnemonic(
            await decrypt_wallet_secret(user.developer_wallet_secret)
        )
        call = substrate.compose_call(
            call_module="Balances",
            call_function="transfer_all",
            call_params={
                "dest": destination,
                "keep_alive": False,
            },
        )

        # Perform the actual transfer, but remove developer permissions first.
        Permissioning.disable(user, Permissioning.developer)
        await session.commit()
        await session.refresh(user)
        logger.info(
            f"Transfer of {balance} rao (minus fee) to {destination} from {user.user_id=} {user.developer_payment_address=} incoming..."
        )
        extrinsic = substrate.create_signed_extrinsic(call=call, keypair=keypair)
        receipt = substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        message = "\n".join(
            [
                f"Return of developer deposit for {user.user_id=} successful!",
                f"Block hash: {receipt.block_hash}",
                f"Amount transferred: {balance} rao (minus fee)",
            ]
        )
        logger.success(message)
        return True, message
