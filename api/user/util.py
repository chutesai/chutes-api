import re
from typing import Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from substrateinterface import Keypair, SubstrateInterface
from loguru import logger
from api.config import settings
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


async def return_developer_deposit(self, session: AsyncSession, destination: str):
    """
    Return the developer deposit.
    """
    # Discover the balance - we're returning all of it, whatever they sent.
    substrate = SubstrateInterface(url=settings.subtensor)
    result = substrate.query(
        module="System",
        storage_function="Account",
        params=[self.developer_payment_address],
    )
    balance = 0.0
    if result:
        balance = result.value["data"]["free"]
    if not balance:
        message = f"Wallet {self.developer_payment_address} does not have any free balance!"
        logger.warning(message)
        return False, message

    # Calculate the fee.
    keypair = Keypair.create_from_mnemonic(
        await decrypt_wallet_secret(self.developer_wallet_secret)
    )
    call = substrate.compose_call(
        call_module="Balances",
        call_function="transfer",
        call_params={
            "dest": destination,
            "value": balance,
        },
    )
    payment_info = substrate.get_payment_info(call, keypair)
    fee = payment_info.get("partialFee", 0)
    transfer_amount = balance - fee
    if transfer_amount <= 0:
        message = (
            f"Balance would become negative: {balance=}, {fee=}, {self.user_id=} {destination=}"
        )
        logger.warning(message)
        return False, message

    # Perform the actual transfer, but remove developer permissions first.
    Permissioning.disable(self, Permissioning.developer)
    session.commit()
    await session.refresh(self)
    logger.info(
        f"Transfer of {transfer_amount} rao to {destination} from {self.user_id=} {self.developer_payment_address=}"
    )
    call = substrate.compose_call(
        call_module="Balances",
        call_function="transfer",
        call_params={"dest": destination, "value": transfer_amount},
    )
    extrinsic = substrate.create_signed_extrinsic(call=call, keypair=keypair)
    receipt = substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
    message = "\n".join(
        [
            f"Return of developer deposit for {self.user_id=} successful!",
            f"Block hash: {receipt.block_hash}",
            f"Amount transferred: {transfer_amount} rao",
            f"Fee paid: {fee} rao",
        ]
    )
    logger.success(message)
    return True, message
