import sys
import asyncio
import traceback
from time import sleep
from typing import Optional
from loguru import logger
from api.database import get_session
from api.user.schemas import User
from api.payment.util import decrypt_wallet_secret
from sqlalchemy.future import select
from api.config import settings
from async_substrate_interface.sync_substrate import SubstrateInterface
from bittensor_wallet.keypair import Keypair
import api.database.orms  # noqa
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend


broker = ListQueueBroker(url=settings.redis_url, queue_name="autostaker").with_result_backend(
    RedisAsyncResultBackend(redis_url=settings.redis_url, result_ex_time=3600)
)


class InsufficientBalance(Exception): ...


def get_balance(substrate, address, block_hash) -> int:
    """
    Get free balance on an account.
    """
    result = substrate.query(
        module="System",
        storage_function="Account",
        params=[address],
        block_hash=block_hash,
    )
    return result["data"]["free"]


def get_stake(substrate, address, block_hash) -> int:
    """
    Get stake amount for an account.
    """
    result = substrate.runtime_call(
        "StakeInfoRuntimeApi",
        "get_stake_info_for_hotkey_coldkey_netuid",
        [settings.validator_ss58, address, settings.netuid],
        block_hash=block_hash,
    )
    logger.info(f"DEBUG: get_stake(..) {result=}")
    if result and result.value and "stake" in result.value:
        return result.value["stake"]
    return 0


def _add_stake(
    substrate,
    keypair: Keypair,
    hotkey_ss58: Optional[str] = settings.validator_ss58,
    netuid: Optional[int] = settings.netuid,
    amount: Optional[float] = settings.autostake_amount,
) -> float:
    """
    Create an subnet an extrinsic to stake to the chutes validator.
    """
    logger.info(f"Syncing with chain: {settings.subtensor}...")
    block = substrate.get_block_number(substrate.get_chain_head())
    block_hash = substrate.get_block_hash(block)
    old_balance = get_balance(substrate, keypair.ss58_address, block_hash)
    old_stake = get_stake(substrate, keypair.ss58_address, block_hash)

    result = substrate.get_constant(
        module_name="Balances",
        constant_name="ExistentialDeposit",
        block_hash=block_hash,
    )
    if result is None:
        raise Exception("Unable to retrieve existential deposit amount.")
    existential_deposit = int(getattr(result, "value", 0)) + 500000
    staking_balance = int(amount * pow(10, 9))
    if staking_balance > old_balance - existential_deposit:
        logger.warning(
            f"Fallback to existential deposit min: {old_balance=} {existential_deposit=}"
        )
        staking_balance = old_balance - existential_deposit
    logger.info(
        f"Using values: {existential_deposit=} {staking_balance=} {old_balance=} {old_stake=}"
    )

    # Check enough to stake.
    if staking_balance > old_balance or staking_balance < 1000000:
        logger.error("Not enough stake:")
        logger.error(f"\t\tbalance:{old_balance}")
        logger.error(f"\t\tamount: {staking_balance}")
        raise InsufficientBalance(
            f"Account {keypair.ss58_address} has insufficient balance to stake."
        )

    # Perform the actual staking operation.
    logger.info(
        f"Staking to netuid: {netuid}, amount: {staking_balance} from {keypair.ss58_address} to {hotkey_ss58}"
    )
    call = substrate.compose_call(
        call_module="SubtensorModule",
        call_function="add_stake",
        call_params={
            "hotkey": hotkey_ss58,
            "amount_staked": staking_balance,
            "netuid": netuid,
            "rate_tolerance": 0.05,
        },
    )
    extrinsic = substrate.create_signed_extrinsic(call=call, keypair=keypair)
    receipt = substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
    if not receipt.is_success:
        logger.error(f"Failed to add stake: {receipt.error_message=}")
        if receipt.error_message and receipt.error_message["name"] == "AmountTooLow":
            raise InsufficientBalance(
                f"Account {keypair.ss58_address} has insufficient balance to stake."
            )
        raise Exception(f"Failed to submit stake extrinsic: {str(receipt.error_message)}")
    logger.success(f"Receipt success: {receipt.is_success=} {receipt.error_message=}")

    # Check balance and stake.
    while (new_block := substrate.get_block_number(substrate.get_chain_head())) == block:
        sleep(3)
    block_hash = substrate.get_block_hash(new_block)
    new_balance = get_balance(substrate, keypair.ss58_address, block_hash)
    new_stake = get_stake(substrate, keypair.ss58_address, block_hash)
    logger.info(f"Balance of {keypair.ss58_address} after stake operation is now {new_balance}")
    logger.info(f"Stake of {keypair.ss58_address} after stake operation is now {new_stake}")
    return (new_balance - existential_deposit) / 10**9


@broker.task
async def stake(user_id: str) -> None:
    """
    When a payment is received, automatically begin staking via DCA
    to chutes until the balance is zero.
    """
    try:
        if not (await settings.redis_client.setnx(f"autostake:{user_id}", "1")):
            logger.warning(f"Staking operation already in progress for {user_id=}")
            return
    finally:
        await settings.redis_client.expire(f"autostake:{user_id}", 60 * 60)

    async with get_session() as session:
        user = (
            (await session.execute(select(User).where(User.user_id == user_id)))
            .unique()
            .scalar_one_or_none()
        )
        if user is None:
            logger.warning(f"User {user_id} not found")
            await settings.redis_client.delete(f"autostake:{user_id}")
            return

    # Load the keypair.
    try:
        keypair = Keypair.create_from_mnemonic(await decrypt_wallet_secret(user.wallet_secret))
    except Exception as exc:
        logger.error(f"Failed to initialize wallet: {exc}")
        return

    consecutive_failures = 0
    substrate = None
    while True:
        amount = settings.autostake_amount
        try:
            if not substrate:
                substrate = SubstrateInterface(url=settings.subtensor)
            available = _add_stake(substrate, keypair, amount=amount)
            if available < amount:
                amount = available
                logger.warning(f"Fallback to lower available balance: {available=} {amount=}")
        except InsufficientBalance:
            logger.success(f"All balance is now staked to {settings.validator_ss58}")
            await settings.redis_client.delete(f"autostake:{user_id}")
            break
        except Exception as exc:
            await asyncio.sleep(30)
            substrate = SubstrateInterface(url=settings.subtensor)
            logger.error(
                f"Unhandled exception performing staking operation: {exc}\n{traceback.format_exc()}"
            )
            consecutive_failures += 1
            if consecutive_failures >= 15:
                logger.error(
                    "Giving up staking, max consecutive failures reached for {user.user_id=} {keypair.ss58_address=}"
                )
                break
        await asyncio.sleep(12)


async def main():
    await stake(sys.argv[1])


if __name__ == "__main__":
    asyncio.run(main())
