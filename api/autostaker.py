import sys
import asyncio
import traceback
from time import sleep
from typing import Optional
from loguru import logger
from api.database import get_session
from api.user.schemas import User
from api.payment.util import decrypt_secret
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


def get_alpha_stake(substrate, coldkey_address, hotkey_address, netuid, block_hash) -> int:
    """
    Get alpha stake amount for a cold/hot key pair on a specific subnet.
    """
    try:
        result = substrate.runtime_call(
            "StakeInfoRuntimeApi",
            "get_stake_info_for_hotkey_coldkey_netuid",
            [hotkey_address, coldkey_address, netuid],
            block_hash=block_hash,
        )
        logger.info(f"DEBUG: get_alpha_stake(..) {result=}")
        if result and result.value and "stake" in result.value:
            return result.value["stake"]
    except Exception as e:
        logger.warning(f"Could not get alpha stake via runtime API: {e}")
        try:
            result = substrate.query(
                module="SubtensorModule",
                storage_function="Alpha",
                params=[netuid, hotkey_address, coldkey_address],
                block_hash=block_hash,
            )
            if result:
                return int(result.value or 0)
        except Exception as e2:
            logger.warning(f"Could not get alpha stake via storage query: {e2}")
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


def _burn_alpha(
    substrate,
    keypair: Keypair,
    hotkey_ss58: Optional[str] = settings.validator_ss58,
    netuid: Optional[int] = settings.netuid,
    amount: Optional[int] = None,
) -> bool:
    """
    Burn alpha after it's staked.
    """
    if netuid == 0:
        logger.error("Cannot burn alpha on root subnet (netuid=0)")
        return False
    logger.info(f"🔥 Preparing to burn alpha on netuid {netuid}...")

    block = substrate.get_block_number(substrate.get_chain_head())
    block_hash = substrate.get_block_hash(block)
    old_alpha_stake = get_alpha_stake(
        substrate, keypair.ss58_address, hotkey_ss58, netuid, block_hash
    )
    if old_alpha_stake == 0:
        logger.info(f"No alpha to burn on netuid {netuid} for {keypair.ss58_address}")
        return True

    # Burn all, if not specified.
    if amount is None:
        burn_amount = old_alpha_stake
        logger.info(
            f"Burning all available alpha: {burn_amount / 10**9:.9f} "
            f"from hotkey: {hotkey_ss58} on netuid: {netuid}"
        )
    else:
        burn_amount = min(amount, old_alpha_stake)
        logger.info(
            f"Burning alpha: {burn_amount / 10**9:.9f} "
            f"from hotkey: {hotkey_ss58} on netuid: {netuid}"
        )

    try:
        call = substrate.compose_call(
            call_module="SubtensorModule",
            call_function="burn_alpha",
            call_params={
                "hotkey": hotkey_ss58,
                "amount": burn_amount,
                "netuid": netuid,
            },
        )
        extrinsic = substrate.create_signed_extrinsic(call=call, keypair=keypair)
        receipt = substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        if not receipt.is_success:
            logger.error(f"Failed to burn alpha: {receipt.error_message=}")
            if receipt.error_message:
                error_name = receipt.error_message.get("name", "")
                if error_name == "SubNetworkDoesNotExist":
                    logger.error(f"Subnet {netuid} does not exist")
                elif error_name == "CannotBurnOrRecycleOnRootSubnet":
                    logger.error("Cannot burn alpha on root subnet")
                elif error_name == "HotKeyAccountNotExists":
                    logger.error(f"Hotkey {hotkey_ss58} does not exist")
                elif error_name == "InsufficientLiquidity":
                    logger.error(f"Insufficient liquidity on subnet {netuid}")
            return False
        logger.success(f"✅ Alpha burn successful: {receipt.is_success=}")
        while (new_block := substrate.get_block_number(substrate.get_chain_head())) == block:
            sleep(3)
        block_hash = substrate.get_block_hash(new_block)
        new_alpha_stake = get_alpha_stake(
            substrate, keypair.ss58_address, hotkey_ss58, netuid, block_hash
        )
        actual_burned = old_alpha_stake - new_alpha_stake
        logger.info(
            f"Alpha stake on netuid {netuid}: "
            f"{old_alpha_stake / 10**9:.9f} → {new_alpha_stake / 10**9:.9f}"
        )
        logger.info(f"Actual alpha burned: {actual_burned / 10**9:.9f}")
        return True

    except Exception as e:
        logger.error(f"Error burning alpha: {e}\n{traceback.format_exc()}")
        return False


@broker.task
async def stake(user_id: str) -> None:
    """
    When a payment is received, automatically begin staking via DCA
    to chutes until the balance is zero, then burn all alpha.
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
        keypair = Keypair.create_from_mnemonic(await decrypt_secret(user.wallet_secret))
    except Exception as exc:
        logger.error(f"Failed to initialize wallet: {exc}")
        return

    consecutive_failures = 0
    substrate = None
    staking_complete = False

    # Phase 1: Stake all TAO
    while not staking_complete:
        amount = settings.autostake_amount
        try:
            if not substrate:
                substrate = SubstrateInterface(url=settings.subtensor)
            available = _add_stake(substrate, keypair, amount=amount)
            if available < amount:
                amount = available
                logger.warning(f"Fallback to lower available balance: {available=} {amount=}")
        except InsufficientBalance:
            logger.success(f"All TAO balance is now staked to {settings.validator_ss58}")
            staking_complete = True
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
                    f"Giving up staking, max consecutive failures reached for {user.user_id=} {keypair.ss58_address=}"
                )
                await settings.redis_client.delete(f"autostake:{user_id}")
                return
        await asyncio.sleep(12)

    # Phase 2: Burn all alpha.
    if staking_complete:
        logger.info(f"🔥 Starting alpha burn phase for {user.user_id=}")
        if not substrate:
            substrate = SubstrateInterface(url=settings.subtensor)
        burn_success = False
        burn_attempts = 0
        max_burn_attempts = 3
        while not burn_success and burn_attempts < max_burn_attempts:
            try:
                burn_attempts += 1
                logger.info(f"Alpha burn attempt {burn_attempts}/{max_burn_attempts}")
                burn_success = _burn_alpha(
                    substrate=substrate,
                    keypair=keypair,
                    hotkey_ss58=settings.validator_ss58,
                    netuid=settings.netuid,
                    amount=None,
                )
                if burn_success:
                    logger.success(
                        f"✅ Successfully burned all alpha for {user.user_id=} on netuid {settings.netuid}"
                    )
                else:
                    logger.warning(f"Alpha burn attempt {burn_attempts} failed, retrying...")
                    await asyncio.sleep(10)
            except Exception as exc:
                logger.error(
                    f"Exception during alpha burn attempt {burn_attempts}: {exc}\n{traceback.format_exc()}"
                )
                await asyncio.sleep(10)
                substrate = SubstrateInterface(url=settings.subtensor)
        if not burn_success:
            logger.error(
                f"Failed to burn alpha after {max_burn_attempts} attempts for {user.user_id=}"
            )

    await settings.redis_client.delete(f"autostake:{user_id}")
    logger.info(f"Auto-staking and alpha burning completed for {user.user_id=}")


async def main():
    await stake(sys.argv[1])


if __name__ == "__main__":
    asyncio.run(main())
