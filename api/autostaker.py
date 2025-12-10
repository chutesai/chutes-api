import sys
import asyncio
import hashlib
import traceback
from typing import Optional
from loguru import logger
from api.database import get_session
from api.user.schemas import User
from api.payment.util import decrypt_secret
from sqlalchemy.future import select
from api.config import settings
from async_substrate_interface import AsyncSubstrateInterface
from bittensor_wallet.keypair import Keypair
from bittensor_drand import encrypt_mlkem768
import api.database.orms  # noqa
from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend


broker = ListQueueBroker(url=settings.redis_url, queue_name="autostaker").with_result_backend(
    RedisAsyncResultBackend(redis_url=settings.redis_url, result_ex_time=3600)
)


class InsufficientBalance(Exception): ...


async def get_mev_shield_next_key(substrate: AsyncSubstrateInterface) -> Optional[bytes]:
    """Get the ML-KEM-768 public key for MEV shield encryption."""
    try:
        result = await substrate.query(
            module="MevShield",
            storage_function="NextKey",
            params=[],
        )
        if result and result.value:
            return bytes(result.value)
    except Exception as e:
        logger.warning(f"Could not get MEV shield key: {e}")
    return None


async def encrypt_extrinsic(substrate: AsyncSubstrateInterface, signed_extrinsic) -> Optional[object]:
    """Encrypt an extrinsic for MEV protection."""
    ml_kem_768_public_key = await get_mev_shield_next_key(substrate)
    if ml_kem_768_public_key is None:
        logger.warning("MEV Shield NextKey not available on chain, skipping MEV protection")
        return None

    plaintext = bytes(signed_extrinsic.data.data)
    ciphertext = encrypt_mlkem768(ml_kem_768_public_key, plaintext)
    commitment_hash = hashlib.blake2b(plaintext, digest_size=32).digest()
    commitment_hex = "0x" + commitment_hash.hex()

    encrypted_call = await substrate.compose_call(
        call_module="MevShield",
        call_function="submit_encrypted",
        call_params={
            "commitment": commitment_hex,
            "ciphertext": list(ciphertext),
        },
    )
    return encrypted_call


def extract_mev_shield_id(receipt) -> Optional[str]:
    """Extract the MEV shield ID from an extrinsic receipt."""
    try:
        events = receipt.triggered_events if hasattr(receipt, "triggered_events") else []
        for event in events:
            event_data = event.value if hasattr(event, "value") else event
            if isinstance(event_data, dict):
                event_id = event_data.get("event_id") or event_data.get("event", {}).get("event_id")
                if event_id == "EncryptedSubmitted":
                    attrs = event_data.get("attributes") or event_data.get("event", {}).get("attributes", {})
                    return attrs.get("id")
    except Exception as e:
        logger.warning(f"Could not extract MEV shield ID: {e}")
    return None


async def wait_for_mev_extrinsic(
    substrate: AsyncSubstrateInterface,
    extrinsic_hash: str,
    shield_id: str,
    submit_block: int,
    timeout_blocks: int = 3,
) -> tuple[bool, Optional[str]]:
    """Wait for MEV-protected extrinsic to be executed."""
    current_block = submit_block + 1

    while current_block - submit_block <= timeout_blocks:
        logger.info(f"Waiting for MEV shield (block {current_block - submit_block}/{timeout_blocks})...")

        # Wait for the block to exist
        head = await substrate.get_chain_head()
        while await substrate.get_block_number(head) < current_block:
            await asyncio.sleep(3)
            head = await substrate.get_chain_head()

        block_hash = await substrate.get_block_hash(current_block)
        try:
            block_data = await substrate.get_block(block_hash=block_hash)
            extrinsics = block_data.get("extrinsics", []) if block_data else []

            for idx, extrinsic in enumerate(extrinsics):
                ext_hash = f"0x{extrinsic.extrinsic_hash.hex()}" if hasattr(extrinsic, "extrinsic_hash") else None
                if ext_hash == extrinsic_hash:
                    logger.success(f"MEV-protected extrinsic found in block {current_block}")
                    return True, None

                # Check for decryption failure
                ext_value = extrinsic.value if hasattr(extrinsic, "value") else extrinsic
                if isinstance(ext_value, dict):
                    call = ext_value.get("call", {})
                    if (
                        call.get("call_module") == "MevShield"
                        and call.get("call_function") == "mark_decryption_failed"
                    ):
                        call_args = call.get("call_args", [])
                        for arg in call_args:
                            if arg.get("name") == "id" and arg.get("value") == shield_id:
                                return False, "MEV shield decryption failed"
        except Exception as e:
            logger.warning(f"Error checking block {current_block}: {e}")

        current_block += 1

    return False, "MEV shield timeout - inner extrinsic not found"


async def submit_extrinsic(
    substrate: AsyncSubstrateInterface,
    call,
    keypair: Keypair,
    wait_for_inclusion: bool = True,
) -> tuple[bool, Optional[str], Optional[object]]:
    """
    Submit an extrinsic with MEV protection (when available).
    """
    # Create the inner signed extrinsic
    inner_extrinsic = await substrate.create_signed_extrinsic(call=call, keypair=keypair)
    inner_hash = f"0x{inner_extrinsic.extrinsic_hash.hex()}"

    # Try to encrypt for MEV protection
    encrypted_call = await encrypt_extrinsic(substrate, inner_extrinsic)
    if encrypted_call is None:
        # MEV shield not available, submit directly
        logger.info("MEV shield not available, submitting directly")
        receipt = await substrate.submit_extrinsic(inner_extrinsic, wait_for_inclusion=wait_for_inclusion)
        error_msg = None
        if not receipt.is_success:
            error_msg = getattr(receipt, "error_message", None)
        return receipt.is_success, error_msg, receipt

    # Submit the encrypted wrapper
    logger.info("Submitting with MEV protection...")
    wrapper_extrinsic = await substrate.create_signed_extrinsic(call=encrypted_call, keypair=keypair)
    head = await substrate.get_chain_head()
    submit_block = await substrate.get_block_number(head)
    wrapper_receipt = await substrate.submit_extrinsic(wrapper_extrinsic, wait_for_inclusion=True)

    if not wrapper_receipt.is_success:
        return False, f"MEV wrapper submission failed: {wrapper_receipt.error_message}", wrapper_receipt

    # Extract shield ID and wait for inner extrinsic
    shield_id = extract_mev_shield_id(wrapper_receipt)
    if shield_id is None:
        logger.warning("Could not extract shield ID, assuming success")
        return True, None, wrapper_receipt

    success, error = await wait_for_mev_extrinsic(substrate, inner_hash, shield_id, submit_block)
    return success, error, wrapper_receipt


async def get_balance(substrate: AsyncSubstrateInterface, address: str, block_hash: str) -> int:
    """Get free balance on an account."""
    result = await substrate.query(
        module="System",
        storage_function="Account",
        params=[address],
        block_hash=block_hash,
    )
    return result["data"]["free"]


async def get_stake(substrate: AsyncSubstrateInterface, address: str, block_hash: str) -> int:
    """Get stake amount for an account."""
    result = await substrate.runtime_call(
        "StakeInfoRuntimeApi",
        "get_stake_info_for_hotkey_coldkey_netuid",
        [settings.validator_ss58, address, settings.netuid],
        block_hash=block_hash,
    )
    logger.info(f"DEBUG: get_stake(..) {result=}")
    if result and result.value and "stake" in result.value:
        return result.value["stake"]
    return 0


async def get_alpha_stake(
    substrate: AsyncSubstrateInterface,
    coldkey_address: str,
    hotkey_address: str,
    netuid: int,
    block_hash: str,
) -> int:
    """Get alpha stake amount for a cold/hot key pair on a specific subnet."""
    try:
        result = await substrate.runtime_call(
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
            result = await substrate.query(
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


async def _add_stake(
    substrate: AsyncSubstrateInterface,
    keypair: Keypair,
    hotkey_ss58: Optional[str] = None,
    netuid: Optional[int] = None,
    amount: Optional[float] = None,
) -> float:
    """
    Create a subnet extrinsic to stake to the chutes validator.
    """
    hotkey_ss58 = hotkey_ss58 or settings.validator_ss58
    netuid = netuid if netuid is not None else settings.netuid
    amount = amount if amount is not None else settings.autostake_amount

    logger.info(f"Syncing with chain: {settings.subtensor}...")
    head = await substrate.get_chain_head()
    block = await substrate.get_block_number(head)
    block_hash = await substrate.get_block_hash(block)
    old_balance = await get_balance(substrate, keypair.ss58_address, block_hash)
    old_stake = await get_stake(substrate, keypair.ss58_address, block_hash)

    result = await substrate.get_constant(
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
    call = await substrate.compose_call(
        call_module="SubtensorModule",
        call_function="add_stake",
        call_params={
            "hotkey": hotkey_ss58,
            "amount_staked": staking_balance,
            "netuid": netuid,
            "rate_tolerance": 0.05,
        },
    )

    success, error_msg, receipt = await submit_extrinsic(substrate, call, keypair, wait_for_inclusion=True)

    if not success:
        logger.error(f"Failed to add stake: {error_msg}")
        if error_msg and "AmountTooLow" in str(error_msg):
            raise InsufficientBalance(
                f"Account {keypair.ss58_address} has insufficient balance to stake."
            )
        raise Exception(f"Failed to submit stake extrinsic: {error_msg}")
    logger.success(f"Stake extrinsic succeeded")

    # Check balance and stake.
    head = await substrate.get_chain_head()
    new_block = await substrate.get_block_number(head)
    while new_block == block:
        await asyncio.sleep(3)
        head = await substrate.get_chain_head()
        new_block = await substrate.get_block_number(head)

    block_hash = await substrate.get_block_hash(new_block)
    new_balance = await get_balance(substrate, keypair.ss58_address, block_hash)
    new_stake = await get_stake(substrate, keypair.ss58_address, block_hash)
    logger.info(f"Balance of {keypair.ss58_address} after stake operation is now {new_balance}")
    logger.info(f"Stake of {keypair.ss58_address} after stake operation is now {new_stake}")
    return (new_balance - existential_deposit) / 10**9


async def _burn_alpha(
    substrate: AsyncSubstrateInterface,
    keypair: Keypair,
    hotkey_ss58: Optional[str] = None,
    netuid: Optional[int] = None,
    amount: Optional[int] = None,
) -> bool:
    """
    Burn alpha after it's staked.
    Note: Burning doesn't need MEV protection since it's not subject to sandwich attacks.
    """
    hotkey_ss58 = hotkey_ss58 or settings.validator_ss58
    netuid = netuid if netuid is not None else settings.netuid

    if netuid == 0:
        logger.error("Cannot burn alpha on root subnet (netuid=0)")
        return False
    logger.info(f"🔥 Preparing to burn alpha on netuid {netuid}...")

    head = await substrate.get_chain_head()
    block = await substrate.get_block_number(head)
    block_hash = await substrate.get_block_hash(block)
    old_alpha_stake = await get_alpha_stake(
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
        call = await substrate.compose_call(
            call_module="SubtensorModule",
            call_function="burn_alpha",
            call_params={
                "hotkey": hotkey_ss58,
                "amount": burn_amount,
                "netuid": netuid,
            },
        )

        extrinsic = await substrate.create_signed_extrinsic(call=call, keypair=keypair)
        receipt = await substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)

        if not receipt.is_success:
            error_msg = getattr(receipt, "error_message", None)
            logger.error(f"Failed to burn alpha: {error_msg}")
            if error_msg:
                if "SubNetworkDoesNotExist" in str(error_msg):
                    logger.error(f"Subnet {netuid} does not exist")
                elif "CannotBurnOrRecycleOnRootSubnet" in str(error_msg):
                    logger.error("Cannot burn alpha on root subnet")
                elif "HotKeyAccountNotExists" in str(error_msg):
                    logger.error(f"Hotkey {hotkey_ss58} does not exist")
                elif "InsufficientLiquidity" in str(error_msg):
                    logger.error(f"Insufficient liquidity on subnet {netuid}")
            return False

        logger.success(f"✅ Alpha burn successful")

        head = await substrate.get_chain_head()
        new_block = await substrate.get_block_number(head)
        while new_block == block:
            await asyncio.sleep(3)
            head = await substrate.get_chain_head()
            new_block = await substrate.get_block_number(head)

        block_hash = await substrate.get_block_hash(new_block)
        new_alpha_stake = await get_alpha_stake(
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


async def _move_stake(
    substrate: AsyncSubstrateInterface,
    keypair: Keypair,
    origin_hotkey: str,
    origin_netuid: int,
    destination_hotkey: str,
    destination_netuid: int,
    amount: Optional[int] = None,
) -> bool:
    """
    Move stake from one hotkey/netuid to another.
    Used for converting incoming alpha token payments to our subnet stake.
    """
    logger.info(
        f"Moving stake from {origin_hotkey}:{origin_netuid} to {destination_hotkey}:{destination_netuid}"
    )

    head = await substrate.get_chain_head()
    block_hash = await substrate.get_block_hash(await substrate.get_block_number(head))

    # Get current stake at origin
    origin_stake = await get_alpha_stake(
        substrate, keypair.ss58_address, origin_hotkey, origin_netuid, block_hash
    )
    if origin_stake == 0:
        logger.warning(f"No stake to move from {origin_hotkey}:{origin_netuid}")
        return True

    move_amount = amount if amount is not None else origin_stake
    move_amount = min(move_amount, origin_stake)

    logger.info(f"Moving {move_amount / 10**9:.9f} stake")

    try:
        call = await substrate.compose_call(
            call_module="SubtensorModule",
            call_function="move_stake",
            call_params={
                "origin_hotkey": origin_hotkey,
                "origin_netuid": origin_netuid,
                "destination_hotkey": destination_hotkey,
                "destination_netuid": destination_netuid,
                "alpha_amount": move_amount,
            },
        )

        success, error_msg, receipt = await submit_extrinsic(substrate, call, keypair, wait_for_inclusion=True)

        if not success:
            logger.error(f"Failed to move stake: {error_msg}")
            return False

        logger.success(f"✅ Stake move successful")
        return True

    except Exception as e:
        logger.error(f"Error moving stake: {e}\n{traceback.format_exc()}")
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

    try:
        keypair = Keypair.create_from_mnemonic(await decrypt_secret(user.wallet_secret))
    except Exception as exc:
        logger.error(f"Failed to initialize wallet: {exc}")
        return

    consecutive_failures = 0
    staking_complete = False

    # Phase 1: Stake all TAO
    async with AsyncSubstrateInterface(url=settings.subtensor) as substrate:
        while not staking_complete:
            amount = settings.autostake_amount
            try:
                available = await _add_stake(substrate, keypair, amount=amount)
                if available < amount:
                    amount = available
                    logger.warning(f"Fallback to lower available balance: {available=} {amount=}")
            except InsufficientBalance:
                logger.success(f"All TAO balance is now staked to {settings.validator_ss58}")
                staking_complete = True
                break
            except Exception as exc:
                await asyncio.sleep(30)
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
            burn_success = False
            burn_attempts = 0
            max_burn_attempts = 3
            while not burn_success and burn_attempts < max_burn_attempts:
                try:
                    burn_attempts += 1
                    logger.info(f"Alpha burn attempt {burn_attempts}/{max_burn_attempts}")
                    burn_success = await _burn_alpha(
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
            if not burn_success:
                logger.error(
                    f"Failed to burn alpha after {max_burn_attempts} attempts for {user.user_id=}"
                )

    await settings.redis_client.delete(f"autostake:{user_id}")
    logger.info(f"Auto-staking and alpha burning completed for {user.user_id=}")


@broker.task
async def process_alpha_payment(
    user_id: str,
    origin_hotkey: str,
    origin_netuid: int,
    amount_rao: int,
) -> None:
    """
    Process an incoming alpha token payment by moving the stake to chutes, then burning.
    """
    try:
        lock_key = f"alpha_payment:{user_id}:{origin_netuid}:{amount_rao}"
        if not (await settings.redis_client.setnx(lock_key, "1")):
            logger.warning(f"Alpha payment already being processed for {user_id=}")
            return
    finally:
        await settings.redis_client.expire(lock_key, 60 * 60)

    async with get_session() as session:
        user = (
            (await session.execute(select(User).where(User.user_id == user_id)))
            .unique()
            .scalar_one_or_none()
        )
        if user is None:
            logger.warning(f"User {user_id} not found")
            await settings.redis_client.delete(lock_key)
            return

    # Load the keypair.
    try:
        keypair = Keypair.create_from_mnemonic(await decrypt_secret(user.wallet_secret))
    except Exception as exc:
        logger.error(f"Failed to initialize wallet: {exc}")
        return

    async with AsyncSubstrateInterface(url=settings.subtensor) as substrate:
        # Step 1: Move stake from origin netuid to our validator on our netuid
        move_success = await _move_stake(
            substrate=substrate,
            keypair=keypair,
            origin_hotkey=origin_hotkey,
            origin_netuid=origin_netuid,
            destination_hotkey=settings.validator_ss58,
            destination_netuid=settings.netuid,
            amount=amount_rao,
        )

        if not move_success:
            logger.error(f"Failed to move stake for alpha payment from {origin_netuid} to {settings.netuid}")
            await settings.redis_client.delete(lock_key)
            return

        # Step 2: Burn ALL alpha on our validator (not just the moved amount)
        burn_success = await _burn_alpha(
            substrate=substrate,
            keypair=keypair,
            hotkey_ss58=settings.validator_ss58,
            netuid=settings.netuid,
            amount=None,  # Burn all
        )
        if not burn_success:
            logger.error(f"Failed to burn alpha after move for {user.user_id=}")

    await settings.redis_client.delete(lock_key)
    logger.info(f"Alpha payment processing completed for {user.user_id=}")


async def main():
    await stake(sys.argv[1])


if __name__ == "__main__":
    asyncio.run(main())
