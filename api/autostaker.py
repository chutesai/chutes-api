"""
Cronjob-based autostaker for processing pending stakes.

Runs every N minute(s) and:
1. Fetches wallets with pending_balance > 0 from pending_stakes table
2. Reconciles local balance with on-chain data
3. Performs stake operations in random chunks to avoid MEV pattern detection
4. Burns alpha after staking is complete
"""

import asyncio
import hashlib
import random
import traceback

from datetime import datetime, timedelta, timezone
from typing import Optional

from async_substrate_interface import AsyncSubstrateInterface
from bittensor_drand import encrypt_mlkem768
from bittensor_wallet.keypair import Keypair
from loguru import logger
from sqlalchemy import select, update, and_, or_, func, case
from sqlalchemy.dialects.postgresql import insert

from api.config import settings
from api.database import get_session
from api.payment.schemas import PendingStake
from api.payment.util import decrypt_secret
from api.user.schemas import User
import api.database.orms  # noqa


# Constants
ONE_TAO_RAO = 10**9  # 1 TAO = 1e9 rao
MAX_STAKE_PER_ITERATION_TAO = 25  # Max TAO worth to stake per iteration
MIN_STAKE_TAO = 0.1  # Minimum stake amount
MAX_SLIPPAGE_PERCENT = 0.003  # 0.3% max slippage before chunking
AUTOSTAKER_CONCURRENCY = 24  # Max number of wallets processed concurrently
STALE_BASE_MINUTES = 15  # Default stale threshold for "processing" rows
STALE_MAX_MINUTES = 60  # Upper bound for adaptive stale threshold
EXPECTED_SECONDS_PER_WALLET = 45  # Heuristic for adaptive stale calculation


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
            value = result.value
            # Handle tuple format (key_bytes, round_number)
            if isinstance(value, (tuple, list)):
                value = value[0]
            return bytes(value)
    except Exception as e:
        logger.warning(f"Could not get MEV shield key: {e}")
    return None


async def encrypt_extrinsic(
    substrate: AsyncSubstrateInterface, signed_extrinsic
) -> Optional[object]:
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
            "ciphertext": ciphertext,
        },
    )
    return encrypted_call


async def extract_mev_shield_id(receipt) -> Optional[str]:
    """Extract the MEV shield ID from an extrinsic receipt."""
    try:
        events = await receipt.triggered_events
        for event in events:
            event_data = event.value if hasattr(event, "value") else event
            if isinstance(event_data, dict):
                event_id = event_data.get("event_id") or event_data.get("event", {}).get("event_id")
                if event_id == "EncryptedSubmitted":
                    attrs = event_data.get("attributes") or event_data.get("event", {}).get(
                        "attributes", {}
                    )
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
        logger.info(
            f"Waiting for MEV shield (block {current_block - submit_block}/{timeout_blocks})..."
        )

        head = await substrate.get_chain_head()
        while await substrate.get_block_number(head) < current_block:
            await asyncio.sleep(3)
            head = await substrate.get_chain_head()

        block_hash = await substrate.get_block_hash(current_block)
        try:
            # Check block events for DecryptedAndExecuted with our shield_id
            events = await substrate.get_events(block_hash=block_hash)
            for event in events:
                event_data = event.value if hasattr(event, "value") else event
                if isinstance(event_data, dict):
                    event_id = event_data.get("event_id")
                    if event_id == "DecryptedAndExecuted":
                        attrs = event_data.get("attributes", {})
                        event_shield_id = attrs.get("id")
                        if event_shield_id == shield_id:
                            success = attrs.get("result", {}).get("Ok") is not None or attrs.get(
                                "success", False
                            )
                            if success or "Ok" in str(attrs.get("result", {})):
                                logger.success(
                                    f"MEV-protected extrinsic executed successfully in block {current_block}"
                                )
                                return True, None
                            else:
                                error = attrs.get("result", {}).get("Err", "Unknown error")
                                return False, f"MEV inner extrinsic failed: {error}"
                    elif event_id == "DecryptionFailed":
                        attrs = event_data.get("attributes", {})
                        if attrs.get("id") == shield_id:
                            return False, "MEV shield decryption failed"

            # Fallback: check extrinsics directly
            block_data = await substrate.get_block(block_hash=block_hash)
            extrinsics = block_data.get("extrinsics", []) if block_data else []

            for extrinsic in extrinsics:
                ext_hash = (
                    f"0x{extrinsic.extrinsic_hash.hex()}"
                    if hasattr(extrinsic, "extrinsic_hash")
                    else None
                )
                if ext_hash == extrinsic_hash:
                    logger.success(f"MEV-protected extrinsic found in block {current_block}")
                    return True, None

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


async def submit_extrinsic_with_mev(
    substrate: AsyncSubstrateInterface,
    call,
    keypair: Keypair,
) -> tuple[bool, Optional[str]]:
    """Submit an extrinsic with MEV protection (when available)."""
    # Get the current nonce explicitly via RPC to avoid caching issues
    result = await substrate.rpc_request("system_accountNextIndex", [keypair.ss58_address])
    current_nonce = result["result"]
    logger.info(f"Current nonce for {keypair.ss58_address}: {current_nonce}")

    # Inner extrinsic uses nonce+1 (will be executed after wrapper)
    inner_extrinsic = await substrate.create_signed_extrinsic(
        call=call, keypair=keypair, nonce=current_nonce + 1
    )
    inner_hash = f"0x{inner_extrinsic.extrinsic_hash.hex()}"

    encrypted_call = await encrypt_extrinsic(substrate, inner_extrinsic)
    if encrypted_call is None:
        logger.info("MEV shield not available, submitting directly")
        # Re-create with correct nonce for direct submission
        inner_extrinsic = await substrate.create_signed_extrinsic(
            call=call, keypair=keypair, nonce=current_nonce
        )
        receipt = await substrate.submit_extrinsic(inner_extrinsic, wait_for_inclusion=True)
        is_success = await receipt.is_success
        if not is_success:
            error_msg = (
                await receipt.error_message
                if hasattr(receipt.error_message, "__await__")
                else receipt.error_message
            )
            return False, f"Extrinsic failed: {error_msg}"
        return True, None

    logger.info("Submitting with MEV protection...")
    # Wrapper extrinsic uses current nonce (submitted first)
    wrapper_extrinsic = await substrate.create_signed_extrinsic(
        call=encrypted_call, keypair=keypair, nonce=current_nonce
    )
    head = await substrate.get_chain_head()
    submit_block = await substrate.get_block_number(head)
    wrapper_receipt = await substrate.submit_extrinsic(wrapper_extrinsic, wait_for_inclusion=True)

    wrapper_is_success = await wrapper_receipt.is_success
    if not wrapper_is_success:
        error_message = (
            await wrapper_receipt.error_message
            if hasattr(wrapper_receipt.error_message, "__await__")
            else wrapper_receipt.error_message
        )
        return False, f"MEV wrapper submission failed: {error_message}"

    shield_id = await extract_mev_shield_id(wrapper_receipt)
    if shield_id is None:
        logger.warning("Could not extract shield ID, assuming success")
        return True, None

    return await wait_for_mev_extrinsic(substrate, inner_hash, shield_id, submit_block)


async def get_free_balance(
    substrate: AsyncSubstrateInterface, address: str, block_hash: str
) -> int:
    """Get free TAO balance on an account (in rao)."""
    result = await substrate.query(
        module="System",
        storage_function="Account",
        params=[address],
        block_hash=block_hash,
    )
    return result["data"]["free"]


async def get_alpha_stake(
    substrate: AsyncSubstrateInterface,
    coldkey_address: str,
    hotkey_address: str,
    netuid: int,
    block_hash: str,
) -> int:
    """Get alpha stake amount (in rao) for a cold/hot key pair on a subnet."""
    try:
        result = await substrate.runtime_call(
            "StakeInfoRuntimeApi",
            "get_stake_info_for_hotkey_coldkey_netuid",
            [hotkey_address, coldkey_address, netuid],
            block_hash=block_hash,
        )
        if result and result.value and "stake" in result.value:
            return int(result.value["stake"])
    except Exception as e:
        logger.warning(f"Could not get alpha stake via runtime API: {e}")

    # Fallback to storage query
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


async def get_subnet_dynamic_info(
    substrate: AsyncSubstrateInterface, netuid: int, block_hash: str
) -> dict | None:
    """
    Get dynamic info for a subnet including tao_in and alpha_in reserves.
    Returns dict with tao_in, alpha_in, etc. or None on error.
    """
    try:
        result = await substrate.runtime_call(
            api="SubnetInfoRuntimeApi",
            method="get_dynamic_info",
            params=[netuid],
            block_hash=block_hash,
        )
        info_data = result if isinstance(result, dict) else (result.value if result else None)
        return info_data
    except Exception as e:
        logger.warning(f"Could not get dynamic info for netuid {netuid}: {e}")
    return None


async def get_subnet_alpha_price(
    substrate: AsyncSubstrateInterface, netuid: int, block_hash: str
) -> float:
    """
    Get alpha price (in TAO) for a subnet.
    Returns the ratio tao_in / alpha_in from the subnet's dynamic info.
    """
    info_data = await get_subnet_dynamic_info(substrate, netuid, block_hash)
    if info_data:
        tao_in = info_data.get("tao_in", 0)
        alpha_in = info_data.get("alpha_in", 0)
        if alpha_in > 0:
            return tao_in / alpha_in
    return 1.0  # Default to 1:1 if we can't get the price


def calculate_slippage_for_alpha_sell(
    alpha_amount: int, tao_in: int, alpha_in: int
) -> tuple[int, float]:
    """
    Calculate the TAO received and slippage for selling alpha_amount.

    Uses constant product formula: (tao_in - tao_out) * (alpha_in + alpha_amount) = tao_in * alpha_in
    Solving for tao_out: tao_out = tao_in * alpha_amount / (alpha_in + alpha_amount)

    Slippage is the difference between spot price and effective price.

    Returns (tao_received, slippage_percent)
    """
    if alpha_in == 0 or tao_in == 0:
        return 0, 1.0  # 100% slippage if no liquidity

    # Spot price (what you'd get with infinitesimal trade)
    spot_price = tao_in / alpha_in
    ideal_tao = int(alpha_amount * spot_price)

    # Actual TAO received using constant product AMM formula
    # tao_out = tao_in * alpha_amount / (alpha_in + alpha_amount)
    tao_received = (tao_in * alpha_amount) // (alpha_in + alpha_amount)

    if ideal_tao == 0:
        return tao_received, 0.0

    slippage = (ideal_tao - tao_received) / ideal_tao
    return tao_received, slippage


def calculate_max_alpha_for_slippage(tao_in: int, alpha_in: int, max_slippage: float) -> int:
    """
    Calculate the maximum alpha that can be sold while staying under max_slippage.

    From the slippage formula:
    slippage = 1 - (alpha_in / (alpha_in + alpha_amount))

    Solving for alpha_amount:
    alpha_amount = alpha_in * slippage / (1 - slippage)

    Returns max alpha amount in rao.
    """
    if max_slippage >= 1.0:
        return alpha_in  # Can sell everything
    if max_slippage <= 0:
        return 0

    # alpha_amount = alpha_in * slippage / (1 - slippage)
    max_alpha = int(alpha_in * max_slippage / (1 - max_slippage))
    return max_alpha


async def add_stake(
    substrate: AsyncSubstrateInterface,
    keypair: Keypair,
    amount_rao: int,
    hotkey_ss58: Optional[str] = None,
    netuid: Optional[int] = None,
) -> tuple[bool, int]:
    """
    Add stake to a hotkey on a subnet.
    Returns (success, amount_actually_staked).
    """
    hotkey_ss58 = hotkey_ss58 or settings.validator_ss58
    netuid = netuid if netuid is not None else settings.netuid

    logger.info(
        f"Adding stake: {amount_rao / ONE_TAO_RAO:.9f} TAO to {hotkey_ss58} on netuid {netuid}"
    )

    try:
        call = await substrate.compose_call(
            call_module="SubtensorModule",
            call_function="add_stake",
            call_params={
                "hotkey": hotkey_ss58,
                "amount_staked": amount_rao,
                "netuid": netuid,
            },
        )

        success, error_msg = await submit_extrinsic_with_mev(substrate, call, keypair)

        if not success:
            logger.error(f"Failed to add stake: {error_msg}")
            return False, 0

        logger.success(f"✅ Added stake: {amount_rao / ONE_TAO_RAO:.9f} TAO")
        return True, amount_rao

    except Exception as e:
        logger.error(f"Error adding stake: {e}\n{traceback.format_exc()}")
        return False, 0


async def move_stake(
    substrate: AsyncSubstrateInterface,
    keypair: Keypair,
    origin_hotkey: str,
    origin_netuid: int,
    amount_rao: int,
    destination_hotkey: Optional[str] = None,
    destination_netuid: Optional[int] = None,
) -> tuple[bool, int]:
    """
    Move stake from one hotkey/netuid to another.
    Used for converting incoming alpha payments to our subnet stake.
    Returns (success, amount_actually_moved).
    """
    destination_hotkey = destination_hotkey or settings.validator_ss58
    destination_netuid = destination_netuid if destination_netuid is not None else settings.netuid

    logger.info(
        f"Moving stake: {amount_rao / ONE_TAO_RAO:.9f} alpha from "
        f"{origin_hotkey}:{origin_netuid} to {destination_hotkey}:{destination_netuid}"
    )

    try:
        call = await substrate.compose_call(
            call_module="SubtensorModule",
            call_function="move_stake",
            call_params={
                "origin_hotkey": origin_hotkey,
                "destination_hotkey": destination_hotkey,
                "origin_netuid": origin_netuid,
                "destination_netuid": destination_netuid,
                "alpha_amount": amount_rao,
            },
        )

        success, error_msg = await submit_extrinsic_with_mev(substrate, call, keypair)

        if not success:
            logger.error(f"Failed to move stake: {error_msg}")
            return False, 0

        logger.success(f"✅ Moved stake: {amount_rao / ONE_TAO_RAO:.9f} alpha")
        return True, amount_rao

    except Exception as e:
        logger.error(f"Error moving stake: {e}\n{traceback.format_exc()}")
        return False, 0


async def burn_alpha(
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

    logger.info(f"🔥 Burning alpha on netuid {netuid}...")

    head = await substrate.get_chain_head()
    block_hash = await substrate.get_block_hash(await substrate.get_block_number(head))

    current_stake = await get_alpha_stake(
        substrate, keypair.ss58_address, hotkey_ss58, netuid, block_hash
    )
    if current_stake == 0:
        logger.info(f"No alpha to burn on netuid {netuid}")
        return True

    burn_amount = amount if amount is not None else current_stake
    burn_amount = min(burn_amount, current_stake)

    logger.info(f"Burning {burn_amount / ONE_TAO_RAO:.9f} alpha from {hotkey_ss58}")

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

        is_success = await receipt.is_success
        if not is_success:
            error_msg = (
                await receipt.error_message
                if hasattr(receipt.error_message, "__await__")
                else receipt.error_message
            )
            logger.error(f"Failed to burn alpha: {error_msg}")
            return False

        logger.success(f"✅ Burned {burn_amount / ONE_TAO_RAO:.9f} alpha")
        return True

    except Exception as e:
        logger.error(f"Error burning alpha: {e}\n{traceback.format_exc()}")
        return False


def calculate_stake_amount(balance_rao: int, alpha_price: float) -> int:
    """
    Calculate how much to stake this iteration to avoid MEV pattern detection.

    - If less than 1 TAO worth: stake the whole thing
    - If more than 1 TAO worth: stake a random amount up to 25 TAO worth
    """
    # Calculate the TAO equivalent value
    tao_equivalent = (balance_rao / ONE_TAO_RAO) * alpha_price

    if tao_equivalent < 1.0:
        # Less than 1 TAO worth: do the whole thing
        return balance_rao
    else:
        # More than 1 TAO worth: random amount between 0.1 and min(25 TAO worth, balance)
        min_amount_rao = int(MIN_STAKE_TAO / alpha_price * ONE_TAO_RAO)
        max_amount_rao = min(
            balance_rao, int(MAX_STAKE_PER_ITERATION_TAO / alpha_price * ONE_TAO_RAO)
        )

        if min_amount_rao >= max_amount_rao:
            return max_amount_rao

        return random.randint(min_amount_rao, max_amount_rao)


async def reconcile_and_process_stake(
    substrate: AsyncSubstrateInterface,
    pending_stake: PendingStake,
    keypair: Keypair,
    block_hash: str,
) -> tuple[bool, int, bool, Optional[str]]:
    """
    Reconcile local balance with chain data and process a stake operation.

    Returns (success, amount_processed, is_complete, error_message).
    is_complete=True means there's no more balance to stake (triggers burn).
    """
    if pending_stake.netuid == 0:
        # TAO payment: check free balance
        chain_balance = await get_free_balance(substrate, pending_stake.wallet_address, block_hash)

        # Account for existential deposit
        result = await substrate.get_constant(
            module_name="Balances",
            constant_name="ExistentialDeposit",
            block_hash=block_hash,
        )
        existential_deposit = int(getattr(result, "value", 0)) + 500000 if result else 500000
        available_balance = max(0, chain_balance - existential_deposit)

        # If chain has no balance, we're done (regardless of what DB says)
        if available_balance <= 0:
            logger.info(
                f"No TAO balance to stake for {pending_stake.wallet_address} (chain exhausted)"
            )
            return True, pending_stake.pending_balance, True, None  # Mark as complete

        # Use the actual chain balance as the source of truth
        actual_pending = available_balance

        # For TAO, alpha_price is 1.0
        stake_amount = calculate_stake_amount(actual_pending, 1.0)

        success, amount_staked = await add_stake(
            substrate,
            keypair,
            stake_amount,
            hotkey_ss58=settings.validator_ss58,
            netuid=settings.netuid,
        )

        if success:
            # Check if we staked everything available
            remaining_on_chain = actual_pending - amount_staked
            is_complete = remaining_on_chain <= 0
            # Return the full pending_balance as processed if chain is exhausted
            amount_to_deduct = pending_stake.pending_balance if is_complete else amount_staked
            return True, amount_to_deduct, is_complete, None
        else:
            return False, 0, False, "Failed to add stake"

    else:
        # Alpha payment: check stake on source hotkey
        chain_stake = await get_alpha_stake(
            substrate,
            pending_stake.wallet_address,
            pending_stake.source_hotkey,
            pending_stake.netuid,
            block_hash,
        )

        # If chain has no stake, we're done (regardless of what DB says)
        if chain_stake <= 0:
            logger.info(
                f"No alpha stake to move for {pending_stake.wallet_address} "
                f"on {pending_stake.source_hotkey}:{pending_stake.netuid} (chain exhausted)"
            )
            return True, pending_stake.pending_balance, True, None  # Mark as complete

        # Use the actual chain stake as the source of truth
        actual_pending = chain_stake

        # Get subnet dynamic info to calculate slippage
        origin_info = await get_subnet_dynamic_info(substrate, pending_stake.netuid, block_hash)

        if origin_info:
            origin_tao_in = origin_info.get("tao_in", 0)
            origin_alpha_in = origin_info.get("alpha_in", 0)

            # Calculate slippage if we moved the full amount
            _, full_slippage = calculate_slippage_for_alpha_sell(
                actual_pending, origin_tao_in, origin_alpha_in
            )

            logger.info(
                f"Slippage analysis: {actual_pending / ONE_TAO_RAO:.9f} alpha, "
                f"pool: {origin_tao_in / ONE_TAO_RAO:.2f} TAO / {origin_alpha_in / ONE_TAO_RAO:.2f} alpha, "
                f"full slippage: {full_slippage * 100:.3f}%"
            )

            # Check for zero liquidity edge case
            if origin_tao_in == 0 or origin_alpha_in == 0:
                logger.warning(
                    f"Zero liquidity for netuid {pending_stake.netuid}: "
                    f"tao_in={origin_tao_in}, alpha_in={origin_alpha_in}. Skipping."
                )
                return False, 0, False, "Zero liquidity in pool, cannot process"

            # If slippage > 0.3%, chunk based on max slippage
            if full_slippage > MAX_SLIPPAGE_PERCENT:
                max_chunk = calculate_max_alpha_for_slippage(
                    origin_tao_in, origin_alpha_in, MAX_SLIPPAGE_PERCENT
                )
                # Safety check: if max_chunk is 0, pool is unusable
                if max_chunk == 0:
                    logger.warning(
                        f"Cannot calculate chunk size for netuid {pending_stake.netuid}. Skipping."
                    )
                    return False, 0, False, "Cannot calculate safe chunk size"

                # Also apply MEV protection chunking
                alpha_price = origin_tao_in / origin_alpha_in
                mev_chunk = calculate_stake_amount(actual_pending, alpha_price)
                # Use the smaller of slippage-limited or MEV-limited chunk
                stake_amount = min(max_chunk, mev_chunk, actual_pending)
                logger.info(
                    f"Chunking for slippage: max {max_chunk / ONE_TAO_RAO:.9f} alpha "
                    f"(MEV limit: {mev_chunk / ONE_TAO_RAO:.9f}), using {stake_amount / ONE_TAO_RAO:.9f}"
                )
            else:
                # Slippage is acceptable, just apply MEV protection chunking
                alpha_price = origin_tao_in / origin_alpha_in
                stake_amount = calculate_stake_amount(actual_pending, alpha_price)
        else:
            # Fallback if we can't get pool info
            alpha_price = await get_subnet_alpha_price(substrate, pending_stake.netuid, block_hash)
            stake_amount = calculate_stake_amount(actual_pending, alpha_price)

        success, amount_moved = await move_stake(
            substrate,
            keypair,
            origin_hotkey=pending_stake.source_hotkey,
            origin_netuid=pending_stake.netuid,
            amount_rao=stake_amount,
            destination_hotkey=settings.validator_ss58,
            destination_netuid=settings.netuid,
        )

        if success:
            # Check if we moved everything available
            remaining_on_chain = actual_pending - amount_moved
            is_complete = remaining_on_chain <= 0
            # Return the full pending_balance as processed if chain is exhausted
            amount_to_deduct = pending_stake.pending_balance if is_complete else amount_moved
            return True, amount_to_deduct, is_complete, None
        else:
            return False, 0, False, "Failed to move stake"


async def process_pending_stakes():
    """
    Main cronjob function. Processes all pending stakes.
    """
    logger.info("Starting pending stakes processing...")

    # Adaptive stale threshold based on queue size and concurrency.
    def _stale_minutes(pending_count: int) -> int:
        if pending_count <= 0:
            return STALE_BASE_MINUTES
        estimated_minutes = int(
            (pending_count / max(1, AUTOSTAKER_CONCURRENCY)) * EXPECTED_SECONDS_PER_WALLET / 60.0
        )
        return min(STALE_MAX_MINUTES, max(STALE_BASE_MINUTES, estimated_minutes * 2))

    # Single cronjob process - just select pending rows, no locking needed
    async with get_session() as session:
        pending_count = await session.execute(
            select(func.count())
            .select_from(PendingStake)
            .where(
                and_(
                    PendingStake.pending_balance > 0,
                    or_(
                        PendingStake.status == "pending",
                        PendingStake.status == "processing",
                    ),
                )
            )
        )
        total_pending = int(pending_count.scalar_one() or 0)
        stale_minutes = _stale_minutes(total_pending)
        stale_cutoff = datetime.now(timezone.utc) - timedelta(minutes=stale_minutes)

        result = await session.execute(
            select(PendingStake)
            .where(
                or_(
                    and_(
                        PendingStake.status == "pending",
                        PendingStake.pending_balance > 0,
                    ),
                    and_(
                        PendingStake.status == "processing",
                        PendingStake.pending_balance > 0,
                        or_(
                            PendingStake.last_attempt_at.is_(None),
                            PendingStake.last_attempt_at < stale_cutoff,
                        ),
                    ),
                )
            )
            .order_by(PendingStake.created_at.asc())
            .limit(250)
        )
        pending_stakes = result.scalars().all()

    if not pending_stakes:
        logger.info("No pending stakes to process")
        return

    logger.info(
        f"Found {len(pending_stakes)} pending stakes to process "
        f"(stale_threshold={stale_minutes}m, concurrency={AUTOSTAKER_CONCURRENCY})"
    )

    # Metrics tracking
    metrics = {
        "total": len(pending_stakes),
        "succeeded": 0,
        "failed": 0,
        "completed": 0,
        "partial": 0,
    }

    async with AsyncSubstrateInterface(url=settings.subtensor) as substrate:
        # Block hash with periodic refresh
        block_hash_lock = asyncio.Lock()
        block_hash_state = {"hash": None, "refreshed_at": 0, "refresh_count": 0}

        async def get_fresh_block_hash() -> str:
            """Get block hash, refreshing every 60 seconds."""
            async with block_hash_lock:
                now = asyncio.get_event_loop().time()
                if block_hash_state["hash"] is None or now - block_hash_state["refreshed_at"] > 60:
                    head = await substrate.get_chain_head()
                    block_hash_state["hash"] = await substrate.get_block_hash(
                        await substrate.get_block_number(head)
                    )
                    block_hash_state["refreshed_at"] = now
                    block_hash_state["refresh_count"] += 1
                return block_hash_state["hash"]

        # Initialize block hash
        await get_fresh_block_hash()

        semaphore = asyncio.Semaphore(AUTOSTAKER_CONCURRENCY)

        async def _process_one(pending_stake: PendingStake) -> None:
            async with semaphore:
                # Small jitter to avoid thundering herd
                await asyncio.sleep(random.uniform(0.2, 1.0))
                logger.info(
                    f"Processing stake for {pending_stake.wallet_address}: "
                    f"netuid={pending_stake.netuid}, balance={pending_stake.pending_balance / ONE_TAO_RAO:.9f}"
                )

                # Load user and keypair
                async with get_session() as session:
                    user = (
                        await session.execute(
                            select(User).where(User.user_id == pending_stake.user_id)
                        )
                    ).scalar_one_or_none()

                    if not user:
                        logger.warning(f"User {pending_stake.user_id} not found, skipping")
                        return

                try:
                    keypair = Keypair.create_from_mnemonic(await decrypt_secret(user.wallet_secret))
                except Exception as e:
                    logger.error(f"Failed to load keypair for {pending_stake.user_id}: {e}")
                    return

                # Update last attempt timestamp
                async with get_session() as session:
                    await session.execute(
                        update(PendingStake)
                        .where(
                            and_(
                                PendingStake.wallet_address == pending_stake.wallet_address,
                                PendingStake.netuid == pending_stake.netuid,
                                PendingStake.source_hotkey == pending_stake.source_hotkey,
                            )
                        )
                        .values(
                            last_attempt_at=func.now(),
                            attempt_count=pending_stake.attempt_count + 1,
                            status="processing",
                        )
                    )
                    await session.commit()

                try:
                    # Get fresh block hash (refreshes every 60s)
                    current_block_hash = await get_fresh_block_hash()

                    (
                        success,
                        amount_processed,
                        is_complete,
                        error_msg,
                    ) = await reconcile_and_process_stake(
                        substrate, pending_stake, keypair, current_block_hash
                    )

                    async with get_session() as session:
                        if is_complete:
                            # Staking complete (chain exhausted), burn alpha if applicable
                            # For alpha payments (any netuid != 0), stake moves to settings.netuid
                            # so we always burn on settings.netuid after move_stake completes
                            if pending_stake.netuid != 0:
                                # Burn alpha on our subnet (settings.netuid)
                                burn_success = await burn_alpha(substrate, keypair)
                                if not burn_success:
                                    await session.execute(
                                        update(PendingStake)
                                        .where(
                                            and_(
                                                PendingStake.wallet_address
                                                == pending_stake.wallet_address,
                                                PendingStake.netuid == pending_stake.netuid,
                                                PendingStake.source_hotkey
                                                == pending_stake.source_hotkey,
                                            )
                                        )
                                        .values(
                                            status="pending",
                                            last_processed_at=func.now(),
                                            error_message="Alpha burn failed, will retry",
                                        )
                                    )
                                    await session.commit()
                                    return

                            # Atomically decrement by original amount (not set to 0) to handle
                            # concurrent upserts that may have added to pending_balance.
                            # Set status to "pending" if balance > 0 after decrement, else "completed"
                            original_balance = pending_stake.pending_balance
                            new_balance_expr = func.greatest(
                                0, PendingStake.pending_balance - original_balance
                            )
                            await session.execute(
                                update(PendingStake)
                                .where(
                                    and_(
                                        PendingStake.wallet_address == pending_stake.wallet_address,
                                        PendingStake.netuid == pending_stake.netuid,
                                        PendingStake.source_hotkey == pending_stake.source_hotkey,
                                    )
                                )
                                .values(
                                    pending_balance=new_balance_expr,
                                    status=case(
                                        (new_balance_expr > 0, "pending"),
                                        else_="completed",
                                    ),
                                    last_processed_at=func.now(),
                                    error_message=None,
                                )
                            )
                            logger.success(
                                f"✅ Completed staking for {pending_stake.wallet_address} "
                                f"netuid={pending_stake.netuid}"
                            )
                            metrics["succeeded"] += 1
                            metrics["completed"] += 1
                        else:
                            # More to stake - just update timestamp, don't decrement pending_balance
                            # (we use chain state as truth, pending_balance just flags "needs processing")
                            await session.execute(
                                update(PendingStake)
                                .where(
                                    and_(
                                        PendingStake.wallet_address == pending_stake.wallet_address,
                                        PendingStake.netuid == pending_stake.netuid,
                                        PendingStake.source_hotkey == pending_stake.source_hotkey,
                                    )
                                )
                                .values(
                                    status="pending",
                                    last_processed_at=func.now(),
                                    error_message=error_msg,
                                )
                            )
                            logger.info(
                                f"Processed {amount_processed / ONE_TAO_RAO:.9f}, more remaining on chain"
                            )
                            metrics["succeeded"] += 1
                            metrics["partial"] += 1

                        await session.commit()

                except Exception as e:
                    error_msg = f"Error processing stake: {e}"
                    logger.error(f"{error_msg}\n{traceback.format_exc()}")
                    metrics["failed"] += 1

                    async with get_session() as session:
                        # Mark as pending with error, will retry next run
                        new_attempt_count = pending_stake.attempt_count + 1
                        new_status = "failed" if new_attempt_count >= 15 else "pending"

                        await session.execute(
                            update(PendingStake)
                            .where(
                                and_(
                                    PendingStake.wallet_address == pending_stake.wallet_address,
                                    PendingStake.netuid == pending_stake.netuid,
                                    PendingStake.source_hotkey == pending_stake.source_hotkey,
                                )
                            )
                            .values(
                                status=new_status,
                                error_message=error_msg[:500],
                            )
                        )
                        await session.commit()

        tasks = [
            asyncio.create_task(_process_one(pending_stake)) for pending_stake in pending_stakes
        ]
        await asyncio.gather(*tasks)

    # Log summary metrics
    logger.info(
        f"Finished processing pending stakes: "
        f"total={metrics['total']}, succeeded={metrics['succeeded']}, failed={metrics['failed']}, "
        f"completed={metrics['completed']}, partial={metrics['partial']}, "
        f"block_refreshes={block_hash_state['refresh_count']}"
    )


async def upsert_pending_stake(
    user_id: str,
    wallet_address: str,
    netuid: int,
    amount_rao: int,
    source_hotkey: str = "",
):
    """
    Insert or update a pending stake record.
    If a record exists, add to the pending_balance.
    """
    async with get_session() as session:
        stmt = insert(PendingStake).values(
            wallet_address=wallet_address,
            netuid=netuid,
            source_hotkey=source_hotkey,
            user_id=user_id,
            pending_balance=amount_rao,
            status="pending",
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["wallet_address", "netuid", "source_hotkey"],
            set_={
                "pending_balance": PendingStake.pending_balance + amount_rao,
                "status": "pending",
                "updated_at": func.now(),
            },
        )
        await session.execute(stmt)
        await session.commit()

    logger.info(
        f"Upserted pending stake: wallet={wallet_address}, netuid={netuid}, "
        f"hotkey={source_hotkey or 'N/A'}, amount={amount_rao / ONE_TAO_RAO:.9f}"
    )


async def main():
    """Entry point for the cronjob."""
    await process_pending_stakes()


if __name__ == "__main__":
    asyncio.run(main())
