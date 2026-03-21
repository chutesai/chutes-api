#!/usr/bin/env python
"""
Fully programmatic agent registration for Chutes.

Usable as both a CLI tool and an importable library.

CLI usage:
    # Full flow (register + pay + poll + setup):
    python scripts/agent_register.py \\
        --hotkey-seed 0xabcdef... \\
        --coldkey-mnemonic "word1 word2 ..." \\
        --api-base https://api.chutes.ai

    # Or with a bittensor wallet on disk:
    python scripts/agent_register.py \\
        --wallet-name my_wallet \\
        --wallet-hotkey my_hotkey

    # Resume an existing registration (poll + setup only, no new registration):
    python scripts/agent_register.py --resume \\
        --hotkey-seed 0xabcdef...

Library usage:
    from scripts.agent_register import register_agent

    result = await register_agent(
        hotkey_seed="0xabcdef...",
        coldkey_mnemonic="word1 word2 ...",
    )
    # result is an AgentRegistrationResult with all account details
"""

import asyncio
import os
import time
from dataclasses import dataclass
from getpass import getpass
from typing import Optional

import click
import requests
from loguru import logger

DEFAULT_API_BASE = "https://api.chutes.ai"
DEFAULT_SUBTENSOR_URL = "wss://entrypoint-finney.opentensor.ai:443"
POLL_INTERVAL_SECONDS = 12


@dataclass
class AgentRegistrationResult:
    """Result of a successful agent registration."""

    user_id: str
    username: str
    api_key: str
    hotkey_ss58: str
    coldkey_ss58: str
    payment_address: str
    config_ini: str
    setup_instructions: str


def _get_keypair_from_seed(seed: str):
    """Create a Keypair from a hex seed."""
    from bittensor_wallet.keypair import Keypair

    seed = seed.removeprefix("0x")
    return Keypair.create_from_seed(f"0x{seed}")


def _get_keypair_from_mnemonic(mnemonic: str):
    """Create a Keypair from a mnemonic phrase."""
    from bittensor_wallet.keypair import Keypair

    return Keypair.create_from_mnemonic(mnemonic)


def _get_keypairs_from_wallet(wallet_name: str, wallet_hotkey: str, wallet_path: str):
    """Load keypairs from a bittensor wallet on disk."""
    from bittensor_wallet import Wallet

    wallet = Wallet(name=wallet_name, hotkey=wallet_hotkey, path=wallet_path)
    return wallet.hotkey, wallet.coldkey


def _sign_registration(hotkey_keypair, coldkey_ss58: str) -> tuple[str, str, str]:
    """Sign the registration message. Returns (hotkey_ss58, coldkey_ss58, signature_hex)."""
    hotkey_ss58 = hotkey_keypair.ss58_address
    message = f"chutes_signup:{hotkey_ss58}:{coldkey_ss58}"
    signature = hotkey_keypair.sign(message.encode()).hex()
    return hotkey_ss58, coldkey_ss58, signature


def _api_register(
    api_base: str,
    hotkey_ss58: str,
    coldkey_ss58: str,
    signature: str,
    username: Optional[str] = None,
) -> dict:
    """POST /users/agent_registration — returns registration response dict."""
    payload = {
        "hotkey": hotkey_ss58,
        "coldkey": coldkey_ss58,
        "signature": signature,
    }
    if username:
        payload["username"] = username

    resp = requests.post(f"{api_base}/users/agent_registration", json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Registration failed ({resp.status_code}): {resp.text}")
    return resp.json()


def _api_poll_status(api_base: str, hotkey_ss58: str) -> dict:
    """GET /users/agent_registration/{hotkey} — returns status response dict."""
    resp = requests.get(f"{api_base}/users/agent_registration/{hotkey_ss58}", timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Status check failed ({resp.status_code}): {resp.text}")
    return resp.json()


def _api_setup(api_base: str, user_id: str, hotkey_keypair) -> dict:
    """POST /users/{user_id}/agent_setup — returns setup response dict.
    Requires hotkey signature for authentication."""
    signing_message = f"chutes_setup:{user_id}"
    signature = hotkey_keypair.sign(signing_message.encode()).hex()
    payload = {
        "hotkey": hotkey_keypair.ss58_address,
        "signature": signature,
    }
    resp = requests.post(f"{api_base}/users/{user_id}/agent_setup", json=payload, timeout=30)
    if resp.status_code == 409:
        raise RuntimeError("Agent setup has already been completed for this user (one-time only).")
    if resp.status_code != 200:
        raise RuntimeError(f"Setup failed ({resp.status_code}): {resp.text}")
    return resp.json()


async def _send_tao(
    coldkey_keypair,
    destination: str,
    amount_rao: int,
    subtensor_url: str,
):
    """Transfer TAO to the payment address."""
    from async_substrate_interface import AsyncSubstrateInterface

    logger.info(
        f"Sending {amount_rao / 1e9:.4f} TAO to {destination} "
        f"from {coldkey_keypair.ss58_address}..."
    )
    async with AsyncSubstrateInterface(url=subtensor_url) as substrate:
        call = await substrate.compose_call(
            call_module="Balances",
            call_function="transfer_keep_alive",
            call_params={
                "dest": destination,
                "value": amount_rao,
            },
        )
        extrinsic = await substrate.create_signed_extrinsic(call=call, keypair=coldkey_keypair)
        receipt = await substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        is_success = await receipt.is_success
        if not is_success:
            error_msg = (
                await receipt.error_message
                if hasattr(receipt.error_message, "__await__")
                else receipt.error_message
            )
            raise RuntimeError(f"TAO transfer failed: {error_msg}")

        block_hash = receipt.block_hash
        logger.success(f"TAO transfer confirmed in block {block_hash}")
        return block_hash


def _poll_and_setup(
    api_base: str,
    hotkey_ss58: str,
    hotkey_keypair,
    hotkey_seed: Optional[str],
    poll_interval: int,
    write_config: bool,
    config_path: str,
) -> AgentRegistrationResult:
    """
    Poll registration status until completed, then call setup.
    Shared by both register_agent() and resume_agent().
    """
    logger.info("Polling registration status...")
    user_id = None
    coldkey_ss58 = None
    while True:
        status_resp = _api_poll_status(api_base, hotkey_ss58)
        reg_status = status_resp["status"]
        user_id = status_resp["user_id"]
        coldkey_ss58 = status_resp["coldkey"]

        if reg_status == "completed":
            logger.success("Payment confirmed, account created!")
            break
        elif reg_status == "expired":
            raise RuntimeError(
                "Registration expired. Funds sent to expired registrations are not recoverable. "
                "Each hotkey can only be used for one registration attempt — you must use a new hotkey."
            )
        else:
            received = status_resp.get("received_amount", 0)
            logger.info(f"Waiting for payment confirmation... received ${received:.2f}")
            time.sleep(poll_interval)

    # Setup.
    logger.info("Calling agent setup endpoint...")
    setup = _api_setup(api_base, user_id, hotkey_keypair)

    # Replace hotkey seed placeholder in config.
    hotkey_seed_hex = hotkey_seed or hotkey_keypair.seed_hex
    hotkey_seed_hex = hotkey_seed_hex.removeprefix("0x")
    config_ini = setup["config_ini"].replace("REPLACE_WITH_YOUR_HOTKEY_SEED", hotkey_seed_hex)

    result = AgentRegistrationResult(
        user_id=user_id,
        username=setup["username"],
        api_key=setup["api_key"],
        hotkey_ss58=hotkey_ss58,
        coldkey_ss58=coldkey_ss58,
        payment_address=setup["payment_address"],
        config_ini=config_ini,
        setup_instructions=setup["setup_instructions"],
    )

    # Optionally write config to disk.
    if write_config:
        expanded = os.path.expanduser(config_path)
        os.makedirs(os.path.dirname(expanded), exist_ok=True)
        with open(expanded, "w") as f:
            f.write(config_ini)
        logger.success(f"Config written to {expanded}")

    logger.success("Agent registration complete!")
    logger.info(f"  User ID:         {result.user_id}")
    logger.info(f"  Username:        {result.username}")
    logger.info(f"  API Key:         {result.api_key}")
    logger.info(f"  Payment Address: {result.payment_address}")

    return result


async def register_agent(
    hotkey_seed: Optional[str] = None,
    coldkey_mnemonic: Optional[str] = None,
    hotkey_keypair=None,
    coldkey_keypair=None,
    username: Optional[str] = None,
    api_base: str = DEFAULT_API_BASE,
    subtensor_url: str = DEFAULT_SUBTENSOR_URL,
    amount_extra_percent: float = 5.0,
    write_config: bool = False,
    config_path: str = "~/.chutes/config.ini",
    poll_interval: int = POLL_INTERVAL_SECONDS,
) -> AgentRegistrationResult:
    """
    Full agent registration flow: register -> pay -> poll -> setup.

    Provide keys via either:
      - hotkey_seed + coldkey_mnemonic (strings)
      - hotkey_keypair + coldkey_keypair (Keypair objects)

    Args:
        hotkey_seed: Hex seed for the hotkey (e.g. "0xabcdef...")
        coldkey_mnemonic: Mnemonic phrase for the coldkey
        hotkey_keypair: Pre-constructed Keypair for the hotkey
        coldkey_keypair: Pre-constructed Keypair for the coldkey
        username: Optional username (3-15 alphanum/underscore/dash). Auto-generated if omitted.
        api_base: Chutes API base URL
        subtensor_url: Subtensor RPC endpoint
        amount_extra_percent: Extra % to send above required amount (default 5%)
        write_config: If True, write config.ini to disk
        config_path: Path for config.ini (default ~/.chutes/config.ini)
        poll_interval: Seconds between status polls (default 12, ~1 block)

    Returns:
        AgentRegistrationResult with user_id, api_key, config, etc.
    """
    # Resolve keypairs.
    if hotkey_keypair is None:
        if hotkey_seed is None:
            raise ValueError("Provide either hotkey_seed or hotkey_keypair")
        hotkey_keypair = _get_keypair_from_seed(hotkey_seed)

    if coldkey_keypair is None:
        if coldkey_mnemonic is None:
            raise ValueError("Provide coldkey_mnemonic or coldkey_keypair")
        coldkey_keypair = _get_keypair_from_mnemonic(coldkey_mnemonic)

    coldkey_ss58 = coldkey_keypair.ss58_address
    hotkey_ss58 = hotkey_keypair.ss58_address
    logger.info(f"Hotkey: {hotkey_ss58}")
    logger.info(f"Coldkey: {coldkey_ss58}")

    # Step 1: Sign and register.
    logger.info("Signing registration message...")
    _, _, signature = _sign_registration(hotkey_keypair, coldkey_ss58)

    logger.info("Submitting registration...")
    reg = _api_register(api_base, hotkey_ss58, coldkey_ss58, signature, username)
    payment_address = reg["payment_address"]
    required_tao = reg["required_amount"]
    logger.success(f"Registration created: user_id={reg['user_id']}")
    logger.info(f"Payment address: {payment_address}")
    logger.info(f"Required TAO: {required_tao}")

    # Step 2: Send TAO.
    send_amount_tao = required_tao * (1 + amount_extra_percent / 100)
    send_amount_rao = int(send_amount_tao * 1e9)
    logger.info(f"Sending {send_amount_tao:.4f} TAO ({amount_extra_percent}% buffer)...")
    await _send_tao(coldkey_keypair, payment_address, send_amount_rao, subtensor_url)

    # Step 3+4: Poll and setup.
    return _poll_and_setup(
        api_base,
        hotkey_ss58,
        hotkey_keypair,
        hotkey_seed,
        poll_interval,
        write_config,
        config_path,
    )


async def resume_agent(
    hotkey_seed: Optional[str] = None,
    hotkey_keypair=None,
    api_base: str = DEFAULT_API_BASE,
    write_config: bool = False,
    config_path: str = "~/.chutes/config.ini",
    poll_interval: int = POLL_INTERVAL_SECONDS,
) -> AgentRegistrationResult:
    """
    Resume an existing agent registration: poll for completion then setup.

    Use this when you've already created a registration and sent payment manually,
    or if the script was interrupted after payment.

    Args:
        hotkey_seed: Hex seed for the hotkey
        hotkey_keypair: Pre-constructed Keypair for the hotkey
        api_base: Chutes API base URL
        write_config: If True, write config.ini to disk
        config_path: Path for config.ini (default ~/.chutes/config.ini)
        poll_interval: Seconds between status polls (default 12, ~1 block)

    Returns:
        AgentRegistrationResult with user_id, api_key, config, etc.
    """
    if hotkey_keypair is None:
        if hotkey_seed is None:
            raise ValueError("Provide either hotkey_seed or hotkey_keypair")
        hotkey_keypair = _get_keypair_from_seed(hotkey_seed)

    hotkey_ss58 = hotkey_keypair.ss58_address
    logger.info(f"Resuming registration for hotkey: {hotkey_ss58}")

    return _poll_and_setup(
        api_base,
        hotkey_ss58,
        hotkey_keypair,
        hotkey_seed,
        poll_interval,
        write_config,
        config_path,
    )


# --- CLI ---


@click.command()
@click.option(
    "--hotkey-seed",
    default=None,
    help="Hotkey seed hex (e.g. 0xabcdef...). Prompted if not provided and --wallet-name is not set.",
)
@click.option(
    "--coldkey-mnemonic",
    default=None,
    help="Coldkey mnemonic phrase. Prompted securely if not provided and --wallet-name is not set.",
)
@click.option(
    "--wallet-name",
    default=None,
    help="Bittensor wallet name (alternative to --hotkey-seed/--coldkey-mnemonic).",
)
@click.option(
    "--wallet-hotkey",
    default="default",
    show_default=True,
    help="Hotkey name within the wallet.",
)
@click.option(
    "--wallet-path",
    default="~/.bittensor/wallets",
    show_default=True,
    help="Path to bittensor wallets directory.",
)
@click.option(
    "--username",
    default=None,
    help="Desired username (3-15 chars, alphanumeric/underscore/dash). Auto-generated if omitted.",
)
@click.option(
    "--api-base",
    default=DEFAULT_API_BASE,
    show_default=True,
    help="Chutes API base URL.",
)
@click.option(
    "--subtensor-url",
    default=DEFAULT_SUBTENSOR_URL,
    show_default=True,
    help="Subtensor RPC endpoint.",
)
@click.option(
    "--extra-percent",
    default=5.0,
    show_default=True,
    help="Extra %% to send above required TAO amount.",
)
@click.option(
    "--write-config",
    is_flag=True,
    default=False,
    help="Write config.ini to ~/.chutes/config.ini after registration.",
)
@click.option(
    "--config-path",
    default="~/.chutes/config.ini",
    show_default=True,
    help="Path for config.ini output.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume an existing registration: poll for payment completion and run setup. "
    "Does not create a new registration or send TAO.",
)
def cli(
    hotkey_seed: Optional[str],
    coldkey_mnemonic: Optional[str],
    wallet_name: Optional[str],
    wallet_hotkey: str,
    wallet_path: str,
    username: Optional[str],
    api_base: str,
    subtensor_url: str,
    extra_percent: float,
    write_config: bool,
    config_path: str,
    resume: bool,
):
    """
    Register an AI agent on Chutes programmatically.

    Provide keys via one of:

    \b
    1. --hotkey-seed + --coldkey-mnemonic (or prompted)
    2. --wallet-name (loads from bittensor wallet on disk)

    Use --resume to poll and setup an existing registration without creating a new one.
    """
    hotkey_kp = None
    coldkey_kp = None

    if wallet_name:
        # Load from wallet on disk.
        logger.info(f"Loading wallet '{wallet_name}' hotkey '{wallet_hotkey}'...")
        password = os.environ.get("BT_WALLET_PASSWORD")
        if password is None:
            password = getpass("Wallet password (or press Enter if unencrypted): ").strip()
            if password:
                os.environ["BT_WALLET_PASSWORD"] = password

        hotkey_kp, coldkey_kp = _get_keypairs_from_wallet(
            wallet_name, wallet_hotkey, os.path.expanduser(wallet_path)
        )
        # Extract seed for config.ini.
        hotkey_seed = f"0x{hotkey_kp.seed_hex}"
    else:
        # Use seed/mnemonic directly.
        if not hotkey_seed:
            hotkey_seed = getpass("Hotkey seed hex (will not be echoed): ").strip()
            if not hotkey_seed:
                logger.error("Hotkey seed is required.")
                raise SystemExit(1)

        hotkey_kp = _get_keypair_from_seed(hotkey_seed)

        if not resume:
            if not coldkey_mnemonic:
                coldkey_mnemonic = getpass("Coldkey mnemonic (will not be echoed): ").strip()
                if not coldkey_mnemonic:
                    logger.error("Coldkey mnemonic is required for TAO transfer.")
                    raise SystemExit(1)
            coldkey_kp = _get_keypair_from_mnemonic(coldkey_mnemonic)

    # Show summary and confirm.
    click.echo()
    click.echo("=" * 60)
    if resume:
        click.echo("CHUTES AGENT REGISTRATION (RESUME)")
    else:
        click.echo("CHUTES AGENT REGISTRATION")
    click.echo("=" * 60)
    click.echo(f"  Hotkey:       {hotkey_kp.ss58_address}")
    if coldkey_kp:
        click.echo(f"  Coldkey:      {coldkey_kp.ss58_address}")
    click.echo(f"  API:          {api_base}")
    if not resume:
        click.echo(f"  Subtensor:    {subtensor_url}")
        if username:
            click.echo(f"  Username:     {username}")
        click.echo(f"  Extra buffer: {extra_percent}%")
    click.echo(f"  Write config: {write_config}")
    if resume:
        click.echo("  Mode:         Resume existing registration (poll + setup only)")
    click.echo("=" * 60)
    click.echo()

    if not click.confirm("Proceed?", default=True):
        logger.info("Aborted.")
        raise SystemExit(0)

    try:
        if resume:
            result = asyncio.run(
                resume_agent(
                    hotkey_seed=hotkey_seed,
                    hotkey_keypair=hotkey_kp,
                    api_base=api_base,
                    write_config=write_config,
                    config_path=config_path,
                )
            )
        else:
            result = asyncio.run(
                register_agent(
                    hotkey_seed=hotkey_seed,
                    coldkey_mnemonic=coldkey_mnemonic,
                    hotkey_keypair=hotkey_kp,
                    coldkey_keypair=coldkey_kp,
                    username=username,
                    api_base=api_base,
                    subtensor_url=subtensor_url,
                    amount_extra_percent=extra_percent,
                    write_config=write_config,
                    config_path=config_path,
                )
            )
    except RuntimeError as e:
        logger.error(f"Registration failed: {e}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        raise SystemExit(1)

    click.echo()
    click.echo("=" * 60)
    click.echo("REGISTRATION COMPLETE")
    click.echo("=" * 60)
    click.echo(f"  User ID:         {result.user_id}")
    click.echo(f"  Username:        {result.username}")
    click.echo(f"  API Key:         {result.api_key}")
    click.echo(f"  Hotkey:          {result.hotkey_ss58}")
    click.echo(f"  Payment Address: {result.payment_address}")
    click.echo()
    click.echo("Config file contents:")
    click.echo("-" * 40)
    click.echo(result.config_ini)
    click.echo("-" * 40)
    click.echo()
    click.echo(result.setup_instructions)
    click.echo()

    if not write_config:
        click.echo("Tip: Re-run with --write-config to automatically save to ~/.chutes/config.ini")


if __name__ == "__main__":
    cli()
