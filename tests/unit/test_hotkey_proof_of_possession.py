"""
Miner-hotkey proof of possession on the TDX boot-attestation / provision flow.

The vulnerability this closes: the boot POST's ``miner_hotkey`` was an unproven string, so a miner
running a genuine TDX host on the published image could POST a VICTIM's ``(miner_hotkey, vm_name)``
with ``first_boot: true`` and trigger a root LUKS passphrase rotation for the victim's record --
permanently bricking that VM. The fix is an sr25519 signature over
``{hotkey}:{nonce}:{sha256(body)}`` (the same message shape and headers the userspace hotkey auth
already uses), verified at the edge against ``X-Chutes-Hotkey``.

The split these tests enforce:

  * AUTHENTICATION happens once, at the edge (``verify_hotkey_auth``). A signature either verifies
    or 401s there; no service re-runs crypto, so none can forget to.
  * WHERE a proof is REQUIRED depends on who can reach the route. The 1.4.0-exclusive /provision
    endpoints use ``require_hotkey_auth`` and 401 an unsigned caller at the door. /boot/attestation
    is shared, so it uses ``extract_hotkey_auth`` and decides once the quote names the image: an
    attested version that ships the signer must have used it; an older one could not have, and
    still boots on its bare claim. That last allowance disappears on its own when
    tee_minimum_boot_version reaches 1.4.0 and unsigned images stop attesting at all.
  * ABSENCE is spelled None. ``verify_hotkey_auth`` raises unless the signature verifies, so any
    HotkeyAuth a service holds came back from it and its ``miner_hotkey`` is proven. There is no
    "was this verified" flag to set wrongly.
"""

import secrets
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from bittensor_wallet.keypair import Keypair

from api.server.exceptions import UnauthorizedError
from api.server.schemas import BootAttestationArgs, HotkeyAuth, ProvisionRequest
from api.server.service import process_boot_attestation, process_provision_request
from api.server.util import verify_hotkey_auth
from api.util import get_signing_message

TEST_IP = "127.0.0.1"
TEST_CERT_HASH = "certhash"
BOOT_NONCE = "b" * 64  # a server-issued boot nonce (secrets.token_hex(32) shape)
BODY_HASH = "a" * 64  # request.state.body_sha256 over the exact body bytes
SIGNING_VERSION = "1.4.0"  # image whose measured initramfs ships the sr25519 signer
LEGACY_VERSION = "1.3.0"  # predates the signer -> unsigned requests still accepted


def _kp():
    return Keypair.create_from_seed("0x" + secrets.token_hex(32))


def _auth(
    kp,
    *,
    nonce=BOOT_NONCE,
    body_sha256=BODY_HASH,
    sign_with=None,
    claim_as=None,
    tamper=False,
    purpose=None,
):
    """The raw header material a caller would send.

    ``sign_with`` signs as somebody else (forgery); ``claim_as`` puts somebody else's ss58 in
    X-Chutes-Hotkey while ``kp`` does the signing.
    """
    signer = sign_with or kp
    hotkey = claim_as or kp.ss58_address
    message = get_signing_message(
        hotkey=hotkey, nonce=nonce, payload_str=None, payload_hash=body_sha256, purpose=purpose
    )
    signature = signer.sign(message).hex()
    if tamper:
        signature = ("00" if signature[:2] != "00" else "11") + signature[2:]
    return HotkeyAuth(
        miner_hotkey=hotkey,
        signature=signature,
        nonce=nonce,
        body_sha256=body_sha256,
        purpose=purpose,
    )


def _verified(kp, **kw):
    """That material run through the edge -- i.e. the state a service actually receives."""
    return verify_hotkey_auth(_auth(kp, **kw))


# ---------------------------------------------------------------------------
# verify_hotkey_auth -- authentication, at the edge
# ---------------------------------------------------------------------------


def test_valid_signature_yields_a_proven_identity():
    kp = _kp()
    auth = _verified(kp)
    assert auth.miner_hotkey == kp.ss58_address


def test_signature_by_another_key_is_rejected():
    """The core attack: claim a victim's hotkey, sign with your own key."""
    victim, attacker = _kp(), _kp()
    with pytest.raises(UnauthorizedError):
        verify_hotkey_auth(_auth(attacker, claim_as=victim.ss58_address))


def test_tampered_signature_is_rejected():
    with pytest.raises(UnauthorizedError):
        verify_hotkey_auth(_auth(_kp(), tamper=True))


def test_non_hex_signature_is_rejected():
    auth = _auth(_kp())
    auth.signature = "not-hex"
    with pytest.raises(UnauthorizedError):
        verify_hotkey_auth(auth)


def test_malformed_ss58_is_rejected_not_raised_as_500():
    auth = _auth(_kp())
    auth.miner_hotkey = "definitely-not-an-ss58"
    with pytest.raises(UnauthorizedError):
        verify_hotkey_auth(auth)


def test_signature_over_a_different_body_is_rejected():
    """The body hash is part of the signed message, so a header set cannot be lifted onto a
    different request body -- e.g. one flipping first_boot."""
    auth = _auth(_kp())
    auth.body_sha256 = "f" * 64  # what the server actually hashed
    with pytest.raises(UnauthorizedError):
        verify_hotkey_auth(auth)


def test_signature_over_a_different_nonce_is_rejected():
    """Replay: a header set captured from an earlier call was signed over that call's nonce."""
    auth = _auth(_kp())
    auth.nonce = "c" * 64  # the nonce this call actually used
    with pytest.raises(UnauthorizedError):
        verify_hotkey_auth(auth)


def test_missing_material_is_rejected():
    """A nonce is part of the signed message, so incomplete material can't verify."""
    with pytest.raises(UnauthorizedError):
        verify_hotkey_auth(HotkeyAuth())

    no_nonce = _auth(_kp())
    no_nonce.nonce = None
    with pytest.raises(UnauthorizedError):
        verify_hotkey_auth(no_nonce)

    no_sig = _auth(_kp())
    no_sig.signature = None
    with pytest.raises(UnauthorizedError):
        verify_hotkey_auth(no_sig)


def test_purpose_signatures_verify_too():
    """The userspace/proxy convention signs a purpose instead of a body hash; same primitive."""
    kp = _kp()
    auth = _verified(kp, nonce=str(int(time.time())), body_sha256=None, purpose="tee")
    assert auth.miner_hotkey == kp.ss58_address


# ---------------------------------------------------------------------------
# process_boot_attestation -- authorization, end to end
# ---------------------------------------------------------------------------


@pytest.fixture
def boot_env():
    """Patch out everything downstream of the gate so the tests observe one thing: whether boot
    secrets were issued."""
    db = AsyncMock()
    db.refresh.side_effect = lambda o: setattr(o, "attestation_id", "boot-1")
    root = AsyncMock(return_value=("root-key", "root-next", "confirm-nonce"))
    luks_nonce = AsyncMock(return_value="luks-nonce")

    def _run(version=SIGNING_VERSION):
        service_settings = Mock()
        service_settings.tee_minimum_boot_version = "0.0.0"
        util_settings = Mock()
        return (
            patch("api.server.service.settings", service_settings),
            patch("api.server.util.settings", util_settings),
            patch("api.server.service.BootTdxQuote.from_base64", return_value=Mock()),
            patch(
                "api.server.service.verify_quote",
                new=AsyncMock(return_value=(Mock(), MagicMock(version=version, rc=False))),
            ),
            patch("api.server.service.generate_luks_quote_nonce", luks_nonce),
            patch("api.server.service._handle_boot_version_update", new=AsyncMock()),
            patch("api.server.service.get_root_passphrase_for_boot", root),
            patch("api.server.service._generate_and_store_vm_auth_key", new=AsyncMock()),
            patch("api.server.service.func", MagicMock()),
        )

    return db, root, luks_nonce, _run


async def _boot(db, patches, args, auth):
    from contextlib import ExitStack

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return await process_boot_attestation(db, TEST_IP, args, BOOT_NONCE, TEST_CERT_HASH, auth)


def _args(hotkey, *, first_boot=True, vm_name="victim-vm"):
    return BootAttestationArgs(
        quote="cXVvdGU=", miner_hotkey=hotkey, vm_name=vm_name, first_boot=first_boot
    )


@pytest.mark.asyncio
async def test_boot_with_a_proven_hotkey_provisions_as_before(boot_env):
    """Acceptance: a valid signature for its own registered hotkey provisions/rotates as before."""
    db, root, _, run = boot_env
    kp = _kp()
    result = await _boot(db, run(), _args(kp.ss58_address), _verified(kp))

    assert result.root_key == "root-key"
    assert result.root_next == "root-next"
    # The rotation is keyed by the PROVEN hotkey.
    assert root.await_args.args[1] == kp.ss58_address


@pytest.mark.asyncio
async def test_boot_claiming_another_miners_hotkey_cannot_rotate_its_passphrase(boot_env):
    """Acceptance: the cross-miner brick. An attacker asserts the victim's (hotkey, vm_name) with
    first_boot=True but signs with their own key -- the edge rejects it outright."""
    db, root, luks_nonce, run = boot_env
    victim, attacker = _kp(), _kp()

    with pytest.raises(UnauthorizedError):
        # Never reaches the service: authentication fails at the edge.
        verify_hotkey_auth(_auth(attacker, claim_as=victim.ss58_address))

    root.assert_not_awaited()
    luks_nonce.assert_not_awaited()


@pytest.mark.asyncio
async def test_pre_signer_image_without_a_signature_still_boots(boot_env):
    """Acceptance: in-field VMs whose measured initramfs predates the signer keep booting. They
    cannot be retrofitted, so refusing them would brick the fleet on its next reboot."""
    db, root, _, run = boot_env
    result = await _boot(db, run(version=LEGACY_VERSION), _args("5FLegacyMiner"), None)

    assert result.root_key == "root-key"
    # Nothing was proven, so the body's bare claim is still what the record is keyed by.
    assert root.await_args.args[1] == "5FLegacyMiner"


@pytest.mark.asyncio
async def test_signing_image_that_did_not_sign_is_refused(boot_env):
    """The other half of the same rule: once the ATTESTED version ships the signer, silence is a
    fault rather than a limitation, and no boot secret is resolved."""
    db, root, luks_nonce, run = boot_env

    with pytest.raises(UnauthorizedError):
        await _boot(db, run(version=SIGNING_VERSION), _args("5FVictim"), None)

    root.assert_not_awaited()
    luks_nonce.assert_not_awaited()


@pytest.mark.asyncio
async def test_boot_keys_the_record_by_the_proven_hotkey_not_the_body(boot_env):
    """The body's miner_hotkey is a claim; the signature is the identity. Everything the boot
    issues -- the record, the rotation, the quote nonce -- is keyed by the proven one."""
    db, root, luks_nonce, run = boot_env
    kp = _kp()

    await _boot(db, run(), _args("5FSomeoneElse"), _verified(kp))

    assert root.await_args.args[1] == kp.ss58_address
    assert luks_nonce.await_args.args[0] == kp.ss58_address


# ---------------------------------------------------------------------------
# the confirm routes -- the requirement is a property of the ROUTE, not the VM
#
# /provision/confirm sits behind require_cvm_proxy + require_hotkey_auth and is reachable only by
# 1.4.0+ VMs, every one of which signs; an unsigned caller is turned away at the door. /luks/confirm
# serves in-field 1.3.x VMs that have no signer at all, so it carries no hotkey dependency --
# refusing those would strand a rotated-away passphrase unpromoted and brick the volume next boot.
#
# Both share one auth-free service: by the time process_luks_confirm runs, the route has already
# decided. It identifies the VM by the same X-Chutes-Hotkey header require_hotkey_auth proves, so
# there is nothing left for it to re-check.
# ---------------------------------------------------------------------------


@pytest.fixture
def confirm_env():
    from api.server.schemas import LuksConfirmRequest, LuksVolumeConfirmStatus

    db = AsyncMock()
    vm_config = MagicMock()
    vm_config.volume_passphrases = {"pending_storage": "enc"}
    body = LuksConfirmRequest(volumes={"storage": LuksVolumeConfirmStatus(rotated=True)})
    return db, vm_config, body


async def _confirm(db, vm_config, body, hotkey):
    from api.server.service import process_luks_confirm

    with (
        patch("api.server.service._get_vm_cache_config", new=AsyncMock(return_value=vm_config)),
        patch("api.server.service.func", MagicMock()),
    ):
        return await process_luks_confirm(db, hotkey, "victim-vm", body)


@pytest.mark.asyncio
async def test_confirm_promotes_the_pending_passphrase(confirm_env):
    db, vm_config, body = confirm_env
    result = await _confirm(db, vm_config, body, "5FMiner")
    assert result.volumes["storage"]["result"] == "promoted"


def test_every_provision_route_demands_a_proven_hotkey():
    """The requirement is carried by the route's dependency list, and the service below has no
    auth argument to reveal its absence -- so assert on the wiring itself. Both confirm calls the
    1.4.0 initramfs makes (root, in init-premount; storage, in init-bottom) are signed, so
    dropping this dependency would silently stop checking a signature the VM is still sending."""
    from api.server.router import router
    from api.server.util import require_hotkey_auth

    guarded = {"/{vm_name}/provision", "/{vm_name}/provision/confirm"}
    for route in router.routes:
        if getattr(route, "path", None) not in guarded:
            continue
        assert any(
            getattr(d.call, "__qualname__", "").startswith(require_hotkey_auth.__name__)
            for d in route.dependant.dependencies
        ), f"{route.path} lost its hotkey-auth dependency"
        guarded.discard(route.path)
    assert not guarded, f"routes not found in the router: {guarded}"


@pytest.mark.asyncio
async def test_an_unsigned_provision_flow_call_is_turned_away():
    """1.4.0 signs every provision-flow call, so the route can demand it unconditionally -- the
    service below never sees an unsigned one."""
    from api.server.util import require_hotkey_auth

    request = Mock()
    request.state.body_sha256 = BODY_HASH
    request.headers = {}
    with pytest.raises(UnauthorizedError):
        await require_hotkey_auth()(request)


@pytest.mark.asyncio
async def test_legacy_luks_confirm_needs_no_signature(confirm_env):
    """A 1.3.x VM cannot sign; its confirm must still promote, or the passphrase it just rotated
    away is stranded and the volume bricks on the next boot."""
    db, vm_config, body = confirm_env
    result = await _confirm(db, vm_config, body, "5FLegacyMiner")
    assert result.volumes["storage"]["result"] == "promoted"


# ---------------------------------------------------------------------------
# process_provision_request -- the other passphrase-issuing entry point
# ---------------------------------------------------------------------------


@pytest.fixture
def provision_env():
    db = AsyncMock()
    issue = AsyncMock(return_value=MagicMock())
    record_ca = AsyncMock()

    def _run(version=SIGNING_VERSION):
        service_settings = Mock()
        util_settings = Mock()
        return (
            patch("api.server.service.settings", service_settings),
            patch("api.server.util.settings", util_settings),
            patch("api.server.service.RuntimeTdxQuote.from_base64", return_value=Mock()),
            patch(
                "api.server.service.verify_quote",
                new=AsyncMock(return_value=(Mock(), MagicMock(version=version, rc=False))),
            ),
            patch("api.server.service.get_public_key_hash", return_value="hash"),
            patch("api.server.service.record_vm_ca_identity", record_ca),
            patch("api.server.service._issue_storage_secrets", issue),
        )

    return db, issue, record_ca, _run


async def _provision(db, patches, hotkey, auth):
    from contextlib import ExitStack

    body = ProvisionRequest(quote="cXVvdGU=", volumes=["storage", "tdx-cache"])
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return await process_provision_request(
            db, hotkey, "victim-vm", body, BOOT_NONCE, Mock(), auth=auth
        )


@pytest.mark.asyncio
async def test_provision_with_a_proven_hotkey_issues_secrets_for_it(provision_env):
    db, issue, _, run = provision_env
    kp = _kp()
    await _provision(db, run(), kp.ss58_address, _verified(kp))
    assert issue.await_args.args[1] == kp.ss58_address


@pytest.mark.asyncio
async def test_provision_route_rejects_an_unsigned_request():
    """The requirement lives on the route, not the service: /provision is 1.4.0-only (behind
    require_cvm_proxy) and every VM that can reach it signs, so an unsigned request never reaches a
    handler. No version check here, unlike boot."""
    from api.server.util import require_hotkey_auth

    request = Mock()
    request.state.body_sha256 = BODY_HASH
    request.headers = {}
    with pytest.raises(UnauthorizedError):
        await require_hotkey_auth()(request)


@pytest.mark.asyncio
async def test_provision_forwards_the_proven_hotkey_to_the_rc_gate(provision_env):
    """What the caller proved is what the rc gate is told -- the one thing the service does with
    auth. The hotkey it acts as comes from the same X-Chutes-Hotkey header the route verified, so
    the two cannot disagree."""
    db, issue, _, run = provision_env
    kp = _kp()
    mock_vq = AsyncMock(return_value=(Mock(), MagicMock(version=SIGNING_VERSION, rc=False)))
    auth = _verified(kp)

    patches = list(run())
    patches[3] = patch("api.server.service.verify_quote", mock_vq)
    await _provision(db, tuple(patches), kp.ss58_address, auth)

    _, kwargs = mock_vq.await_args
    assert kwargs["auth"] is auth


# ---------------------------------------------------------------------------
# Guarding the trust boundary
#
# Verification happens once, at the edge, so gates downstream READ an identity rather than
# re-deriving it. Against a compromised process that changes nothing -- code execution inside the
# API bypasses any in-process check. What it changes is the cost of a MISTAKE, and the shape of the
# type is what keeps that cost low: there is no "already verified" flag a future caller could fill
# in without verifying. A service either holds a HotkeyAuth that came back from verify_hotkey_auth,
# or it holds None.
# ---------------------------------------------------------------------------


def test_absence_is_none_not_an_empty_auth():
    """The dependencies hand a service None when nothing was offered -- never a blank HotkeyAuth
    that would read as an identity of None and invite `if auth:` to mean the wrong thing."""
    from api.server.util import extract_hotkey_auth

    import asyncio

    request = Mock()
    request.state.body_sha256 = BODY_HASH
    request.headers = {}
    assert asyncio.run(extract_hotkey_auth()(request)) is None


def test_an_unverified_auth_cannot_be_mistaken_for_a_verified_one():
    """Raw material and a proven identity are not interchangeable: verify_hotkey_auth is the only
    way to turn one into the other, and it raises rather than returning something unproven."""
    forged = _auth(_kp(), sign_with=_kp())
    with pytest.raises(UnauthorizedError):
        verify_hotkey_auth(forged)
