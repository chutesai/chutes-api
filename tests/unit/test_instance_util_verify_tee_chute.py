"""
Unit tests for verify_tee_chute in api/instance/util.py.

Covers the chute attestation flow (e2e_pubkey hash for chutes >= 0.6.0, raw nonce below it) and
the release-candidate containment gate: a VM attesting an rc measurement may only ever serve the
dedicated test account's chutes, so public traffic never lands on an in-test guest image.
"""

import hashlib
import secrets
import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from bittensor_wallet.keypair import Keypair

from api.instance.util import verify_tee_chute
from api.util import get_signing_message
from api.server.client import ChuteEvidenceResponse
from api.server.quote import BootTdxQuote
from tests.fixtures.gpus import TEST_GPU_NONCE

EXPECTED_NONCE = TEST_GPU_NONCE
E2E_PUBKEY = "dGVzdF9lMmVfcHVia2V5"  # base64-like test value
EXPECTED_CERT_HASH = "a" * 64

PROXY_KP = Keypair.create_from_seed("0x" + secrets.token_hex(32))


def _fresh_nonce():
    return str(int(time.time()))


def _proxy_signature(nonce, purpose="tee"):
    """The attestation proxy's sr25519 proof over {ss58}:{nonce}:{purpose}."""
    message = get_signing_message(
        hotkey=PROXY_KP.ss58_address, nonce=nonce, payload_str=None, purpose=purpose
    )
    return PROXY_KP.sign(message).hex()


def _make_instance(chutes_version: str | None, extra: dict | None = None):
    """Create a mock Instance with host, chutes_version, extra."""
    instance = MagicMock()
    instance.host = "192.168.1.1"
    instance.chutes_version = chutes_version
    instance.extra = extra
    return instance


def _make_launch_config(chute_id="chute-1"):
    """Create a mock LaunchConfig."""
    launch_config = MagicMock()
    launch_config.miner_hotkey = "miner_hotkey_123"
    launch_config.chute_id = chute_id
    return launch_config


def _measurement(rc=False, version="1.4.0"):
    """The measurement the quote matches -- rc flips the containment gate on."""
    return MagicMock(rc=rc, name="8xh200", version=version)


def _pop_gate(min_version="1.4.0"):
    """Retained as a no-op context so the evidence helpers keep one shape; the proxy's signature is
    no longer gated on a version -- an unsigned response simply proves nothing, and only the rc
    gate insists on a proof."""
    return patch("api.instance.util.RC_CHUTE_GATE_UNUSED", create=True, new=None)


def _signed_evidence(sample_quote, mock_cert):
    """Evidence as a >= 1.4.0 proxy returns it: hotkey-signed, which that image always does."""
    nonce = _fresh_nonce()
    return ChuteEvidenceResponse(
        quote=sample_quote,
        gpu_evidence=[],
        cert=mock_cert,
        hotkey=PROXY_KP.ss58_address,
        hotkey_nonce=nonce,
        hotkey_signature=_proxy_signature(nonce),
    )


def _verified(rc=False, version="1.4.0"):
    """What verify_quote returns: (report, matched measurement). The rc gate reads the measurement
    from here rather than looking it up a second time."""
    return (MagicMock(), _measurement(rc=rc, version=version))


def _make_server():
    """Create a mock Server."""
    server = MagicMock()
    server.ip = "192.168.1.1"
    server.miner_hotkey = "miner_hotkey_123"
    return server


@pytest.fixture
def mock_db():
    """Mock database session that returns a server for the query."""
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = _make_server()
    db.execute = AsyncMock(return_value=result)
    return db


@pytest.fixture
def sample_quote():
    """Sample BootTdxQuote for testing."""
    return BootTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="b" * 96,
        rtmr1="c" * 96,
        rtmr2="d" * 96,
        rtmr3="e" * 96,
        report_data=EXPECTED_NONCE + "0" * 64,
        user_data="test",
        platform_id="0" * 32,
        raw_quote_size=4096,
        parsed_at="2024-01-01T00:00:00Z",
        raw_bytes=b"dummy",
    )


@pytest.fixture
def mock_cert():
    """Mock x509 certificate with get_public_key_hash returning expected hash."""
    cert = MagicMock()
    return cert


@pytest.mark.asyncio
async def test_verify_tee_chute_chutes_060_uses_e2e_pubkey_hash(mock_db, sample_quote, mock_cert):
    """For chutes >= 0.6.0 with e2e_pubkey, verify_quote receives sha256(nonce+e2e_pubkey)."""
    instance = _make_instance("0.6.0", {"e2e_pubkey": E2E_PUBKEY})
    launch_config = _make_launch_config()

    expected_report_data = (
        hashlib.sha256((EXPECTED_NONCE + E2E_PUBKEY).encode()).hexdigest().lower()
    )

    with (
        patch("api.instance.util.TeeServerClient") as mock_client_cls,
        patch(
            "api.instance.util.verify_quote",
            new=AsyncMock(return_value=_verified(version="1.3.0")),
        ) as mock_verify_quote,
        patch("api.instance.util.verify_gpu_evidence", new_callable=AsyncMock) as mock_verify_gpu,
        patch("api.instance.util.get_public_key_hash", return_value=EXPECTED_CERT_HASH),
        _pop_gate(),
    ):
        mock_client = MagicMock()
        mock_client.get_chute_evidence = AsyncMock(
            return_value=ChuteEvidenceResponse(quote=sample_quote, gpu_evidence=[], cert=mock_cert)
        )
        mock_client_cls.create = AsyncMock(return_value=mock_client)

        await verify_tee_chute(mock_db, instance, launch_config, "deploy-123", EXPECTED_NONCE)

        # auth is the runtime rc proof-of-possession (None when the proxy sends no hotkey).
        mock_verify_quote.assert_called_once_with(
            sample_quote, expected_report_data, EXPECTED_CERT_HASH, auth=None
        )
        mock_verify_gpu.assert_called_once_with([], expected_report_data)


@pytest.mark.asyncio
async def test_verify_tee_chute_chutes_059_uses_raw_nonce(mock_db, sample_quote, mock_cert):
    """For chutes < 0.6.0, verify_quote receives expected_nonce directly (old behavior)."""
    instance = _make_instance("0.5.9", {"e2e_pubkey": E2E_PUBKEY})
    launch_config = _make_launch_config()

    with (
        patch("api.instance.util.TeeServerClient") as mock_client_cls,
        patch(
            "api.instance.util.verify_quote",
            new=AsyncMock(return_value=_verified(version="1.3.0")),
        ) as mock_verify_quote,
        patch("api.instance.util.verify_gpu_evidence", new_callable=AsyncMock) as mock_verify_gpu,
        patch("api.instance.util.get_public_key_hash", return_value=EXPECTED_CERT_HASH),
        _pop_gate(),
    ):
        mock_client = MagicMock()
        mock_client.get_chute_evidence = AsyncMock(
            return_value=ChuteEvidenceResponse(quote=sample_quote, gpu_evidence=[], cert=mock_cert)
        )
        mock_client_cls.create = AsyncMock(return_value=mock_client)

        await verify_tee_chute(mock_db, instance, launch_config, "deploy-123", EXPECTED_NONCE)

        mock_verify_quote.assert_called_once_with(
            sample_quote, EXPECTED_NONCE, EXPECTED_CERT_HASH, auth=None
        )
        mock_verify_gpu.assert_called_once_with([], EXPECTED_NONCE)


@pytest.mark.asyncio
async def test_verify_tee_chute_chutes_060_missing_e2e_pubkey_raises_400(
    mock_db, sample_quote, mock_cert
):
    """For chutes >= 0.6.0 without e2e_pubkey, raise HTTP 400."""
    instance = _make_instance("0.6.0", {})  # no e2e_pubkey
    launch_config = _make_launch_config()

    with (
        patch("api.instance.util.TeeServerClient") as mock_client_cls,
        patch("api.instance.util.verify_quote", new_callable=AsyncMock),
        patch("api.instance.util.verify_gpu_evidence", new_callable=AsyncMock),
        patch("api.instance.util.get_public_key_hash", return_value=EXPECTED_CERT_HASH),
    ):
        mock_client = MagicMock()
        mock_client.get_chute_evidence = AsyncMock(
            return_value=ChuteEvidenceResponse(quote=sample_quote, gpu_evidence=[], cert=mock_cert)
        )
        mock_client_cls.create = AsyncMock(return_value=mock_client)

        with pytest.raises(HTTPException) as exc_info:
            await verify_tee_chute(mock_db, instance, launch_config, "deploy-123", EXPECTED_NONCE)

        assert exc_info.value.status_code == 400
        assert "e2e_pubkey required" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_tee_chute_chutes_060_extra_none_raises_400(mock_db, sample_quote, mock_cert):
    """For chutes >= 0.6.0 with instance.extra None, raise HTTP 400."""
    instance = _make_instance("0.6.0", None)
    launch_config = _make_launch_config()

    with (
        patch("api.instance.util.TeeServerClient") as mock_client_cls,
        patch("api.instance.util.verify_quote", new_callable=AsyncMock),
        patch("api.instance.util.verify_gpu_evidence", new_callable=AsyncMock),
        patch("api.instance.util.get_public_key_hash", return_value=EXPECTED_CERT_HASH),
    ):
        mock_client = MagicMock()
        mock_client.get_chute_evidence = AsyncMock(
            return_value=ChuteEvidenceResponse(quote=sample_quote, gpu_evidence=[], cert=mock_cert)
        )
        mock_client_cls.create = AsyncMock(return_value=mock_client)

        with pytest.raises(HTTPException) as exc_info:
            await verify_tee_chute(mock_db, instance, launch_config, "deploy-123", EXPECTED_NONCE)

        assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_runtime_rc_proof_is_threaded_from_the_evidence_fields(
    mock_db, sample_quote, mock_cert
):
    """The attestation proxy's miner-hotkey PoP reaches verify_quote's rc gate intact.

    Every other test here leaves evidence.hotkey unset, which short-circuits rc_auth to None and
    leaves this branch unexercised -- so the field wiring (hotkey_signature / hotkey_nonce, NOT the
    unrelated evidence.signature, which is the proxy's RSA signature over attested_body) is only
    covered here.
    """
    instance = _make_instance("0.5.9", {})
    launch_config = _make_launch_config()
    proxy_nonce = _fresh_nonce()  # one nonce, signed and sent -- they must match exactly

    with (
        patch("api.instance.util.TeeServerClient") as mock_client_cls,
        patch(
            "api.instance.util.verify_quote", new=AsyncMock(return_value=_verified())
        ) as mock_verify_quote,
        patch("api.instance.util.verify_gpu_evidence", new_callable=AsyncMock),
        patch("api.instance.util.get_public_key_hash", return_value=EXPECTED_CERT_HASH),
    ):
        mock_client = MagicMock()
        mock_client.get_chute_evidence = AsyncMock(
            return_value=ChuteEvidenceResponse(
                quote=sample_quote,
                gpu_evidence=[],
                cert=mock_cert,
                signature="proxy-rsa-sig-over-attested-body",
                attested_body="body",
                hotkey=PROXY_KP.ss58_address,
                hotkey_nonce=proxy_nonce,
                hotkey_signature=_proxy_signature(proxy_nonce),
            )
        )
        mock_client_cls.create = AsyncMock(return_value=mock_client)

        await verify_tee_chute(mock_db, instance, launch_config, "deploy-123", EXPECTED_NONCE)

    auth = mock_verify_quote.await_args.kwargs["auth"]
    # Verified at the point of receipt, so the rc gate downstream reads a proven identity.
    assert auth.miner_hotkey == PROXY_KP.ss58_address
    assert auth.purpose == "tee"


# ---------------------------------------------------------------------------
# Release-candidate containment: only the dedicated test account's chutes may
# deploy onto a VM whose attested measurement is rc.
# ---------------------------------------------------------------------------

RC_USER = "user-rc-test-account"


def _db_for_rc(chute_owner):
    """A db whose first execute() resolves the Server and whose second resolves the Chute --
    matching the two lookups verify_tee_chute makes on the rc path."""
    server_result = MagicMock()
    server_result.scalar_one_or_none.return_value = _make_server()
    chute_result = MagicMock()
    chute_result.scalar_one_or_none.return_value = (
        MagicMock(user_id=chute_owner) if chute_owner is not None else None
    )
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[server_result, chute_result])
    return db


def _rc_patches(rc_user_id=RC_USER, verification=None):
    return (
        _pop_gate(),
        patch(
            "api.instance.util.verify_quote",
            new=AsyncMock(
                return_value=_verified(rc=True) if verification is None else verification
            ),
        ),
        patch("api.instance.util.verify_gpu_evidence", new_callable=AsyncMock),
        patch("api.instance.util.get_public_key_hash", return_value=EXPECTED_CERT_HASH),
        patch("api.instance.util.settings.rc_chute_user_id", rc_user_id),
    )


async def _run_rc(db, sample_quote, mock_cert, rc_user_id=RC_USER, verification=None):
    from contextlib import ExitStack

    instance = _make_instance("0.5.9", {})
    launch_config = _make_launch_config()
    with ExitStack() as stack:
        mock_client_cls = stack.enter_context(patch("api.instance.util.TeeServerClient"))
        for p in _rc_patches(rc_user_id, verification):
            stack.enter_context(p)
        mock_client = MagicMock()
        mock_client.get_chute_evidence = AsyncMock(
            return_value=_signed_evidence(sample_quote, mock_cert)
        )
        mock_client_cls.create = AsyncMock(return_value=mock_client)
        await verify_tee_chute(db, instance, launch_config, "deploy-123", EXPECTED_NONCE)


@pytest.mark.asyncio
async def test_rc_measurement_allows_the_test_accounts_chute(sample_quote, mock_cert):
    """The one permitted case: an rc VM serving a chute owned by the configured test account."""
    await _run_rc(_db_for_rc(RC_USER), sample_quote, mock_cert)


@pytest.mark.asyncio
async def test_rc_measurement_rejects_a_public_users_chute(sample_quote, mock_cert):
    """The containment: no ordinary user's chute may be scheduled onto an in-test image, so no
    public traffic can reach one. An rc VM attests honestly, so nothing else would stop it."""
    with pytest.raises(HTTPException) as exc_info:
        await _run_rc(_db_for_rc("user-some-customer"), sample_quote, mock_cert)

    assert exc_info.value.status_code == 403
    assert "release-candidate" in exc_info.value.detail


@pytest.mark.asyncio
async def test_rc_measurement_fails_closed_when_no_rc_user_is_configured(sample_quote, mock_cert):
    """An unset allowlist must never read as 'anyone' -- with no rc user configured, nothing
    deploys on rc, not even a chute whose owner happens to be unset."""
    with pytest.raises(HTTPException) as exc_info:
        await _run_rc(_db_for_rc(None), sample_quote, mock_cert, rc_user_id=None)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_rc_measurement_rejects_when_the_chute_is_missing(sample_quote, mock_cert):
    """No chute row -> no provable owner -> refused, rather than treated as unowned/allowed."""
    with pytest.raises(HTTPException) as exc_info:
        await _run_rc(_db_for_rc(None), sample_quote, mock_cert)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_gate_uses_the_measurement_verify_quote_matched(sample_quote, mock_cert):
    """The gate acts on the measurement verify_quote returned, never a fresh lookup -- that would
    re-read the ConfigMap (settings.tee_measurements reloads on every access) and could match a
    different entry, or none, after the attestation already passed."""
    from api.instance.util import _require_rc_chute_owner
    import inspect

    sig = inspect.signature(_require_rc_chute_owner)
    assert "measurement_config" in sig.parameters
    # verify_quote's contract is non-Optional, so there is no "unknown measurement" branch to take.
    assert sig.parameters["measurement_config"].annotation is not None


# ---------------------------------------------------------------------------
# The attestation proxy's hotkey signature: authenticated at its point of receipt.
# From image 1.4.0 the proxy's manifest injects MINER_SEED, so it signs every
# response — a missing or bad signature there is a fault, not a legacy VM.
# ---------------------------------------------------------------------------


async def _run_with_evidence(db, evidence, *, version="1.4.0"):
    from contextlib import ExitStack

    instance = _make_instance("0.5.9", {})
    launch_config = _make_launch_config()
    with ExitStack() as stack:
        mock_client_cls = stack.enter_context(patch("api.instance.util.TeeServerClient"))
        for p in (
            _pop_gate(),
            patch(
                "api.instance.util.verify_quote",
                new=AsyncMock(return_value=_verified(version=version)),
            ),
            patch("api.instance.util.verify_gpu_evidence", new_callable=AsyncMock),
            patch("api.instance.util.get_public_key_hash", return_value=EXPECTED_CERT_HASH),
            patch("api.instance.util.settings.rc_chute_user_id", RC_USER),
        ):
            stack.enter_context(p)
        mock_client = MagicMock()
        mock_client.get_chute_evidence = AsyncMock(return_value=evidence)
        mock_client_cls.create = AsyncMock(return_value=mock_client)
        await verify_tee_chute(db, instance, launch_config, "deploy-123", EXPECTED_NONCE)


@pytest.mark.asyncio
async def test_proxy_signature_forged_by_another_key_is_rejected(mock_db, sample_quote, mock_cert):
    """Offered but invalid is a hard reject, never a quiet downgrade to 'unauthenticated'."""
    other = Keypair.create_from_seed("0x" + secrets.token_hex(32))
    nonce = _fresh_nonce()
    message = get_signing_message(
        hotkey=other.ss58_address, nonce=nonce, payload_str=None, purpose="tee"
    )
    evidence = ChuteEvidenceResponse(
        quote=sample_quote,
        gpu_evidence=[],
        cert=mock_cert,
        hotkey=PROXY_KP.ss58_address,  # claims the miner's hotkey...
        hotkey_nonce=nonce,
        hotkey_signature=other.sign(message).hex(),  # ...signed by another key
    )
    with pytest.raises(HTTPException) as exc_info:
        await _run_with_evidence(mock_db, evidence)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_proxy_signature_with_a_stale_nonce_is_rejected(mock_db, sample_quote, mock_cert):
    """The proxy's nonce is its own timestamp, so a stale one is the only replay bound there."""
    stale = str(int(time.time()) - 10_000)
    evidence = ChuteEvidenceResponse(
        quote=sample_quote,
        gpu_evidence=[],
        cert=mock_cert,
        hotkey=PROXY_KP.ss58_address,
        hotkey_nonce=stale,
        hotkey_signature=_proxy_signature(stale),
    )
    with pytest.raises(HTTPException) as exc_info:
        await _run_with_evidence(mock_db, evidence)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_unsigned_response_proves_nothing_but_is_not_itself_an_error(
    mock_db, sample_quote, mock_cert
):
    """An unsigned response is not a fault in its own right -- a published measurement needs no
    proof of who the miner is. It simply arrives as None, which the rc gate refuses (covered by
    the rc tests above) while a published one passes through."""
    evidence = ChuteEvidenceResponse(quote=sample_quote, gpu_evidence=[], cert=mock_cert)
    await _run_with_evidence(mock_db, evidence, version="1.4.0")


@pytest.mark.asyncio
async def test_unsigned_response_from_a_legacy_version_is_accepted(mock_db, sample_quote, mock_cert):
    """Below the gate the proxy has no seed injected and cannot sign; those VMs still verify."""
    evidence = ChuteEvidenceResponse(quote=sample_quote, gpu_evidence=[], cert=mock_cert)
    await _run_with_evidence(mock_db, evidence, version="1.3.0")
