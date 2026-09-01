"""
Unit tests for registry mTLS authentication and VM root-CA provisioning.

Covers:
- verify_leaf_cert_signed_by_ca() utility
- POST /servers/{vm_name}/provision (record CA + issue storage secrets) via
  process_provision_request; legacy /luks/attest records no CA
- GET /registry/auth dual-auth logic
"""

import hashlib
import datetime
import secrets
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

import pybase64
from urllib.parse import quote as url_quote

from bittensor_wallet.keypair import Keypair

from api.registry.router import registry_auth
from api.util import get_signing_message
from api.server.util import (
    verify_hotkey_auth,
    verify_leaf_cert_signed_by_ca,
    verify_server_cert,
    get_public_key_hash,
)
from api.server.quote import RuntimeTdxQuote
from api.server.service import (
    process_provision_request,
    process_luks_attest_request,
    require_server_mtls,
    verify_server,
    sync_vm_root_ca_from_boot_record,
)
from api.server.schemas import (
    Server,
    ProvisionRequest,
    LuksAttestRequest,
    LuksVolumeRotation,
    StorageProvisionResult,
    HotkeyAuth,
)
from api.server.exceptions import AttestationError, ServerNotFoundError, InvalidClientCertError


# ---------------------------------------------------------------------------
# Certificate helpers
# ---------------------------------------------------------------------------


def _gen_rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())


def _make_ca_cert(key, subject_cn="sek8s-vm-root-ca"):
    """Generate a self-signed CA certificate."""
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "chutes"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "sek8s"),
            x509.NameAttribute(NameOID.COMMON_NAME, subject_cn),
        ]
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return cert


def _make_leaf_cert(leaf_key, ca_key, ca_cert, subject_cn="sek8s-vm-registry-client"):
    """Generate a leaf certificate signed by a CA key."""
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "chutes"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "sek8s"),
            x509.NameAttribute(NameOID.COMMON_NAME, subject_cn),
        ]
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(ca_cert.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=1))
        .sign(ca_key, hashes.SHA256())
    )
    return cert


def _cert_pem(cert) -> str:
    return cert.public_bytes(serialization.Encoding.PEM).decode()


def _ca_pubkey_der_hash(ca_cert) -> str:
    """Compute SHA256(SubjectPublicKeyInfo DER) — what the VM puts in REPORTDATA."""
    pubkey_der = ca_cert.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(pubkey_der).hexdigest()


def _make_runtime_quote(report_data_hex: str) -> RuntimeTdxQuote:
    """Build a minimal RuntimeTdxQuote with the given report_data hex string."""
    return RuntimeTdxQuote(
        version=4,
        att_key_type=2,
        tee_type=0x81,
        mrtd="a" * 96,
        rtmr0="d" * 96,
        rtmr1="e" * 96,
        rtmr2="f" * 96,
        rtmr3="0" * 96,
        report_data=report_data_hex,
        user_data=None,
        platform_id="0" * 32,
        raw_quote_size=4096,
        parsed_at=datetime.datetime.utcnow().isoformat(),
        raw_bytes=b"dummy",
    )


# ---------------------------------------------------------------------------
# verify_leaf_cert_signed_by_ca — unit tests
# ---------------------------------------------------------------------------


def test_verify_leaf_cert_signed_by_ca_valid():
    """Leaf signed by the registered CA passes verification."""
    ca_key = _gen_rsa_key()
    ca_cert = _make_ca_cert(ca_key)
    leaf_key = _gen_rsa_key()
    leaf_cert = _make_leaf_cert(leaf_key, ca_key, ca_cert)

    # Should not raise (both leaf and CA are parsed Certificate objects)
    verify_leaf_cert_signed_by_ca(leaf_cert, ca_cert)


def test_verify_leaf_cert_signed_by_ca_wrong_ca():
    """Leaf signed by a different CA is rejected with 403."""
    ca_key = _gen_rsa_key()
    ca_cert = _make_ca_cert(ca_key)

    other_ca_key = _gen_rsa_key()
    other_ca_cert = _make_ca_cert(other_ca_key, subject_cn="other-ca")

    leaf_key = _gen_rsa_key()
    # Sign leaf with other_ca_key but present registered ca_cert for verification
    leaf_cert = _make_leaf_cert(leaf_key, other_ca_key, other_ca_cert)

    with pytest.raises(AttestationError) as exc_info:
        verify_leaf_cert_signed_by_ca(leaf_cert, ca_cert)
    assert exc_info.value.status_code == 403


def test_verify_leaf_cert_signed_by_ca_self_signed_leaf_rejected():
    """Self-signed leaf cert (issuer == subject) is rejected with 403."""
    ca_key = _gen_rsa_key()
    ca_cert = _make_ca_cert(ca_key)

    # Build a self-signed leaf (issuer == subject)
    leaf_key = _gen_rsa_key()
    leaf_cert = _make_ca_cert(leaf_key, subject_cn="sek8s-vm-registry-client")

    with pytest.raises(AttestationError) as exc_info:
        verify_leaf_cert_signed_by_ca(leaf_cert, ca_cert)
    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# Server.vm_root_ca_certificate — parsing property
# ---------------------------------------------------------------------------


def test_vm_root_ca_certificate_parses_pem():
    """The property parses the stored PEM into an x509.Certificate."""
    ca_cert = _make_ca_cert(_gen_rsa_key())
    server = Server(vm_root_ca_cert=_cert_pem(ca_cert))
    parsed = server.vm_root_ca_certificate
    assert parsed.subject == ca_cert.subject


def test_vm_root_ca_certificate_none_when_unset():
    """No CA on file -> None (the pre-provision / legacy signal)."""
    assert Server(vm_root_ca_cert=None).vm_root_ca_certificate is None


def test_vm_root_ca_certificate_malformed_raises():
    """A malformed stored value is a data-integrity bug and is allowed to raise."""
    with pytest.raises(ValueError):
        Server(vm_root_ca_cert="not-a-ca-cert").vm_root_ca_certificate


# ---------------------------------------------------------------------------
# verify_server_cert / require_server_mtls — shared mTLS auth
# ---------------------------------------------------------------------------


def test_verify_server_cert_valid():
    """Leaf signed by the server's registered CA passes."""
    ca_key = _gen_rsa_key()
    ca_cert = _make_ca_cert(ca_key)
    leaf_cert = _make_leaf_cert(_gen_rsa_key(), ca_key, ca_cert)

    server = MagicMock()
    server.vm_root_ca_certificate = ca_cert
    # Should not raise
    verify_server_cert(leaf_cert, server)


def test_verify_server_cert_no_client_cert():
    """No client cert presented -> NoClientCertError (403)."""
    server = MagicMock()
    server.vm_root_ca_certificate = MagicMock()  # CA on file, but no client cert presented
    with pytest.raises(AttestationError) as exc_info:
        verify_server_cert(None, server)
    assert exc_info.value.status_code == 403


def test_verify_server_cert_no_registered_ca():
    """VM has no CA on file -> NoClientCertError (403)."""
    ca_key = _gen_rsa_key()
    leaf_cert = _make_leaf_cert(_gen_rsa_key(), ca_key, _make_ca_cert(ca_key))
    server = MagicMock()
    server.vm_root_ca_certificate = None
    with pytest.raises(AttestationError) as exc_info:
        verify_server_cert(leaf_cert, server)
    assert exc_info.value.status_code == 403


def test_verify_server_cert_wrong_ca():
    """Leaf signed by a different CA -> 403 (delegated to verify_leaf_cert_signed_by_ca)."""
    other_key = _gen_rsa_key()
    leaf_cert = _make_leaf_cert(_gen_rsa_key(), other_key, _make_ca_cert(other_key, "other"))

    registered_ca = _make_ca_cert(_gen_rsa_key())
    server = MagicMock()
    server.vm_root_ca_certificate = registered_ca
    with pytest.raises(AttestationError) as exc_info:
        verify_server_cert(leaf_cert, server)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_require_server_mtls_returns_authenticated_server():
    """The dependency resolves the server by (hotkey, vm_name), verifies the leaf, returns it."""
    ca_key = _gen_rsa_key()
    ca_cert = _make_ca_cert(ca_key)
    leaf_cert = _make_leaf_cert(_gen_rsa_key(), ca_key, ca_cert)

    mock_server = MagicMock()
    mock_server.vm_root_ca_certificate = ca_cert
    db = AsyncMock()

    with patch("api.server.service.get_server_by_name", return_value=mock_server) as mock_get:
        result = await require_server_mtls(vm_name="vm1", db=db, hotkey="hk", client_cert=leaf_cert)

    mock_get.assert_awaited_once_with(db, "hk", "vm1")
    assert result is mock_server


@pytest.mark.asyncio
async def test_require_server_mtls_unknown_vm():
    """Unknown (hotkey, vm_name) -> ServerNotFoundError propagates."""
    ca_key = _gen_rsa_key()
    leaf_cert = _make_leaf_cert(_gen_rsa_key(), ca_key, _make_ca_cert(ca_key))
    db = AsyncMock()

    with patch("api.server.service.get_server_by_name", side_effect=ServerNotFoundError("vm1")):
        with pytest.raises(ServerNotFoundError):
            await require_server_mtls(vm_name="vm1", db=db, hotkey="hk", client_cert=leaf_cert)


@pytest.mark.asyncio
async def test_require_server_mtls_no_registered_ca():
    """A VM that has not provisioned a CA -> NoClientCertError (403)."""
    ca_key = _gen_rsa_key()
    leaf_cert = _make_leaf_cert(_gen_rsa_key(), ca_key, _make_ca_cert(ca_key))
    mock_server = MagicMock()
    mock_server.vm_root_ca_certificate = None
    db = AsyncMock()

    with patch("api.server.service.get_server_by_name", return_value=mock_server):
        with pytest.raises(AttestationError) as exc_info:
            await require_server_mtls(vm_name="vm1", db=db, hotkey="hk", client_cert=leaf_cert)
    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# process_provision_request / record_vm_ca_identity — service-level tests
# ---------------------------------------------------------------------------


def _hotkey_auth(nonce, *, body_sha256="body-hash", keypair=None):
    """The auth a /provision request arrives with once the edge has verified it.

    Only 1.4.0+ VMs reach /provision and every one of them signs, so require_hotkey_auth demands a
    proof at the door -- these tests carry a real signature rather than stubbing the dependency.
    """
    keypair = keypair or Keypair.create_from_seed("0x" + secrets.token_hex(32))
    message = get_signing_message(
        hotkey=keypair.ss58_address, nonce=nonce, payload_str=None, payload_hash=body_sha256
    )
    return keypair, verify_hotkey_auth(
        HotkeyAuth(
            miner_hotkey=keypair.ss58_address,
            signature=keypair.sign(message).hex(),
            nonce=nonce,
            body_sha256=body_sha256,
        )
    )


@pytest.mark.asyncio
async def test_provision_records_ca_and_returns_secrets():
    """
    /provision verifies the runtime quote against SHA256(client cert pubkey), records the VM
    root CA (upsert onto the VM's boot record + write-through onto an existing server), and
    returns rotated volumes + k3s + confirm nonce.
    """
    ca_key = _gen_rsa_key()
    ca_cert = _make_ca_cert(ca_key)

    mock_server = MagicMock()
    vm_config = MagicMock()
    vm_config.k3s_encryption_key = "encrypted-k3s"  # truthy -> reuse existing key path
    volumes_data = {"storage": LuksVolumeRotation(current="cur", next="nxt")}
    # record_vm_ca_identity updates the VM's latest boot record (fetched via db.execute).
    boot_record = MagicMock()
    exec_result = MagicMock()
    exec_result.scalar_one_or_none.return_value = boot_record
    db = AsyncMock()
    db.execute = AsyncMock(return_value=exec_result)

    with (
        patch("api.server.service.RuntimeTdxQuote") as mock_quote_cls,
        patch(
            "api.server.service.verify_quote",
            new=AsyncMock(return_value=(MagicMock(), MagicMock(version="1.4.0", rc=False))),
        ) as mock_verify,
        patch("api.server.service.get_server_by_name", return_value=mock_server),
        patch(
            "api.server.service.rotate_luks_passphrases",
            new_callable=AsyncMock,
            return_value=(volumes_data, vm_config),
        ),
        patch(
            "api.server.service.generate_confirm_nonce",
            new_callable=AsyncMock,
            return_value="confirm-nonce",
        ),
        patch("api.server.service.decrypt_passphrase", return_value="k3s-key"),
    ):
        mock_quote_cls.from_base64.return_value = _make_runtime_quote("0" * 128)
        body = ProvisionRequest(
            quote=pybase64.b64encode(b"q").decode(), volumes=["storage", "tdx-cache"]
        )
        keypair, auth = _hotkey_auth("nonce123")
        result = await process_provision_request(
            db, keypair.ss58_address, "vm1", body, "nonce123", ca_cert, auth=auth
        )

    # The CA + provision quote are written onto the VM's current boot record.
    assert boot_record.vm_root_ca_cert == _cert_pem(ca_cert)
    assert boot_record.provision_quote == body.quote
    # The existing server's synced copy is written through.
    assert mock_server.vm_root_ca_cert == _cert_pem(ca_cert)
    # The runtime quote is verified against the nonce + SHA256(client cert pubkey).
    mock_verify.assert_awaited_once()
    assert mock_verify.await_args.args[1] == "nonce123"
    assert mock_verify.await_args.args[2] == get_public_key_hash(ca_cert)
    # Storage-provisioning secrets are returned.
    assert result.volumes == volumes_data
    assert result.k3s_encryption_key == "k3s-key"
    assert result.confirm_nonce == "confirm-nonce"


@pytest.mark.asyncio
async def test_provision_without_server_records_ca_in_boot_record():
    """/provision runs in initramfs BEFORE POST /servers, so it must NOT require a server row.
    With no server, it records the CA on the VM's boot record (no 404, no write-through)."""
    ca_key = _gen_rsa_key()
    ca_cert = _make_ca_cert(ca_key)
    expected = StorageProvisionResult(volumes={}, confirm_nonce="cn", k3s_encryption_key="k")
    boot_record = MagicMock()
    exec_result = MagicMock()
    exec_result.scalar_one_or_none.return_value = boot_record
    db = AsyncMock()
    db.execute = AsyncMock(return_value=exec_result)

    with (
        patch("api.server.service.RuntimeTdxQuote") as mock_quote_cls,
        patch(
            "api.server.service.verify_quote",
            new=AsyncMock(return_value=(MagicMock(), MagicMock(version="1.4.0", rc=False))),
        ),
        patch(
            "api.server.service.get_server_by_name",
            side_effect=ServerNotFoundError("vm1"),
        ),
        patch(
            "api.server.service._issue_storage_secrets",
            new=AsyncMock(return_value=expected),
        ),
    ):
        mock_quote_cls.from_base64.return_value = _make_runtime_quote("0" * 128)
        body = ProvisionRequest(
            quote=pybase64.b64encode(b"q").decode(), volumes=["storage", "tdx-cache"]
        )
        # No ServerNotFoundError — provision succeeds pre-registration.
        keypair, auth = _hotkey_auth("nonce")
        result = await process_provision_request(
            db, keypair.ss58_address, "vm1", body, "nonce", ca_cert, auth=auth
        )

    assert result is expected
    # CA recorded on the boot record; no server, so no write-through.
    assert boot_record.vm_root_ca_cert == _cert_pem(ca_cert)


@pytest.mark.asyncio
async def test_provision_no_matching_boot_record_fails_closed():
    """The provision nonce must match a boot record from this boot; a miss (no boot attestation on
    record for this VM/nonce) fails closed rather than fabricating a record or attaching to a
    stale/failed one."""
    ca_cert = _make_ca_cert(_gen_rsa_key())
    exec_result = MagicMock()
    exec_result.scalar_one_or_none.return_value = None  # no boot record matches the nonce
    db = AsyncMock()
    db.execute = AsyncMock(return_value=exec_result)

    with (
        patch("api.server.service.RuntimeTdxQuote") as mock_quote_cls,
        patch(
            "api.server.service.verify_quote",
            new=AsyncMock(return_value=(MagicMock(), MagicMock(version="1.4.0", rc=False))),
        ),
    ):
        mock_quote_cls.from_base64.return_value = _make_runtime_quote("0" * 128)
        body = ProvisionRequest(
            quote=pybase64.b64encode(b"q").decode(), volumes=["storage", "tdx-cache"]
        )
        with pytest.raises(AttestationError):
            await process_provision_request(db, "hk", "vm1", body, "stale-nonce", ca_cert)


@pytest.mark.asyncio
async def test_sync_vm_root_ca_from_boot_record_stamps_latest_ca():
    """register_server bridge: the latest provision-phase boot-record CA is stamped onto the
    Server row so mTLS consumers (which read server.vm_root_ca_cert) see it."""
    ca_cert = _make_ca_cert(_gen_rsa_key())
    ca_pem = _cert_pem(ca_cert)

    server = Server(server_id="s1", miner_hotkey="hk", name="vm1", ip="1.2.3.4")
    result = MagicMock()
    result.scalar_one_or_none.return_value = ca_pem
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result)

    await sync_vm_root_ca_from_boot_record(db, server)
    assert server.vm_root_ca_cert == ca_pem


@pytest.mark.asyncio
async def test_sync_vm_root_ca_noop_when_no_boot_record():
    """Legacy VM that never recorded a CA: sync is a no-op (server keeps CERT_NONE fallback)."""
    server = Server(server_id="s1", miner_hotkey="hk", name="vm1", ip="1.2.3.4")
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result)

    await sync_vm_root_ca_from_boot_record(db, server)
    assert server.vm_root_ca_cert is None


@pytest.mark.asyncio
async def test_verify_server_rejects_evidence_cert_not_signed_by_boot_record_ca():
    """Registration attestation is bound to the initramfs-measured CA (read from the boot record,
    since the server row's CA is only stamped after success): if the attestation proxy's server
    cert is NOT signed by that CA, verification fails (403) before the quote is verified."""
    ca_key = _gen_rsa_key()
    ca_cert = _make_ca_cert(ca_key)
    # Evidence cert signed by a DIFFERENT CA than the one on the boot record.
    other_key = _gen_rsa_key()
    other_ca = _make_ca_cert(other_key, subject_cn="other-ca")
    evidence_cert = _make_leaf_cert(_gen_rsa_key(), other_key, other_ca)

    # Server row has NO CA yet (not stamped until attestation passes).
    server = Server(server_id="s1", miner_hotkey="hk", name="vm1", ip="1.2.3.4")
    quote = MagicMock()
    quote.raw_bytes = b"q"
    client = MagicMock()
    from api.server.client import ServerEvidenceResponse

    client.get_server_evidence = AsyncMock(
        return_value=ServerEvidenceResponse(quote=quote, gpu_evidence=[], cert=evidence_cert)
    )
    db = AsyncMock()

    with (
        patch("api.server.service.TeeServerClient.create", new=AsyncMock(return_value=client)),
        patch(
            "api.server.service.get_matching_measurement_config",
            return_value=MagicMock(rc=False, version="1.4.0"),
        ),
        # The CA to bind against comes from the boot record.
        patch(
            "api.server.service.get_boot_record_ca", new=AsyncMock(return_value=_cert_pem(ca_cert))
        ),
        patch("api.server.service.verify_quote", new_callable=AsyncMock) as mock_vq,
    ):
        with pytest.raises(AttestationError) as exc_info:
            await verify_server(db, server, "hk", [])
    assert exc_info.value.status_code == 403
    # Rejected at the CA binding, before the quote is even verified.
    mock_vq.assert_not_awaited()
    # The unverified CA is never stamped onto the server row.
    assert server.vm_root_ca_cert is None


@pytest.mark.asyncio
async def test_luks_attest_records_no_ca():
    """Legacy /luks/attest rotates storage but never records a VM root CA."""
    vm_config = MagicMock()
    vm_config.k3s_encryption_key = "encrypted-k3s"
    volumes_data = {"storage": LuksVolumeRotation(current="cur", next="nxt")}
    db = AsyncMock()

    with (
        patch("api.server.service.RuntimeTdxQuote") as mock_quote_cls,
        patch("api.server.service.verify_quote", new_callable=AsyncMock),
        patch("api.server.service.get_server_by_name") as mock_get_server,
        patch(
            "api.server.service.rotate_luks_passphrases",
            new_callable=AsyncMock,
            return_value=(volumes_data, vm_config),
        ),
        patch(
            "api.server.service.generate_confirm_nonce",
            new_callable=AsyncMock,
            return_value="confirm-nonce",
        ),
        patch("api.server.service.decrypt_passphrase", return_value="k3s-key"),
    ):
        mock_quote_cls.from_base64.return_value = _make_runtime_quote("0" * 128)
        body = LuksAttestRequest(
            quote=pybase64.b64encode(b"q").decode(), volumes=["storage", "tdx-cache"]
        )
        result = await process_luks_attest_request(db, "hk", "vm1", body, "nonce", "cert-hash")

    # The legacy path never looks up the server to record a CA (that only happens in /provision).
    mock_get_server.assert_not_called()
    assert result.k3s_encryption_key == "k3s-key"


@pytest.mark.asyncio
async def test_provision_confirm_delegates_to_shared_helper():
    """POST /provision/confirm delegates to the shared process_luks_confirm."""
    from api.server.router import provision_confirm
    from api.server.schemas import (
        LuksConfirmRequest,
        LuksVolumeConfirmStatus,
        LuksConfirmResult,
    )

    db = AsyncMock()
    body = LuksConfirmRequest(volumes={"storage": LuksVolumeConfirmStatus(rotated=True)})

    with patch(
        "api.server.router.process_luks_confirm",
        new_callable=AsyncMock,
        return_value=LuksConfirmResult(volumes={"storage": {"result": "promoted"}}),
    ) as mock_confirm:
        resp = await provision_confirm(
            vm_name="vm1",
            body=body,
            db=db,
            hotkey="hk",
            _mtls=None,
            _=None,
        )

    # The route's require_hotkey_auth dependency has already proven the hotkey this call acts as,
    # so the shared service is handed the same identity the legacy route passes.
    mock_confirm.assert_awaited_once_with(db, "hk", "vm1", body)
    assert resp.status == "confirmed"
    assert resp.volumes == {"storage": {"result": "promoted"}}


def test_provision_routes_registered_and_vm_root_ca_removed():
    """The /provision routes are registered and the old PUT /vm-root-ca route is gone (404)."""
    from api.server.router import router

    paths = {r.path for r in router.routes}
    assert "/{vm_name}/provision" in paths
    assert "/{vm_name}/provision/confirm" in paths
    assert not any("vm-root-ca" in p for p in paths)


# ---------------------------------------------------------------------------
# Registry dual-auth — GET /registry/auth
# ---------------------------------------------------------------------------


def _make_registry_request(
    *,
    client_cert: str | None = None,
    real_ip: str | None = None,
    proxy_auth: str | None = None,
    hotkey: str | None = None,
    signature: str | None = None,
    nonce: str | None = None,
):
    """Build a mock Request for registry auth tests."""
    headers = {}
    if client_cert is not None:
        headers["X-Client-Cert"] = client_cert
    if real_ip is not None:
        headers["X-Real-IP"] = real_ip
    if proxy_auth is not None:
        headers["X-Registry-Proxy-Auth"] = proxy_auth
    if hotkey is not None:
        headers["X-Chutes-Hotkey"] = hotkey
    if signature is not None:
        headers["X-Chutes-Signature"] = signature
    if nonce is not None:
        headers["X-Chutes-Nonce"] = nonce

    request = MagicMock()
    request.headers = headers
    request.client = MagicMock()
    request.client.host = real_ip or "1.2.3.4"
    return request


@pytest.mark.asyncio
async def test_registry_mtls_valid_leaf():
    """Leaf cert signed by registered CA succeeds."""

    ca_key = _gen_rsa_key()
    ca_cert = _make_ca_cert(ca_key)
    leaf_key = _gen_rsa_key()
    leaf_cert = _make_leaf_cert(leaf_key, ca_key, ca_cert)

    leaf_pem_encoded = url_quote(_cert_pem(leaf_cert))

    mock_server = MagicMock()
    mock_server.vm_root_ca_cert = _cert_pem(ca_cert)
    mock_server.name = "vm1"
    mock_server.version = "1.4.0"

    request = _make_registry_request(client_cert=leaf_pem_encoded, real_ip="10.0.0.1")
    db = AsyncMock()

    with (
        patch("api.registry.router.settings") as mock_settings,
        patch("api.registry.router.lookup_server_by_ip", return_value=mock_server),
        patch("api.registry.router.verify_server_cert"),
    ):
        mock_settings.registry_proxy_secret = None
        mock_settings.registry_mtls_min_version = "1.4.0"
        result = await registry_auth(request=request, db=db, client_cert=leaf_cert)

    assert result == {"authenticated": True}


@pytest.mark.asyncio
async def test_registry_mtls_wrong_ca():
    """Leaf signed by a different CA is rejected with 403."""

    ca_key = _gen_rsa_key()
    ca_cert = _make_ca_cert(ca_key)
    other_ca_key = _gen_rsa_key()
    other_ca_cert = _make_ca_cert(other_ca_key, subject_cn="other-ca")
    leaf_key = _gen_rsa_key()
    # Leaf signed by different CA
    leaf_cert = _make_leaf_cert(leaf_key, other_ca_key, other_ca_cert)

    leaf_pem_encoded = url_quote(_cert_pem(leaf_cert))

    mock_server = MagicMock()
    mock_server.vm_root_ca_cert = _cert_pem(ca_cert)
    mock_server.name = "vm1"
    mock_server.version = "1.4.0"

    request = _make_registry_request(client_cert=leaf_pem_encoded, real_ip="10.0.0.1")
    db = AsyncMock()

    with (
        patch("api.registry.router.settings") as mock_settings,
        patch("api.registry.router.lookup_server_by_ip", return_value=mock_server),
        patch(
            "api.registry.router.verify_server_cert",
            side_effect=InvalidClientCertError(detail="bad sig"),
        ),
    ):
        mock_settings.registry_proxy_secret = None
        mock_settings.registry_mtls_min_version = "1.4.0"
        with pytest.raises(HTTPException) as exc_info:
            await registry_auth(request=request, db=db, client_cert=leaf_cert)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_registry_legacy_no_ca_registered():
    """No vm_root_ca_cert in DB; valid legacy miner auth succeeds."""

    request = _make_registry_request(
        hotkey="5FakeHotkey",
        signature="deadbeef",
        nonce="123456",
    )
    db = AsyncMock()

    with (
        patch("api.registry.router.settings") as mock_settings,
        patch("api.registry.router.lookup_server_by_ip", return_value=None),
        patch("api.registry.router._legacy_registry_auth", new_callable=AsyncMock) as mock_legacy,
    ):
        mock_settings.registry_proxy_secret = None
        mock_settings.registry_mtls_min_version = "1.4.0"
        result = await registry_auth(request=request, db=db, client_cert=None)

    mock_legacy.assert_awaited_once()
    assert result == {"authenticated": True}


@pytest.mark.asyncio
async def test_registry_legacy_no_cert_ca_registered():
    """CA registered in DB but no client cert presented; legacy auth still attempted."""

    mock_server = MagicMock()
    mock_server.vm_root_ca_cert = "some-ca-cert-pem"
    mock_server.name = "vm1"
    mock_server.version = None  # never attested a version -> not subject to the mTLS version gate

    # No X-Client-Cert header, but server has a CA registered
    request = _make_registry_request(real_ip="10.0.0.1")
    db = AsyncMock()

    with (
        patch("api.registry.router.settings") as mock_settings,
        patch("api.registry.router.lookup_server_by_ip", return_value=mock_server),
        patch(
            "api.registry.router._legacy_registry_auth",
            new_callable=AsyncMock,
            side_effect=HTTPException(status_code=401, detail="Invalid BT Auth."),
        ),
    ):
        mock_settings.registry_proxy_secret = None
        mock_settings.registry_mtls_min_version = "1.4.0"
        with pytest.raises(HTTPException) as exc_info:
            await registry_auth(request=request, db=db, client_cert=None)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_registry_kill_switch_forces_mtls():
    """min_version '0.0.0' forces every attested VM onto mTLS; legacy auth never runs."""

    mock_server = MagicMock()
    mock_server.vm_root_ca_cert = None
    mock_server.name = "vm1"
    mock_server.version = "1.0.0"  # old VM, but the kill switch still forces mTLS

    request = _make_registry_request(
        hotkey="5FakeHotkey", signature="deadbeef", nonce="123456", real_ip="10.0.0.1"
    )
    db = AsyncMock()

    with (
        patch("api.registry.router.settings") as mock_settings,
        patch("api.registry.router.lookup_server_by_ip", return_value=mock_server),
        patch("api.registry.router._legacy_registry_auth", new_callable=AsyncMock) as mock_legacy,
    ):
        mock_settings.registry_proxy_secret = None
        mock_settings.registry_mtls_min_version = "0.0.0"
        with pytest.raises(HTTPException) as exc_info:
            await registry_auth(request=request, db=db, client_cert=None)

    assert exc_info.value.status_code == 403
    mock_legacy.assert_not_awaited()


@pytest.mark.asyncio
async def test_registry_version_gate_rejects_new_vm_without_mtls():
    """A VM attested at >= registry_mtls_min_version may not fall back to legacy auth."""

    mock_server = MagicMock()
    mock_server.vm_root_ca_cert = None  # CA not (yet) registered
    mock_server.name = "vm1"
    mock_server.version = "1.4.0"

    request = _make_registry_request(
        hotkey="5FakeHotkey", signature="deadbeef", nonce="123456", real_ip="10.0.0.1"
    )
    db = AsyncMock()

    with (
        patch("api.registry.router.settings") as mock_settings,
        patch("api.registry.router.lookup_server_by_ip", return_value=mock_server),
        patch("api.registry.router._legacy_registry_auth", new_callable=AsyncMock) as mock_legacy,
    ):
        mock_settings.registry_proxy_secret = None
        mock_settings.registry_mtls_min_version = "1.4.0"
        with pytest.raises(HTTPException) as exc_info:
            await registry_auth(request=request, db=db, client_cert=None)

    assert exc_info.value.status_code == 403
    mock_legacy.assert_not_awaited()


@pytest.mark.asyncio
async def test_registry_version_gate_allows_old_vm():
    """A VM attested below registry_mtls_min_version keeps using legacy auth."""

    mock_server = MagicMock()
    mock_server.vm_root_ca_cert = None
    mock_server.name = "vm1"
    mock_server.version = "1.3.0"

    request = _make_registry_request(
        hotkey="5FakeHotkey", signature="deadbeef", nonce="123456", real_ip="10.0.0.1"
    )
    db = AsyncMock()

    with (
        patch("api.registry.router.settings") as mock_settings,
        patch("api.registry.router.lookup_server_by_ip", return_value=mock_server),
        patch("api.registry.router._legacy_registry_auth", new_callable=AsyncMock) as mock_legacy,
    ):
        mock_settings.registry_proxy_secret = None
        mock_settings.registry_mtls_min_version = "1.4.0"
        result = await registry_auth(request=request, db=db)

    mock_legacy.assert_awaited_once()
    assert result == {"authenticated": True}


@pytest.mark.asyncio
async def test_registry_mtls_succeeds_under_kill_switch():
    """With the kill switch on (min_version '0.0.0'), a valid mTLS pull still succeeds."""

    ca_key = _gen_rsa_key()
    ca_cert = _make_ca_cert(ca_key)
    leaf_key = _gen_rsa_key()
    leaf_cert = _make_leaf_cert(leaf_key, ca_key, ca_cert)

    leaf_pem_encoded = url_quote(_cert_pem(leaf_cert))

    mock_server = MagicMock()
    mock_server.vm_root_ca_cert = _cert_pem(ca_cert)
    mock_server.name = "vm1"
    mock_server.version = "1.0.0"

    request = _make_registry_request(client_cert=leaf_pem_encoded, real_ip="10.0.0.1")
    db = AsyncMock()

    with (
        patch("api.registry.router.settings") as mock_settings,
        patch("api.registry.router.lookup_server_by_ip", return_value=mock_server),
        patch("api.registry.router.verify_server_cert"),
    ):
        mock_settings.registry_proxy_secret = None
        mock_settings.registry_mtls_min_version = "0.0.0"
        result = await registry_auth(request=request, db=db, client_cert=leaf_cert)

    assert result == {"authenticated": True}
