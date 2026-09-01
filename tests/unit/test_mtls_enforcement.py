"""
Unit tests for attestation proxy provenance enforcement:
  - require_mtls_proxy() (either) / require_attestation_proxy() (1.3.x proxy only) /
    require_cvm_proxy() (1.4.0+ cvm proxy only) /
    gate_legacy_attestation() (version-gated permissive)
  - _get_client_certificate() two-secret proxy guard
  - require_proxy_secret / require_registry_proxy_secret
  - extract_client_cert / extract_optional_client_cert
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from api.server.util import (
    require_mtls_proxy,
    require_attestation_proxy,
    require_cvm_proxy,
    gate_legacy_attestation,
    _get_client_certificate,
)
from api.constants import ATTESTATION_PROXY_AUTH_HEADER, CVM_PROXY_AUTH_HEADER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(mtls_auth=None, cvm_auth=None, client_cert=None):
    """Build a minimal mock Request carrying the given proxy headers."""
    headers = {}
    if mtls_auth is not None:
        headers[ATTESTATION_PROXY_AUTH_HEADER] = mtls_auth
    if cvm_auth is not None:
        headers[CVM_PROXY_AUTH_HEADER] = cvm_auth
    if client_cert is not None:
        headers["X-Client-Cert"] = client_cert

    request = MagicMock()
    request.headers = headers
    request.url.path = "/servers/boot/attestation"
    return request


def _configure(mock_settings, *, mtls=None, cvm=None):
    """Set BOTH proxy secrets on the patched settings.

    Explicit because an unset MagicMock attribute is a truthy Mock, which would make the
    provenance match logic (secret.encode()) blow up or silently 'enable' a phantom secret.
    """
    mock_settings.attestation_proxy_secret = mtls
    mock_settings.cvm_proxy_secret = cvm


# ---------------------------------------------------------------------------
# require_mtls_proxy — boot/attestation only (either proxy: every VM boots)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_attestation_accepts_attestation_secret(mock_settings):
    _configure(mock_settings, mtls="att-secret", cvm="cvm-secret")
    await require_mtls_proxy()(_make_request(mtls_auth="att-secret"))


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_attestation_accepts_cvm_secret(mock_settings):
    _configure(mock_settings, mtls="att-secret", cvm="cvm-secret")
    await require_mtls_proxy()(_make_request(cvm_auth="cvm-secret"))


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_attestation_only_attestation_configured_still_works(mock_settings):
    # During early rollout only the attestation secret may be set; legacy VMs still pass.
    _configure(mock_settings, mtls="att-secret", cvm=None)
    await require_mtls_proxy()(_make_request(mtls_auth="att-secret"))


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_attestation_no_valid_secret_rejected(mock_settings):
    _configure(mock_settings, mtls="att-secret", cvm="cvm-secret")
    with pytest.raises(HTTPException) as exc:
        await require_mtls_proxy()(_make_request(mtls_auth="wrong"))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_attestation_no_secret_configured_fails_closed(mock_settings):
    _configure(mock_settings, mtls=None, cvm=None)
    with pytest.raises(HTTPException) as exc:
        await require_mtls_proxy()(_make_request(mtls_auth="x"))
    assert exc.value.status_code == 503


# ---------------------------------------------------------------------------
# require_cvm_proxy — provision, provision/confirm (cvm proxy only)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_cvm_accepts_cvm_secret(mock_settings):
    _configure(mock_settings, mtls="att-secret", cvm="cvm-secret")
    await require_cvm_proxy()(_make_request(cvm_auth="cvm-secret"))


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_cvm_rejects_attestation_secret(mock_settings):
    # A legacy (attestation-proxy) request must NOT reach a cvm-only endpoint.
    _configure(mock_settings, mtls="att-secret", cvm="cvm-secret")
    with pytest.raises(HTTPException) as exc:
        await require_cvm_proxy()(_make_request(mtls_auth="att-secret"))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_cvm_unconfigured_fails_closed(mock_settings):
    _configure(mock_settings, mtls="att-secret", cvm=None)
    with pytest.raises(HTTPException) as exc:
        await require_cvm_proxy()(_make_request(cvm_auth="x"))
    assert exc.value.status_code == 503


# ---------------------------------------------------------------------------
# require_attestation_proxy — luks/attest (1.3.x proxy only)
#
# The mirror of require_cvm_proxy. /luks/attest is superseded by /provision and no current guest
# calls it, so accepting the cvm proxy there would leave a second, weaker way into storage-secret
# issuance that nothing legitimately uses. Each generation gets exactly one route.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_attestation_proxy_accepts_its_own_secret(mock_settings):
    _configure(mock_settings, mtls="att-secret", cvm="cvm-secret")
    await require_attestation_proxy()(_make_request(mtls_auth="att-secret"))


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_attestation_proxy_rejects_cvm_secret(mock_settings):
    """A 1.4.0+ VM must not reach the legacy storage route -- it has /provision, which requires a
    proven hotkey unconditionally."""
    _configure(mock_settings, mtls="att-secret", cvm="cvm-secret")
    with pytest.raises(HTTPException) as exc:
        await require_attestation_proxy()(_make_request(cvm_auth="cvm-secret"))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_attestation_proxy_rejects_unproxied_request(mock_settings):
    _configure(mock_settings, mtls="att-secret", cvm="cvm-secret")
    with pytest.raises(HTTPException) as exc:
        await require_attestation_proxy()(_make_request())
    assert exc.value.status_code == 403


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_attestation_proxy_unconfigured_fails_closed(mock_settings):
    """No attestation-proxy secret also means the legacy path is retired and the route can go."""
    _configure(mock_settings, mtls=None, cvm="cvm-secret")
    with pytest.raises(HTTPException) as exc:
        await require_attestation_proxy()(_make_request(mtls_auth="x"))
    assert exc.value.status_code == 503


# ---------------------------------------------------------------------------
# gate_legacy_attestation — nonce, luks/confirm (version-gated permissive)
# ---------------------------------------------------------------------------


def _mock_db(server):
    """A db session whose execute(...).scalar_one_or_none() returns `server`."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = server
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    return db


def _server(version):
    s = MagicMock()
    s.version = version
    s.name = "vm-test"
    return s


def _legacy_request(cvm_auth=None, client_ip="1.2.3.4"):
    req = _make_request(cvm_auth=cvm_auth)
    req.state.client_ip = client_ip
    return req


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_gate_allows_via_cvm_without_db(mock_settings):
    # cvm secret present -> allow and short-circuit before any DB lookup.
    _configure(mock_settings, mtls="att-secret", cvm="cvm-secret")
    mock_settings.tee_mtls_min_version = "1.4.0"
    db = _mock_db(_server("1.4.0"))  # would 403 if consulted
    await gate_legacy_attestation()(_legacy_request(cvm_auth="cvm-secret"), db)
    db.execute.assert_not_called()


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_gate_rejects_upgraded_server_on_legacy_path(mock_settings):
    _configure(mock_settings, mtls="att-secret", cvm="cvm-secret")
    mock_settings.tee_mtls_min_version = "1.4.0"
    with pytest.raises(HTTPException) as exc:
        await gate_legacy_attestation()(_legacy_request(), _mock_db(_server("1.4.0")))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_gate_allows_old_server(mock_settings):
    _configure(mock_settings, mtls="att-secret", cvm="cvm-secret")
    mock_settings.tee_mtls_min_version = "1.4.0"
    await gate_legacy_attestation()(_legacy_request(), _mock_db(_server("1.3.1")))


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_gate_allows_unknown_ip(mock_settings):
    _configure(mock_settings, mtls="att-secret", cvm="cvm-secret")
    mock_settings.tee_mtls_min_version = "1.4.0"
    await gate_legacy_attestation()(_legacy_request(), _mock_db(None))


# ---------------------------------------------------------------------------
# _get_client_certificate — two-secret proxy guard
# ---------------------------------------------------------------------------


def _make_pem_cert():
    """Generate a minimal self-signed cert PEM for parsing tests."""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    import datetime
    from urllib.parse import quote

    key = ec.generate_private_key(ec.SECP256R1())
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    pem = cert.public_bytes(serialization.Encoding.PEM).decode()
    return quote(pem)


@patch("api.server.util.settings")
def test_get_client_cert_no_secret_reads_header(mock_settings):
    _configure(mock_settings, mtls=None, cvm=None)
    request = _make_request(client_cert=_make_pem_cert())
    assert _get_client_certificate(request) is not None


@patch("api.server.util.settings")
def test_get_client_cert_valid_mtls_secret_reads_header(mock_settings):
    _configure(mock_settings, mtls="mysecret", cvm=None)
    request = _make_request(mtls_auth="mysecret", client_cert=_make_pem_cert())
    assert _get_client_certificate(request) is not None


@patch("api.server.util.settings")
def test_get_client_cert_valid_cvm_secret_reads_header(mock_settings):
    _configure(mock_settings, mtls=None, cvm="cvm-secret")
    request = _make_request(cvm_auth="cvm-secret", client_cert=_make_pem_cert())
    assert _get_client_certificate(request) is not None


@patch("api.server.util.settings")
def test_get_client_cert_wrong_secret_raises(mock_settings):
    from api.server.exceptions import NoClientCertError

    _configure(mock_settings, mtls="mysecret", cvm="cvm-secret")
    request = _make_request(mtls_auth="wrong", client_cert=_make_pem_cert())
    with pytest.raises(NoClientCertError):
        _get_client_certificate(request)


@patch("api.server.util.settings")
def test_get_client_cert_missing_secret_header_raises(mock_settings):
    from api.server.exceptions import NoClientCertError

    _configure(mock_settings, mtls="mysecret", cvm="cvm-secret")
    request = _make_request(client_cert=_make_pem_cert())
    with pytest.raises(NoClientCertError):
        _get_client_certificate(request)


@patch("api.server.util.settings")
def test_get_client_cert_no_cert_header_raises(mock_settings):
    from api.server.exceptions import NoClientCertError

    _configure(mock_settings, mtls=None, cvm=None)
    with pytest.raises(NoClientCertError):
        _get_client_certificate(_make_request())


# ---------------------------------------------------------------------------
# require_proxy_secret — parameterized proxy-trust guard (unchanged)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_require_proxy_secret_disabled_when_unset():
    from api.server.util import require_proxy_secret

    dep = require_proxy_secret(None, "X-Registry-Proxy-Auth")
    await dep(_make_request())


@pytest.mark.asyncio
async def test_require_proxy_secret_correct_passes():
    from api.server.util import require_proxy_secret

    dep = require_proxy_secret("s3cr3t", "X-Registry-Proxy-Auth")
    request = MagicMock()
    request.headers = {"X-Registry-Proxy-Auth": "s3cr3t"}
    await dep(request)


@pytest.mark.asyncio
async def test_require_proxy_secret_wrong_rejected():
    from api.server.util import require_proxy_secret

    dep = require_proxy_secret("s3cr3t", "X-Registry-Proxy-Auth")
    request = MagicMock()
    request.headers = {"X-Registry-Proxy-Auth": "nope"}
    with pytest.raises(HTTPException) as exc:
        await dep(request)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_require_proxy_secret_missing_header_rejected():
    from api.server.util import require_proxy_secret

    dep = require_proxy_secret("s3cr3t", "X-Registry-Proxy-Auth")
    request = MagicMock()
    request.headers = {}
    with pytest.raises(HTTPException) as exc:
        await dep(request)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_require_registry_proxy_secret_binds_setting_and_header(mock_settings):
    from api.constants import REGISTRY_PROXY_AUTH_HEADER
    from api.server.util import require_registry_proxy_secret

    mock_settings.registry_proxy_secret = "reg-secret"
    dep = require_registry_proxy_secret()

    ok = MagicMock()
    ok.headers = {REGISTRY_PROXY_AUTH_HEADER: "reg-secret"}
    await dep(ok)

    bad = MagicMock()
    bad.headers = {REGISTRY_PROXY_AUTH_HEADER: "wrong"}
    with pytest.raises(HTTPException) as exc:
        await dep(bad)
    assert exc.value.status_code == 403


# ---------------------------------------------------------------------------
# extract_optional_client_cert / extract_client_cert — typed extraction deps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_optional_client_cert_absent_returns_none():
    from api.server.util import extract_optional_client_cert

    dep = extract_optional_client_cert()
    assert await dep(_make_request()) is None


@pytest.mark.asyncio
async def test_extract_optional_client_cert_present_returns_certificate():
    from cryptography.x509 import Certificate
    from api.server.util import extract_optional_client_cert

    dep = extract_optional_client_cert()
    request = _make_request(client_cert=_make_pem_cert())
    assert isinstance(await dep(request), Certificate)


@pytest.mark.asyncio
async def test_extract_optional_client_cert_malformed_raises_400():
    from api.server.util import extract_optional_client_cert

    dep = extract_optional_client_cert()
    request = _make_request(client_cert="not-a-cert")
    with pytest.raises(HTTPException) as exc:
        await dep(request)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_extract_client_cert_present_returns_certificate(mock_settings):
    from cryptography.x509 import Certificate
    from api.server.util import extract_client_cert

    _configure(mock_settings, mtls=None, cvm=None)
    dep = extract_client_cert()
    request = _make_request(client_cert=_make_pem_cert())
    assert isinstance(await dep(request), Certificate)


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_extract_client_cert_absent_raises(mock_settings):
    from api.server.exceptions import NoClientCertError
    from api.server.util import extract_client_cert

    _configure(mock_settings, mtls=None, cvm=None)
    dep = extract_client_cert()
    with pytest.raises(NoClientCertError):
        await dep(_make_request())
