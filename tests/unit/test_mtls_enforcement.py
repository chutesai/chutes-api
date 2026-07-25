"""
Unit tests for mTLS proxy enforcement:
  - require_mtls_proxy_secret() FastAPI dependency
  - _get_client_certificate() proxy-secret guard
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException

from api.server.util import require_mtls_proxy_secret, _get_client_certificate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(host: str, proxy_auth: str | None = None, client_cert: str | None = None):
    """Build a minimal mock Request with the given headers."""
    headers = {"host": host}
    if proxy_auth is not None:
        headers["X-Mtls-Proxy-Auth"] = proxy_auth
    if client_cert is not None:
        headers["X-Client-Cert"] = client_cert

    request = MagicMock()
    request.headers = headers
    request.url.path = "/servers/nonce"
    return request


# ---------------------------------------------------------------------------
# require_mtls_proxy_secret — proxy-secret enforcement (fail-closed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_correct_secret_passes(mock_settings):
    mock_settings.mtls_proxy_secret = "supersecret"

    checker = require_mtls_proxy_secret()
    request = _make_request("any-host", proxy_auth="supersecret")
    # Should not raise
    await checker(request)


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_host_is_ignored(mock_settings):
    """Host header is no longer inspected; only the proxy secret matters."""
    mock_settings.mtls_proxy_secret = "supersecret"

    checker = require_mtls_proxy_secret()
    await checker(_make_request("api.chutes.ai", proxy_auth="supersecret"))
    await checker(_make_request("", proxy_auth="supersecret"))


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_wrong_secret_is_rejected(mock_settings):
    mock_settings.mtls_proxy_secret = "supersecret"

    checker = require_mtls_proxy_secret()
    request = _make_request("any-host", proxy_auth="wrongsecret")
    with pytest.raises(HTTPException) as exc_info:
        await checker(request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_missing_secret_header_is_rejected(mock_settings):
    mock_settings.mtls_proxy_secret = "supersecret"

    checker = require_mtls_proxy_secret()
    # No X-Mtls-Proxy-Auth header
    request = _make_request("any-host")
    with pytest.raises(HTTPException) as exc_info:
        await checker(request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_unconfigured_secret_fails_closed(mock_settings):
    """When MTLS_PROXY_SECRET is not configured, the endpoint refuses to serve (503)."""
    mock_settings.mtls_proxy_secret = None

    checker = require_mtls_proxy_secret()
    request = _make_request("any-host", proxy_auth="anything")
    with pytest.raises(HTTPException) as exc_info:
        await checker(request)
    assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# _get_client_certificate — proxy-secret guard
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
    mock_settings.mtls_proxy_secret = None

    pem = _make_pem_cert()
    request = _make_request("tdx-attestation.chutes.ai", client_cert=pem)
    cert = _get_client_certificate(request)
    assert cert is not None


@patch("api.server.util.settings")
def test_get_client_cert_correct_secret_reads_header(mock_settings):
    mock_settings.mtls_proxy_secret = "mysecret"

    pem = _make_pem_cert()
    request = _make_request("tdx-attestation.chutes.ai", proxy_auth="mysecret", client_cert=pem)
    cert = _get_client_certificate(request)
    assert cert is not None


@patch("api.server.util.settings")
def test_get_client_cert_wrong_secret_raises(mock_settings):
    from api.server.exceptions import NoClientCertError

    mock_settings.mtls_proxy_secret = "mysecret"

    pem = _make_pem_cert()
    request = _make_request("tdx-attestation.chutes.ai", proxy_auth="wrongsecret", client_cert=pem)
    with pytest.raises(NoClientCertError):
        _get_client_certificate(request)


@patch("api.server.util.settings")
def test_get_client_cert_missing_secret_header_raises(mock_settings):
    from api.server.exceptions import NoClientCertError

    mock_settings.mtls_proxy_secret = "mysecret"

    pem = _make_pem_cert()
    # No X-Mtls-Proxy-Auth header
    request = _make_request("tdx-attestation.chutes.ai", client_cert=pem)
    with pytest.raises(NoClientCertError):
        _get_client_certificate(request)


@patch("api.server.util.settings")
def test_get_client_cert_no_cert_header_raises(mock_settings):
    from api.server.exceptions import NoClientCertError

    mock_settings.mtls_proxy_secret = None

    request = _make_request("tdx-attestation.chutes.ai")
    with pytest.raises(NoClientCertError):
        _get_client_certificate(request)


# ---------------------------------------------------------------------------
# require_proxy_secret — parameterized proxy-trust guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_require_proxy_secret_disabled_when_unset():
    """No secret configured -> the guard is a no-op regardless of headers."""
    from api.server.util import require_proxy_secret

    dep = require_proxy_secret(None, "X-Registry-Proxy-Auth")
    await dep(_make_request("registry.chutes.ai"))


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
    """The wrapper reads REGISTRY_PROXY_SECRET and checks the registry proxy header."""
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
# extract_optional_client_cert / extract_client_cert — typed extraction dependencies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_optional_client_cert_absent_returns_none():
    from api.server.util import extract_optional_client_cert

    dep = extract_optional_client_cert()
    assert await dep(_make_request("registry.chutes.ai")) is None


@pytest.mark.asyncio
async def test_extract_optional_client_cert_present_returns_certificate():
    from cryptography.x509 import Certificate
    from api.server.util import extract_optional_client_cert

    dep = extract_optional_client_cert()
    request = _make_request("registry.chutes.ai", client_cert=_make_pem_cert())
    assert isinstance(await dep(request), Certificate)


@pytest.mark.asyncio
async def test_extract_optional_client_cert_malformed_raises_400():
    from api.server.util import extract_optional_client_cert

    dep = extract_optional_client_cert()
    request = _make_request("registry.chutes.ai", client_cert="not-a-cert")
    with pytest.raises(HTTPException) as exc:
        await dep(request)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_extract_client_cert_present_returns_certificate(mock_settings):
    from cryptography.x509 import Certificate
    from api.server.util import extract_client_cert

    mock_settings.mtls_proxy_secret = None
    dep = extract_client_cert()
    request = _make_request("tdx-attestation.chutes.ai", client_cert=_make_pem_cert())
    assert isinstance(await dep(request), Certificate)


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_extract_client_cert_absent_raises(mock_settings):
    from api.server.exceptions import NoClientCertError
    from api.server.util import extract_client_cert

    mock_settings.mtls_proxy_secret = None
    dep = extract_client_cert()
    with pytest.raises(NoClientCertError):
        await dep(_make_request("tdx-attestation.chutes.ai"))
