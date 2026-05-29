"""
Unit tests for mTLS domain enforcement:
  - require_mtls_domain() FastAPI dependency
  - _get_client_certificate() proxy-secret guard
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from api.server.util import require_mtls_domain, _get_client_certificate


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


def _make_settings(*, mtls_domain: str = "tdx-attestation.chutes.ai", mtls_proxy_secret=None):
    s = MagicMock()
    s.mtls_domain = mtls_domain
    s.mtls_proxy_secret = mtls_proxy_secret
    return s


# ---------------------------------------------------------------------------
# require_mtls_domain — host-header check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_correct_host_no_secret_passes(mock_settings):
    mock_settings.mtls_domain = "tdx-attestation.chutes.ai"
    mock_settings.mtls_proxy_secret = None

    checker = require_mtls_domain()
    request = _make_request("tdx-attestation.chutes.ai")
    # Should not raise
    await checker(request)


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_wrong_host_is_rejected(mock_settings):
    mock_settings.mtls_domain = "tdx-attestation.chutes.ai"
    mock_settings.mtls_proxy_secret = None

    checker = require_mtls_domain()
    request = _make_request("api.chutes.ai")
    with pytest.raises(HTTPException) as exc_info:
        await checker(request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_host_check_is_case_insensitive(mock_settings):
    mock_settings.mtls_domain = "tdx-attestation.chutes.ai"
    mock_settings.mtls_proxy_secret = None

    checker = require_mtls_domain()
    request = _make_request("TDX-ATTESTATION.CHUTES.AI")
    await checker(request)


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_host_with_port_is_allowed(mock_settings):
    mock_settings.mtls_domain = "tdx-attestation.chutes.ai"
    mock_settings.mtls_proxy_secret = None

    checker = require_mtls_domain()
    request = _make_request("tdx-attestation.chutes.ai:443")
    await checker(request)


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_missing_host_header_is_rejected(mock_settings):
    mock_settings.mtls_domain = "tdx-attestation.chutes.ai"
    mock_settings.mtls_proxy_secret = None

    checker = require_mtls_domain()
    request = _make_request("")
    with pytest.raises(HTTPException) as exc_info:
        await checker(request)
    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# require_mtls_domain — proxy-secret check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_correct_host_and_secret_passes(mock_settings):
    mock_settings.mtls_domain = "tdx-attestation.chutes.ai"
    mock_settings.mtls_proxy_secret = "supersecret"

    checker = require_mtls_domain()
    request = _make_request("tdx-attestation.chutes.ai", proxy_auth="supersecret")
    await checker(request)


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_correct_host_wrong_secret_is_rejected(mock_settings):
    mock_settings.mtls_domain = "tdx-attestation.chutes.ai"
    mock_settings.mtls_proxy_secret = "supersecret"

    checker = require_mtls_domain()
    request = _make_request("tdx-attestation.chutes.ai", proxy_auth="wrongsecret")
    with pytest.raises(HTTPException) as exc_info:
        await checker(request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_correct_host_missing_secret_header_is_rejected(mock_settings):
    mock_settings.mtls_domain = "tdx-attestation.chutes.ai"
    mock_settings.mtls_proxy_secret = "supersecret"

    checker = require_mtls_domain()
    # No X-Mtls-Proxy-Auth header
    request = _make_request("tdx-attestation.chutes.ai")
    with pytest.raises(HTTPException) as exc_info:
        await checker(request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_wrong_host_with_correct_secret_is_still_rejected(mock_settings):
    mock_settings.mtls_domain = "tdx-attestation.chutes.ai"
    mock_settings.mtls_proxy_secret = "supersecret"

    checker = require_mtls_domain()
    request = _make_request("api.chutes.ai", proxy_auth="supersecret")
    with pytest.raises(HTTPException) as exc_info:
        await checker(request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
@patch("api.server.util.settings")
async def test_error_detail_does_not_reveal_which_check_failed(mock_settings):
    """Both host and secret failures must produce the same 403 detail string."""
    mock_settings.mtls_domain = "tdx-attestation.chutes.ai"
    mock_settings.mtls_proxy_secret = "supersecret"

    checker = require_mtls_domain()
    expected_detail = "This endpoint is only accessible via the mTLS attestation domain."

    with pytest.raises(HTTPException) as exc_host:
        await checker(_make_request("api.chutes.ai", proxy_auth="supersecret"))
    with pytest.raises(HTTPException) as exc_secret:
        await checker(_make_request("tdx-attestation.chutes.ai", proxy_auth="bad"))

    assert exc_host.value.detail == expected_detail
    assert exc_secret.value.detail == expected_detail


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
    request = _make_request(
        "tdx-attestation.chutes.ai", proxy_auth="mysecret", client_cert=pem
    )
    cert = _get_client_certificate(request)
    assert cert is not None


@patch("api.server.util.settings")
def test_get_client_cert_wrong_secret_raises(mock_settings):
    from api.server.exceptions import NoClientCertError

    mock_settings.mtls_proxy_secret = "mysecret"

    pem = _make_pem_cert()
    request = _make_request(
        "tdx-attestation.chutes.ai", proxy_auth="wrongsecret", client_cert=pem
    )
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
