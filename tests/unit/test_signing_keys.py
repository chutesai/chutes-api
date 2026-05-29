"""
Unit tests for the public GET /servers/signing-keys endpoint and the
settings.signing_keys_bundle cached_property.
"""

import json
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi import HTTPException

from api.server.router import get_signing_keys


VALID_BUNDLE = {
    "version": 1,
    "keys": {
        "cosign/chutes.pub": "dW50cnVzdGVkIGNvc2lnbg==",
        "cosign/dockerhub.pub": "dW50cnVzdGVkIGRvY2tlcg==",
        "helm-pubkey.gpg": "bXFJTkJHUT09",
    },
    "signatures": {
        "cosign/chutes.pub": "b3dHYndNdk13Q1Uy",
        "cosign/dockerhub.pub": "b3dHYndNdk13Q1Uy",
        "helm-pubkey.gpg": "b3dHYndNdk13Q1Uy",
    },
}


def _write_bundle(bundle: dict) -> Path:
    """Write a bundle dict to a temp file and return its Path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(bundle, fh)
    return Path(path)


def _make_settings(bundle_path: Path):
    """Return a real-ish Settings instance with only signing_keys_bundle_path set."""
    from api.config import Settings

    s = Settings.__new__(Settings)
    object.__setattr__(s, "signing_keys_bundle_path", bundle_path)
    object.__setattr__(s, "_REQUIRED_SIGNING_KEY_NAMES", Settings._REQUIRED_SIGNING_KEY_NAMES)
    return s


# ---------------------------------------------------------------------------
# settings.signing_keys_bundle property tests
# ---------------------------------------------------------------------------


def test_load_valid_bundle_returns_dict():
    bundle_path = _write_bundle(VALID_BUNDLE)
    try:
        s = _make_settings(bundle_path)
        result = s.signing_keys_bundle
        assert result["version"] == 1
        assert "cosign/chutes.pub" in result["keys"]
        assert "helm-pubkey.gpg" in result["signatures"]
    finally:
        bundle_path.unlink()


def test_load_missing_file_returns_none(tmp_path):
    s = _make_settings(tmp_path / "nonexistent.json")
    assert s.signing_keys_bundle is None


def test_load_raises_when_version_not_integer():
    bad = {**VALID_BUNDLE, "version": "not-an-int"}
    bundle_path = _write_bundle(bad)
    try:
        s = _make_settings(bundle_path)
        with pytest.raises(ValueError, match="version"):
            _ = s.signing_keys_bundle
    finally:
        bundle_path.unlink()


def test_load_raises_when_keys_field_missing():
    bad = {k: v for k, v in VALID_BUNDLE.items() if k != "keys"}
    bundle_path = _write_bundle(bad)
    try:
        s = _make_settings(bundle_path)
        with pytest.raises(ValueError, match="'keys'"):
            _ = s.signing_keys_bundle
    finally:
        bundle_path.unlink()


def test_load_raises_when_required_key_missing():
    bad_keys = {k: v for k, v in VALID_BUNDLE["keys"].items() if k != "cosign/chutes.pub"}
    bad_sigs = {k: v for k, v in VALID_BUNDLE["signatures"].items() if k != "cosign/chutes.pub"}
    bundle_path = _write_bundle({**VALID_BUNDLE, "keys": bad_keys, "signatures": bad_sigs})
    try:
        s = _make_settings(bundle_path)
        with pytest.raises(ValueError, match="cosign/chutes.pub"):
            _ = s.signing_keys_bundle
    finally:
        bundle_path.unlink()


def test_load_raises_when_required_signature_missing():
    bad_sigs = {k: v for k, v in VALID_BUNDLE["signatures"].items() if k != "helm-pubkey.gpg"}
    bundle_path = _write_bundle({**VALID_BUNDLE, "signatures": bad_sigs})
    try:
        s = _make_settings(bundle_path)
        with pytest.raises(ValueError, match="helm-pubkey.gpg"):
            _ = s.signing_keys_bundle
    finally:
        bundle_path.unlink()


def test_load_raises_when_value_is_empty_string():
    bad_keys = {**VALID_BUNDLE["keys"], "cosign/chutes.pub": ""}
    bundle_path = _write_bundle({**VALID_BUNDLE, "keys": bad_keys})
    try:
        s = _make_settings(bundle_path)
        with pytest.raises(ValueError, match="non-empty"):
            _ = s.signing_keys_bundle
    finally:
        bundle_path.unlink()


# ---------------------------------------------------------------------------
# GET /signing-keys endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("api.server.router.settings")
async def test_endpoint_returns_bundle_when_loaded(mock_settings):
    mock_settings.signing_keys_bundle = VALID_BUNDLE

    result = await get_signing_keys()

    assert result == VALID_BUNDLE
    assert result["version"] == 1
    assert set(result["keys"].keys()) == {
        "cosign/chutes.pub",
        "cosign/dockerhub.pub",
        "helm-pubkey.gpg",
    }


@pytest.mark.asyncio
@patch("api.server.router.settings")
async def test_endpoint_returns_503_when_bundle_not_available(mock_settings):
    mock_settings.signing_keys_bundle = None

    with pytest.raises(HTTPException) as exc_info:
        await get_signing_keys()

    assert exc_info.value.status_code == 503
