"""Regression tests for public chutes whose implementation remains private."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from api.chute.response import ChuteResponse, MinimalChuteResponse
from api.chute.schemas import Chute, ChuteArgs, ChuteUpdateArgs, NodeSelector
from api.constants import INTEGRATED_SUBNETS, is_chute_source_public
from api.image.schemas import Image
from api.user.schemas import User

with patch("ctypes.CDLL", return_value=MagicMock()):
    from api.chute.router import (
        SOURCE_NOT_PUBLIC_PLACEHOLDER,
        can_view_chute_source,
        get_chute_code,
    )
    from api.miner.router import model_to_dict


SECRET_CODE = "print('TOP_SECRET_SOURCE')"


def _minimal_chute_payload():
    return {
        "chute_id": "chute-1",
        "name": "source-visibility-test",
        "public": True,
        "version": "version-1",
        "slug": "source-visibility-test",
        "chutes_version": "0.5.5",
        "preemptible": True,
        "image": {
            "image_id": "image-1",
            "name": "source-visibility-image",
            "tag": "latest",
            "public": True,
            "chutes_version": "0.5.5",
            "patch_version": "initial",
        },
    }


def _access_chute(*, name="chronoseek-source-visibility-test", public=True):
    return SimpleNamespace(
        chute_id="chute-1",
        user_id="owner-id",
        name=name,
        public=public,
        code=SECRET_CODE,
    )


def _user(user_id):
    return SimpleNamespace(user_id=user_id)


def _db_returning(value):
    result = SimpleNamespace()
    result.unique = lambda: result
    result.scalar_one_or_none = lambda: value
    db = AsyncMock()
    db.execute.return_value = result
    return db


def _miner_chute(*, public=True):
    owner = User(
        user_id="owner-id",
        username="Owner",
        coldkey="coldkey",
        fingerprint_hash="fingerprint",
        permissions_bitmask=0,
    )
    image = Image(
        image_id="image-1",
        user_id=owner.user_id,
        name="Source-Visibility-Image",
        tag="Latest",
        public=False,
        patch_version="initial",
        chutes_version="0.6.11.rc2",
    )
    image.user = owner
    chute = Chute(
        chute_id="chute-1",
        user_id=owner.user_id,
        name="chronoseek-source-visibility-test",
        image_id=image.image_id,
        public=public,
        code=SECRET_CODE,
        filename="chute.py",
        ref_str="chute:chute",
        version="version-1",
        chutes_version="0.6.11.rc2",
        cords=[],
        jobs=[],
        node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16),
    )
    chute.image = image
    return chute


def test_chronoseek_integrated_subnet_configuration():
    assert INTEGRATED_SUBNETS["chronoseek"] == {
        "netuid": 20,
        "model_substring": "chronoseek",
        "max_public_chutes": 3,
        "source_public": False,
    }


def test_source_visibility_is_not_persisted_or_exposed_as_a_chute_field():
    assert "source_public" not in Chute.__table__.columns
    assert "source_public" not in ChuteArgs.model_fields
    assert "source_public" not in ChuteUpdateArgs.model_fields
    assert "source_public" not in ChuteResponse.model_fields
    assert "source_public" not in MinimalChuteResponse.model_fields


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ChronoSeek/model", False),
        ("prefix-chronoseek-suffix", False),
        ("chronoseek-affine-overlap", False),
        ("babelbit/model", True),
        ("ordinary-model", True),
        (None, True),
    ],
)
def test_source_visibility_is_derived_from_integrated_subnet_config(name, expected):
    assert is_chute_source_public(name) is expected


def test_minimal_chute_response_keeps_image_metadata():
    response = MinimalChuteResponse.model_validate(_minimal_chute_payload())

    assert response.image is not None
    assert response.image.image_id == "image-1"


def test_chute_response_still_requires_image_metadata():
    assert ChuteResponse.model_fields["image"].is_required()
    assert MinimalChuteResponse.model_fields["image"].is_required()


@pytest.mark.asyncio
async def test_source_access_allows_fully_public_chute_without_a_user():
    assert await can_view_chute_source(_access_chute(name="ordinary-model"), None) is True


@pytest.mark.asyncio
async def test_source_access_allows_hidden_source_owner():
    assert await can_view_chute_source(_access_chute(), _user("owner-id")) is True


@pytest.mark.asyncio
@patch("api.chute.router.is_shared", new_callable=AsyncMock, return_value=True)
async def test_source_access_allows_explicit_share(_mock_is_shared):
    assert await can_view_chute_source(_access_chute(), _user("shared-user-id")) is True


@pytest.mark.asyncio
@patch("api.chute.router.subnet_role_accessible", return_value=True)
@patch("api.chute.router.is_shared", new_callable=AsyncMock, return_value=False)
async def test_source_access_allows_subnet_admin(_mock_is_shared, _mock_subnet_access):
    assert await can_view_chute_source(_access_chute(), _user("admin-id")) is True


@pytest.mark.asyncio
@patch("api.chute.router.subnet_role_accessible", return_value=False)
@patch("api.chute.router.is_shared", new_callable=AsyncMock, return_value=False)
async def test_source_access_denies_hidden_source_outsider(_mock_is_shared, _mock_subnet_access):
    assert await can_view_chute_source(_access_chute(), _user("outsider-id")) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("current_user", [None, _user("outsider-id")])
@patch("api.chute.router.subnet_role_accessible", return_value=False)
@patch("api.chute.router.is_shared", new_callable=AsyncMock, return_value=False)
@patch("api.chute.router.get_one", new_callable=AsyncMock)
async def test_get_chute_code_returns_placeholder_to_public_caller(
    mock_get_one,
    _mock_is_shared,
    _mock_subnet_access,
    current_user,
):
    mock_get_one.return_value = _access_chute()
    db = _db_returning(mock_get_one.return_value)

    response = await get_chute_code(
        "chute-1",
        db=db,
        current_user=current_user,
    )

    assert response.status_code == 200
    assert response.body.decode() == SOURCE_NOT_PUBLIC_PLACEHOLDER
    assert SECRET_CODE not in response.body.decode()


@pytest.mark.asyncio
@patch("api.chute.router.subnet_role_accessible", return_value=False)
@patch("api.chute.router.is_shared", new_callable=AsyncMock, return_value=False)
@patch("api.chute.router.get_one", new_callable=AsyncMock)
async def test_get_chute_code_keeps_private_chute_hidden(
    mock_get_one,
    _mock_is_shared,
    _mock_subnet_access,
):
    mock_get_one.return_value = _access_chute(public=False)
    db = _db_returning(mock_get_one.return_value)

    with pytest.raises(HTTPException) as exc_info:
        await get_chute_code(
            "chute-1",
            db=db,
            current_user=_user("outsider-id"),
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
@patch("api.chute.router.get_one", new_callable=AsyncMock)
async def test_get_chute_code_returns_hidden_source_to_owner(mock_get_one):
    mock_get_one.return_value = _access_chute()
    db = _db_returning(mock_get_one.return_value)

    response = await get_chute_code(
        "chute-1",
        db=db,
        current_user=_user("owner-id"),
    )

    assert response.body == SECRET_CODE.encode()
    assert response.media_type == "text/plain"


@pytest.mark.asyncio
@patch("api.chute.router.get_one", new_callable=AsyncMock)
async def test_get_chute_code_keeps_ordinary_public_source_visible(mock_get_one):
    mock_get_one.return_value = _access_chute(name="ordinary-model")
    db = _db_returning(mock_get_one.return_value)

    response = await get_chute_code(
        "chute-1",
        db=db,
        current_user=None,
    )

    assert response.body == SECRET_CODE.encode()
    assert response.media_type == "text/plain"


@pytest.mark.asyncio
@patch("api.miner.router.calculate_effective_compute_multiplier", new_callable=AsyncMock)
async def test_miner_always_replaces_public_code_but_keeps_operational_image(mock_multiplier):
    mock_multiplier.return_value = {
        "effective_compute_multiplier": 1.0,
        "compute_multiplier_factors": {},
        "bounty": None,
    }

    data = await model_to_dict(_miner_chute())

    assert SECRET_CODE not in data["code"]
    assert "legacy placeholder" in data["code"]
    assert data["image"] == "owner/source-visibility-image:latest"


@pytest.mark.asyncio
@patch("api.miner.router.calculate_effective_compute_multiplier", new_callable=AsyncMock)
async def test_miner_always_replaces_private_code(mock_multiplier):
    mock_multiplier.return_value = {
        "effective_compute_multiplier": 1.0,
        "compute_multiplier_factors": {},
        "bounty": None,
    }

    data = await model_to_dict(_miner_chute(public=False))

    assert SECRET_CODE not in data["code"]
    assert "legacy placeholder" in data["code"]
