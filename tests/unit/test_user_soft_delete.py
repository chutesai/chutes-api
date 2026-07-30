"""Unit tests for restorable user soft-delete.

Covers the bounty scale-up guard and the API-key cache invalidation that keep a soft-deleted
account inert. The auth-path rejection and the DB-heavy delete/restore endpoints are exercised
by integration tests.
"""

import api.database.orms  # noqa: F401  # register all ORM models so mappers configure

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.bounty import util as bounty_util
from api.api_key import util as api_key_util


# ---------------------------------------------------------------------------
# Bounty scale-up guard
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bounty_blocked_when_owner_deleted():
    with (
        patch.object(bounty_util, "is_chute_disabled", new=AsyncMock(return_value=False)),
        patch.object(bounty_util, "is_owner_deleted", new=AsyncMock(return_value=True)),
    ):
        assert await bounty_util.create_bounty_if_not_exists("chute-1") is False


@pytest.mark.asyncio
async def test_is_owner_deleted_reads_db():
    # is_owner_deleted is DB-backed (reads users.deleted_at) so it can never drift from the
    # source of truth. db_scalar returns the exists() boolean.
    with patch.object(bounty_util, "db_scalar", new=AsyncMock(return_value=True)):
        assert await bounty_util.is_owner_deleted("chute-9") is True
    with patch.object(bounty_util, "db_scalar", new=AsyncMock(return_value=False)):
        assert await bounty_util.is_owner_deleted("chute-9") is False


# ---------------------------------------------------------------------------
# API-key cache invalidation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_invalidate_user_api_key_caches_clears_each_key():
    result = MagicMock()
    result.scalars.return_value.all.return_value = ["k1", "k2"]
    session = AsyncMock()
    session.execute = AsyncMock(return_value=result)

    redis = AsyncMock()
    with (
        patch.object(api_key_util, "settings", SimpleNamespace(redis_client=redis)),
        patch.object(api_key_util._load_key, "cache_invalidate") as invalidate,
    ):
        await api_key_util.invalidate_user_api_key_caches(session, "user-1")

    redis.delete.assert_any_await("akey:k1")
    redis.delete.assert_any_await("akey:k2")
    assert invalidate.call_count == 2
