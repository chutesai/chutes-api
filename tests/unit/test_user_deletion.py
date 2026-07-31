"""Unit tests for support-initiated hard user deletion.

Covers the blocking-resource discovery and the API-key cache flush. The full endpoint
(balance guard, 409-on-blockers, the ordered teardown + user delete) is DB-heavy and
exercised by integration tests.
"""

import api.database.orms  # noqa: F401  # register all ORM models so mappers configure

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.api_key import util as api_key_util
from api.user import service as user_service


# ---------------------------------------------------------------------------
# API-key cache invalidation (auth caches populated by _load_key)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_invalidate_api_key_cache_clears_redis_and_lru():
    redis = AsyncMock()
    with (
        patch.object(api_key_util, "settings", SimpleNamespace(redis_client=redis)),
        patch.object(api_key_util._load_key, "cache_invalidate") as invalidate,
    ):
        await api_key_util.invalidate_api_key_cache("k1")
    redis.delete.assert_awaited_once_with("akey:k1")
    invalidate.assert_called_once_with("k1")


# ---------------------------------------------------------------------------
# Blocking-resource discovery (chutes/images/secrets have NO ACTION FKs)
# ---------------------------------------------------------------------------
def _session_returning(*row_lists):
    """AsyncMock session whose successive execute() calls return the given row lists."""
    results = []
    for rows in row_lists:
        r = MagicMock()
        r.all.return_value = rows
        results.append(r)
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=results)
    return session


@pytest.mark.asyncio
async def test_get_user_resources_returns_typed_object():
    session = _session_returning(
        [SimpleNamespace(chute_id="c1", name="chute", version="v1")],
        [SimpleNamespace(image_id="i1", name="img", tag="latest")],
        [SimpleNamespace(secret_id="s1", key="OPENAI_KEY")],
    )

    resources = await user_service.get_user_resources(session, "user-1")

    assert not resources.is_empty
    assert [c.chute_id for c in resources.chutes] == ["c1"]
    assert resources.chutes[0].version == "v1"
    assert resources.images[0].tag == "latest"
    assert resources.secrets[0].key == "OPENAI_KEY"
    assert session.execute.await_count == 3


@pytest.mark.asyncio
async def test_get_user_resources_empty_is_empty():
    session = _session_returning([], [], [])
    resources = await user_service.get_user_resources(session, "user-1")
    assert resources.is_empty


# ---------------------------------------------------------------------------
# check_user_deletable: single source of truth for the delete guards
# ---------------------------------------------------------------------------
def _user(*, permissions_bitmask=0, balance=0.0, effective_balance=0.0, invoicing=False):
    return SimpleNamespace(
        user_id="u1",
        permissions_bitmask=permissions_bitmask,
        balance=balance,
        current_balance=SimpleNamespace(effective_balance=effective_balance),
        has_role=lambda role: invoicing,  # only invoice_billing is consulted
    )


def test_check_user_deletable_allows_clean_user():
    check = user_service.check_user_deletable(
        _user(), user_service.UserResources(), force=False, delete_resources=False
    )
    assert check.allowed


def test_check_user_deletable_privileged_is_hard_block():
    # Not even force overrides a privileged account.
    check = user_service.check_user_deletable(
        _user(permissions_bitmask=1 << 14),
        user_service.UserResources(),
        force=True,
        delete_resources=True,
    )
    assert not check.allowed
    assert check.context["permissions_bitmask"] == (1 << 14)


def test_check_user_deletable_balance_blocks_unless_forced():
    blocked = user_service.check_user_deletable(
        _user(balance=-5.0), user_service.UserResources(), force=False, delete_resources=False
    )
    assert not blocked.allowed and blocked.context["balance"] == -5.0
    forced = user_service.check_user_deletable(
        _user(balance=-5.0), user_service.UserResources(), force=True, delete_resources=False
    )
    assert forced.allowed


def test_check_user_deletable_resources_block_unless_opted_in():
    resources = user_service.UserResources(
        chutes=[user_service.ChuteRef(chute_id="c1", name="chute", version="v1")]
    )
    blocked = user_service.check_user_deletable(
        _user(), resources, force=False, delete_resources=False
    )
    assert not blocked.allowed and blocked.context["chutes"][0]["chute_id"] == "c1"
    opted_in = user_service.check_user_deletable(
        _user(), resources, force=False, delete_resources=True
    )
    assert opted_in.allowed


# ---------------------------------------------------------------------------
# Teardown: delete_user_and_resources
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_delete_user_and_resources_result_payload():
    resources = user_service.UserResources(
        chutes=[user_service.ChuteRef(chute_id="c1", name="chute", version="v1")],
        images=[user_service.ImageRef(image_id="i1", name="img", tag="latest")],
        secrets=[user_service.SecretRef(secret_id="s1", key="OPENAI_KEY")],
    )

    # execute() call order: select api_keys, select instances, delete instances,
    # update instance_audit, delete chutes, delete images, delete secrets, delete user.
    api_keys = MagicMock()
    api_keys.scalars.return_value.all.return_value = ["k1", "k2"]
    instances = MagicMock()
    instances.all.return_value = [SimpleNamespace(instance_id="inst1", chute_id="c1")]
    execute_results = [api_keys, instances] + [MagicMock() for _ in range(6)]

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=execute_results)
    user = SimpleNamespace(user_id="u1", username="bob")

    result = await user_service.delete_user_and_resources(session, user, resources)

    assert result.api_key_ids == ["k1", "k2"]
    assert result.terminated_instances == [("inst1", "c1")]
    assert {b["reason"] for b in result.broadcasts} == {"chute_deleted", "image_deleted"}
    assert (result.chutes_deleted, result.images_deleted, result.secrets_deleted) == (1, 1, 1)
    # 8 statements: 2 selects, delete instances + audit update, delete chutes/images/secrets/user.
    assert session.execute.await_count == 8
