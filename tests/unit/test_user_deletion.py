"""Unit tests for support-initiated and self-service hard user deletion.

Covers the delete-eligibility rules, blocking-resource discovery, the API-key cache flush,
and the (now minimal) user-row delete. The full endpoint is DB-heavy and exercised by
integration tests.
"""

import api.database.orms  # noqa: F401  # register all ORM models so mappers configure

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from api.api_key import util as api_key_util
from api.permissions import Permissioning
from api.user import service as user_service
from api.user import router as user_router


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
        if isinstance(rows, int):
            r.scalar_one.return_value = rows
        elif isinstance(rows, tuple):
            r.one.return_value = rows
        else:
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
        [SimpleNamespace(account_id="a1", name="provider account")],
        (2, 1),
    )

    resources = await user_service.get_user_resources(session, "user-1")

    assert not resources.is_empty
    assert [c.chute_id for c in resources.chutes] == ["c1"]
    assert resources.chutes[0].version == "v1"
    assert resources.images[0].tag == "latest"
    assert resources.secrets[0].key == "OPENAI_KEY"
    assert resources.external_accounts[0].account_id == "a1"
    assert resources.external_operation_count == 2
    assert resources.unresolved_external_operation_count == 1
    assert session.execute.await_count == 5
    unresolved_statement = session.execute.await_args_list[4].args[0]
    assert any(
        isinstance(value, (list, tuple, set)) and "quarantined" in value
        for value in unresolved_statement.compile().params.values()
    )


@pytest.mark.asyncio
async def test_get_user_resources_empty_is_empty():
    session = _session_returning([], [], [], [], (0, 0))
    resources = await user_service.get_user_resources(session, "user-1")
    assert resources.is_empty


# ---------------------------------------------------------------------------
# check_user_deletable: single source of truth for the delete guards
# ---------------------------------------------------------------------------
def _user(*, permissions_bitmask=0, balance=0.0):
    return SimpleNamespace(
        user_id="u1", permissions_bitmask=permissions_bitmask, balance=balance
    )


def _empty():
    return user_service.UserResources()


def test_check_user_deletable_allows_clean_user():
    assert user_service.check_user_deletable(_user(), _empty()).allowed


def test_check_user_deletable_privileged_is_hard_block():
    eligibility = user_service.check_user_deletable(
        _user(permissions_bitmask=1 << 14), _empty()
    )
    assert not eligibility.allowed
    assert eligibility.context["permissions_bitmask"] == (1 << 14)


def test_check_user_deletable_ignores_benign_free_account_flag():
    eligibility = user_service.check_user_deletable(
        _user(permissions_bitmask=Permissioning.free_account.bitmask), _empty()
    )
    assert eligibility.allowed


def test_check_user_deletable_still_blocks_privilege_combined_with_free_account():
    eligibility = user_service.check_user_deletable(
        _user(
            permissions_bitmask=(
                Permissioning.free_account.bitmask
                | Permissioning.chutes_support.bitmask
            )
        ),
        _empty(),
    )
    assert not eligibility.allowed
    assert (
        eligibility.context["permissions_bitmask"]
        == Permissioning.chutes_support.bitmask
    )


def test_check_user_deletable_allows_balance_within_threshold():
    # Small positive/negative balances (within +/-25, e.g. billing lag) are deletable.
    assert user_service.check_user_deletable(_user(balance=25.0), _empty()).allowed
    assert user_service.check_user_deletable(_user(balance=-25.0), _empty()).allowed


def test_check_user_deletable_balance_outside_threshold_blocks():
    over = user_service.check_user_deletable(_user(balance=25.01), _empty())
    assert not over.allowed and over.context["balance"] == 25.01
    under = user_service.check_user_deletable(_user(balance=-100.0), _empty())
    assert not under.allowed and under.context["balance"] == -100.0


def test_check_user_deletable_resources_block():
    resources = user_service.UserResources(
        chutes=[user_service.ChuteRef(chute_id="c1", name="chute", version="v1")]
    )
    blocked = user_service.check_user_deletable(_user(), resources)
    assert not blocked.allowed
    assert blocked.context["chutes"][0]["chute_id"] == "c1"


def test_check_user_deletable_allows_anonymizable_external_operation_history():
    resources = user_service.UserResources(external_operation_count=3)
    assert user_service.check_user_deletable(_user(), resources).allowed


def test_check_user_deletable_blocks_unresolved_external_operation_settlements():
    resources = user_service.UserResources(
        external_operation_count=3,
        unresolved_external_operation_count=1,
    )

    blocked = user_service.check_user_deletable(_user(), resources)

    assert not blocked.allowed
    assert blocked.context == {
        "external_operation_count": 3,
        "unresolved_external_operation_count": 1,
    }
    assert "unresolved billing" in blocked.message


@pytest.mark.asyncio
async def test_resolve_user_can_lock_target_for_deletion():
    result = MagicMock()
    expected = SimpleNamespace(user_id="u1", username="bob")
    result.unique.return_value.scalar_one_or_none.return_value = expected
    session = AsyncMock()
    session.execute.return_value = result

    resolved = await user_service.resolve_user(session, "bob", for_update=True)

    assert resolved is expected
    statement = session.execute.await_args.args[0]
    assert statement._for_update_arg is not None


@pytest.mark.asyncio
async def test_self_delete_locks_user_and_uses_shared_deletion_flow():
    result = MagicMock()
    user = SimpleNamespace(user_id="u1", username="bob")
    result.unique.return_value.scalar_one_or_none.return_value = user
    session = AsyncMock()
    session.execute.return_value = result

    with (
        patch.object(user_router, "_delete_locked_user", new=AsyncMock()) as remove,
        patch.object(user_router, "update_log_context"),
    ):
        response = await user_router.delete_my_user(
            db=session,
            authorization="Bearer self-delete-token",
        )

    assert response == {"deleted": True}
    statement = session.execute.await_args.args[0]
    assert statement._for_update_arg is not None
    remove.assert_awaited_once_with(session, user)


@pytest.mark.asyncio
async def test_shared_deletion_flow_preserves_safe_conflict_context():
    session = AsyncMock()
    user = SimpleNamespace(
        user_id="u1",
        username="bob",
        permissions_bitmask=0,
        balance=0.0,
    )
    resources = user_service.UserResources(
        external_operation_count=4,
        unresolved_external_operation_count=2,
    )
    eligibility = user_service.check_user_deletable(user, resources)

    with (
        patch.object(
            user_router,
            "get_user_resources",
            new=AsyncMock(return_value=resources),
        ),
        patch.object(
            user_router,
            "check_user_deletable",
            return_value=eligibility,
        ),
        patch.object(user_router, "delete_user", new=AsyncMock()) as remove,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await user_router._delete_locked_user(session, user)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == {
        "message": eligibility.message,
        "user_id": "u1",
        "external_operation_count": 4,
        "unresolved_external_operation_count": 2,
    }
    remove.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_shared_deletion_flow_invalidates_keys_after_commit():
    session = AsyncMock()
    user = SimpleNamespace(
        user_id="u1",
        username="bob",
        permissions_bitmask=0,
        balance=0.0,
    )
    resources = user_service.UserResources()
    eligibility = user_service.check_user_deletable(user, resources)
    events: list[str] = []

    async def commit():
        events.append("commit")

    async def invalidate(api_key_id: str):
        events.append(f"invalidate:{api_key_id}")

    session.commit.side_effect = commit
    with (
        patch.object(
            user_router,
            "get_user_resources",
            new=AsyncMock(return_value=resources),
        ),
        patch.object(
            user_router,
            "check_user_deletable",
            return_value=eligibility,
        ),
        patch.object(
            user_router,
            "delete_user",
            new=AsyncMock(return_value=["k1", "k2"]),
        ),
        patch.object(user_router, "invalidate_api_key_cache", new=invalidate),
    ):
        await user_router._delete_locked_user(session, user)

    assert events == ["commit", "invalidate:k1", "invalidate:k2"]


# ---------------------------------------------------------------------------
# delete_user: minimal row delete (resources are removed manually beforehand)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_delete_user_returns_api_key_ids_and_deletes_row():
    api_keys = MagicMock()
    api_keys.scalars.return_value.all.return_value = ["k1", "k2"]
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[api_keys, MagicMock()])
    user = SimpleNamespace(user_id="u1", username="bob")

    api_key_ids = await user_service.delete_user(session, user)

    assert api_key_ids == ["k1", "k2"]
    # Exactly two statements: select api key ids, then delete the user row.
    assert session.execute.await_count == 2
