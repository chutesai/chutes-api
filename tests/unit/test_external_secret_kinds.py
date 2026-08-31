from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

import api.database.orms  # noqa: F401  # register relationship targets before mapper use

with patch("ctypes.CDLL", return_value=MagicMock()):
    from api.external_backend import admin_router, polling, service
    from api.instance import router as instance_router
    from api.secret import router as secret_router
from api.secret.schemas import Secret


def test_secret_kind_is_a_database_enforced_discriminator():
    assert Secret.__table__.c.kind.nullable is False
    assert Secret.__table__.c.kind.server_default.arg == "chute"
    constraint = next(
        item for item in Secret.__table__.constraints if item.name == "ck_secrets_kind"
    )
    assert "external_backend" in str(constraint.sqltext)


@pytest.mark.asyncio
async def test_hosted_launch_config_structurally_selects_only_chute_secrets():
    result = MagicMock()
    result.unique.return_value.scalars.return_value.all.return_value = []
    db = AsyncMock()
    db.execute.return_value = result
    instance = SimpleNamespace(
        instance_id="instance-id",
        chutes_version="0.0.0",
        job=None,
    )
    launch_config = SimpleNamespace(
        chute_id="chute-id",
        verified_at=datetime.now(UTC),
        config_id="config-id",
    )

    await instance_router._build_launch_config_verified_response(
        db, instance, launch_config
    )

    statement = db.execute.await_args.args[0]
    assert "secrets.kind" in str(statement)


@pytest.mark.asyncio
async def test_user_secret_catalog_structurally_excludes_external_credentials():
    total = MagicMock()
    total.scalar.return_value = 0
    rows = MagicMock()
    rows.scalars.return_value.all.return_value = []
    db = AsyncMock()
    db.execute.side_effect = [total, rows]

    await secret_router.list_secrets(
        db=db,
        current_user=SimpleNamespace(user_id="user-id"),
    )

    statement = db.execute.await_args_list[1].args[0]
    assert "secrets.kind" in str(statement)


@pytest.mark.asyncio
async def test_managed_credentials_are_created_with_external_kind(monkeypatch):
    monkeypatch.setattr(
        admin_router, "encrypt_secret", AsyncMock(return_value="encrypted")
    )
    db = AsyncMock()
    db.get.return_value = None
    db.add = MagicMock()

    references = await admin_router._store_credentials(
        db,
        user_id="user-id",
        account_id="account-id",
        credentials={"primary": SecretStr("credential")},
    )

    stored = db.add.call_args.args[0]
    assert stored.kind == "external_backend"
    assert references["primary"] == f"secret://{stored.secret_id}"


@pytest.mark.asyncio
async def test_request_resolver_structurally_selects_only_external_credentials(
    monkeypatch,
):
    secret = SimpleNamespace(value="encrypted")
    result = MagicMock()
    result.scalar_one_or_none.return_value = secret
    session = AsyncMock()
    session.execute.return_value = result

    @asynccontextmanager
    async def session_factory(**_kwargs):
        yield session

    monkeypatch.setattr(service, "get_session", session_factory)
    monkeypatch.setattr(service, "decrypt_secret", AsyncMock(return_value="credential"))
    account = SimpleNamespace(
        user_id="user-id",
        credential_references={"primary": "secret://secret-id"},
    )

    assert (
        await service.build_secret_resolver(account)("secret://secret-id")
        == "credential"
    )
    statement = session.execute.await_args.args[0]
    assert "secrets.kind" in str(statement)


@pytest.mark.asyncio
async def test_poller_resolver_structurally_selects_only_external_credentials(
    monkeypatch,
):
    secret = SimpleNamespace(value="encrypted")
    result = MagicMock()
    result.scalar_one_or_none.return_value = secret
    session = AsyncMock()
    session.execute.return_value = result

    @asynccontextmanager
    async def session_factory(**_kwargs):
        yield session

    monkeypatch.setattr(polling, "get_session", session_factory)
    from api.payment import util as payment_util

    monkeypatch.setattr(
        payment_util, "decrypt_secret", AsyncMock(return_value="credential")
    )
    account = polling.AccountSnapshot(
        account_id="account-id",
        user_id="user-id",
        base_url="https://gateway.example.test",
        credential_references={"primary": "secret://secret-id"},
        auth_header_templates=(),
        connection_config={},
    )
    executor = polling._default_executor_factory(account)

    assert await executor._secret_resolver("secret://secret-id") == "credential"
    statement = session.execute.await_args.args[0]
    assert "secrets.kind" in str(statement)
