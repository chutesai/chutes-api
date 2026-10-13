"""Access policy and row lookup for stored chute-log reads.

UNCLEAR: whether Loki is network-isolated from callers other than this process.
This module does not add a network check. Matchers are forced here, server-side.

Schema imports stay inside the SQL adapter. Importing them here would configure
mappers before the rest of the models (Logo and others) are loaded, which breaks
a router-only test app.
"""

from datetime import datetime, timezone

from api.chute_logs import service
from api.chute_logs.exceptions import (
    LogReadForbidden,
    LogReadNotFound,
    MinerLogsNotPublic,
    UnknownLaunchConfig,
)
from api.permissions import Permissioning
from api.user.service import subnet_role_accessible


class _InstanceView:
    def __init__(self, instance_id, config_id, chute_id, activated_at):
        self.instance_id = instance_id
        self.config_id = config_id
        self.chute_id = chute_id
        self.activated_at = activated_at


class _ChuteView:
    def __init__(self, chute_id, user_id, public, name):
        self.chute_id = chute_id
        self.user_id = user_id
        self.public = public
        self.name = name


class _LaunchView:
    def __init__(self, config_id, chute_id, miner_hotkey, failed_at, instance):
        self.config_id = config_id
        self.chute_id = chute_id
        self.miner_hotkey = miner_hotkey
        self.failed_at = failed_at
        self.instance = instance


class SqlLogCatalog:
    """Production lookups. Tests substitute a hand-written catalog with the same calls.

    Values are copied out before the session closes. The route reads them after
    the lookup returns, and a live ORM row would be detached by then.
    """

    async def get_instance(self, instance_id):
        from sqlalchemy import select

        from api.database import get_session
        from api.instance.schemas import Instance

        async with get_session(readonly=True) as session:
            row = await session.scalar(select(Instance).where(Instance.instance_id == instance_id))
            if row is None:
                return None
            return _InstanceView(row.instance_id, row.config_id, row.chute_id, row.activated_at)

    async def get_chute(self, chute_id):
        from sqlalchemy import select

        from api.chute.schemas import Chute
        from api.database import get_session

        async with get_session(readonly=True) as session:
            row = await session.scalar(select(Chute).where(Chute.chute_id == chute_id))
            if row is None:
                return None
            return _ChuteView(row.chute_id, row.user_id, row.public, row.name)

    async def is_shared_with(self, chute_id, user_id):
        from api.chute.util import is_shared

        return await is_shared(chute_id, user_id)

    async def get_launch_config(self, config_id):
        from sqlalchemy import select
        from sqlalchemy.orm import joinedload

        from api.database import get_session
        from api.instance.schemas import LaunchConfig

        async with get_session(readonly=True) as session:
            row = await session.scalar(
                select(LaunchConfig)
                .where(LaunchConfig.config_id == config_id)
                .options(joinedload(LaunchConfig.instance))
            )
            if row is None:
                return None
            instance = row.instance
            instance_view = None
            if instance is not None:
                instance_view = _InstanceView(
                    instance.instance_id,
                    instance.config_id,
                    instance.chute_id,
                    instance.activated_at,
                )
            return _LaunchView(
                row.config_id,
                row.chute_id,
                row.miner_hotkey,
                row.failed_at,
                instance_view,
            )


def get_catalog():
    return SqlLogCatalog()


def get_log_reader():
    return service.read_config_logs


def decode_launch_jwt(token):
    """Validate a launch-config JWT. ``sub`` is the config id. No hotkey re-check."""
    from api.instance.util import _decode_chutes_jwt

    return _decode_chutes_jwt(token, require_exp=True)


def authorize_user_read(chute, user, shared):
    """Support bypasses. Otherwise deny when the caller is neither owner nor sharee, or the chute is public, unless a subnet admin's netuid matches the chute name."""
    if user.has_role(Permissioning.chutes_support):
        return
    not_owner = chute.user_id != user.user_id
    if ((not_owner and not shared) or chute.public) and not subnet_role_accessible(
        chute, user, admin=True
    ):
        raise LogReadForbidden()


def require_instance(instance):
    if instance is None:
        raise LogReadNotFound()
    return instance


def authorize_miner_read(chute):
    if chute is None or not chute.public:
        raise MinerLogsNotPublic()


def require_launch_config(launch):
    if launch is None:
        raise UnknownLaunchConfig()
    return launch


def _parse_log_ts(value):
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    if "." in text:
        head, rest = text.split(".", 1)
        split_at = len(rest)
        for index, char in enumerate(rest):
            if char in "+-" and index > 0:
                split_at = index
                break
        frac, tz = rest[:split_at], rest[split_at:]
        text = f"{head}.{frac[:6]}{tz}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def lines_before_activation(lines, activated_at):
    """Keep lines strictly before ``activated_at``.

    The cutoff belongs in the LogQL query (``ts`` compared to ``activated_at``)
    so Loki does the discard. Leaving the drop in process memory is not the
    permanent design. This branch only filters after ``read_config_logs`` returns.

    UNCLEAR: a line whose ``ts`` does not parse is dropped (fail closed) so an
    unreadable timestamp cannot sneak past the cutoff.
    """
    if activated_at is None:
        return list(lines)
    if activated_at.tzinfo is None:
        activated_at = activated_at.replace(tzinfo=timezone.utc)
    kept = []
    for line in lines:
        ts = _parse_log_ts(getattr(line, "ts", None))
        if ts is not None and ts < activated_at:
            kept.append(line)
    return kept
