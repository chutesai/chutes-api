"""Hand-written doubles for the chute log read routes.

No MagicMock. The route tests override catalog lookup and the Loki read with these
objects. They do not stand in for the access policy.
"""

from api.permissions import Permissioning


class UserStub:
    def __init__(self, user_id, netuids=None, roles=None):
        self.user_id = user_id
        self.netuids = list(netuids or [])
        self.permissions_bitmask = 0
        for role in roles or []:
            Permissioning.enable(self, role)

    def has_role(self, role):
        return Permissioning.enabled(self, role)


class ChuteRow:
    def __init__(self, chute_id, user_id, public, name):
        self.chute_id = chute_id
        self.user_id = user_id
        self.public = public
        self.name = name


class InstanceRow:
    def __init__(self, instance_id, config_id, chute_id, activated_at=None):
        self.instance_id = instance_id
        self.config_id = config_id
        self.chute_id = chute_id
        self.activated_at = activated_at


class LaunchConfigRow:
    def __init__(self, config_id, chute_id, miner_hotkey, failed_at=None, instance=None):
        self.config_id = config_id
        self.chute_id = chute_id
        self.miner_hotkey = miner_hotkey
        self.failed_at = failed_at
        self.instance = instance


class LogLine:
    def __init__(self, ts, stream, log):
        self.ts = ts
        self.stream = stream
        self.log = log


class Catalog:
    """In-memory stand-in for instance, chute, share, and launch-config lookups."""

    def __init__(self):
        self.instances = {}
        self.chutes = {}
        self.shares = set()
        self.launch_configs = {}

    async def get_instance(self, instance_id):
        return self.instances.get(instance_id)

    async def get_chute(self, chute_id):
        return self.chutes.get(chute_id)

    async def is_shared_with(self, chute_id, user_id):
        return (chute_id, user_id) in self.shares

    async def get_launch_config(self, config_id):
        return self.launch_configs.get(config_id)


class LokiSpy:
    """Records every (config_id, forced) passed to read_config_logs."""

    def __init__(self, lines=None):
        self.lines = list(lines or [])
        self.calls = []

    async def __call__(self, config_id, forced, limit=5000):
        self.calls.append((config_id, dict(forced)))
        return list(self.lines)
