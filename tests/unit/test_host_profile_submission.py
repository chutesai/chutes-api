"""
Unit tests for miner host profile submissions (POST /servers/tdx/host_profiles).
"""

import copy
import inspect
import pytest
import orjson as json
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.params import Depends as DependsMarker

from api.config import TeeMeasurementConfig
from api.constants import HostProfileStatus
from api.rate_limit import rate_limit_miner
from api.server.router import (
    submit_host_profile,
    tdx_preflight,
    tdx_host_profile_status,
)
from api.server.schemas import HostProfile
from api.server.util import (
    host_profile_is_known,
    resolve_host_profile_status,
    store_host_profile,
    measurements_for_fingerprint,
)


# A trimmed but faithful copy of sek8s discover-profile.sh output.
SAMPLE_PROFILE = {
    "hostname": "tdx-h200-01",
    "timestamp": "2026-08-20T12:00:00Z",
    "host": {
        "product_name": "PowerEdge XE9680",
        "board_vendor": "Dell Inc.",
        "board_name": "0K0M1P",
        "bios_vendor": "Dell Inc.",
        "bios_version": "1.10.2",
        "bios_date": "04/01/2026",
        "os_version_id": "24.04",
    },
    "launch_determinism": {
        "qemu_version": "8.2.2",
        "qemu_version_full": "QEMU emulator version 8.2.2 (Debian)",
        "numa_node_count": 2,
        "numa_topology_eligible": True,
        "cpu_args": "host",
        "host_cpu_topology": "sockets=2,cores_per_socket=48,threads_per_core=2",
    },
    "gpu": {
        "pci_device_ids": ["2335"],
        "bdfs": ["0000:1b:00.0", "0000:43:00.0"],
        "count": 8,
        "vram_gb": 141,
        "bar_size_mb": 262144,
        "numa_nodes": [0, 0, 0, 0, 1, 1, 1, 1],
        "vbios": ["96.00.89.00.01"],
    },
    "pci_topology": "-+-[0000:e0]-+-00.0  Intel...",
    "cpu": {
        "total": 192,
        "sockets": 2,
        "cores_per_socket": 48,
        "threads_per_core": 2,
        "cpu_vendor": "GenuineIntel",
        "cpu_processor_id": "f26c0000fffba91f",
    },
    "memory": {
        "total_gb": 2015,
        "suggested_ram_per_gpu_gb": 128,
        "suggested_total_vm_ram_gb": 1024,
    },
    "numa": {
        "node_count": 2,
        "nodes": [0, 1],
        "cpus_per_node": {"0": "0-47", "1": "48-95"},
    },
    "nic": {
        "ib_class_count": 8,
        "eth_class_count": 2,
        "ib_devices": ["0000:1a:00.0"],
        "bridge_pfs": [],
        "passthrough_candidates": ["0000:1a:00.0", "0000:42:00.0"],
        "passthrough_numa_nodes": [0, 1],
    },
    "nvswitch": {
        "present": True,
        "count": 4,
        "devices": ["0000:0a:00.0"],
        "numa_nodes": [0],
    },
}


def _profile(**overrides) -> HostProfile:
    data = copy.deepcopy(SAMPLE_PROFILE)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value
    return HostProfile(**data)


def _mock_request(body: bytes = b"{}"):
    request = MagicMock()
    request.body = AsyncMock(return_value=body)
    return request


class TestFingerprint:
    def test_identical_host_class_matches(self):
        assert _profile().fingerprint == _profile().fingerprint

    def test_cosmetic_differences_do_not_change_fingerprint(self):
        """Two hosts of the same class differing only in identity/BIOS strings collapse to one."""
        other = _profile(
            hostname="tdx-h200-47",
            host={"bios_version": "1.11.0", "bios_date": "06/01/2026"},
            pci_topology="a completely different lspci tree",
            gpu={"bdfs": ["0000:99:00.0"], "vbios": ["96.00.90.00.01"]},
        )
        assert other.fingerprint == _profile().fingerprint

    @pytest.mark.parametrize(
        "overrides",
        [
            {"gpu": {"count": 4}},
            {"gpu": {"pci_device_ids": ["2901"]}},
            {"gpu": {"bar_size_mb": 131072}},
            {"cpu": {"cpu_processor_id": "f36a0000fffba91f"}},
            {"cpu": {"cpu_vendor": "AuthenticAMD"}},
            {"cpu": {"total": 288}},
            {"memory": {"total_gb": 4031}},
            {"numa": {"node_count": 8}},
            {"launch_determinism": {"qemu_version": "9.1.0"}},
            {"launch_determinism": {"cpu_args": "host,-avx10"}},
            {"nvswitch": {"count": 0}},
            {"nic": {"ib_class_count": 0}},
            {"nic": {"passthrough_candidates": []}},
        ],
    )
    def test_measurement_relevant_differences_change_fingerprint(self, overrides):
        assert _profile(**overrides).fingerprint != _profile().fingerprint

    def test_gpu_id_ordering_is_irrelevant(self):
        a = _profile(gpu={"pci_device_ids": ["2335", "2901"]})
        b = _profile(gpu={"pci_device_ids": ["2901", "2335", "2901"]})
        assert a.fingerprint == b.fingerprint

    def test_is_a_sha256_hex_digest(self):
        fingerprint = _profile().fingerprint
        assert len(fingerprint) == 64
        assert set(fingerprint) <= set("0123456789abcdef")


class TestValidation:
    def test_rejects_profile_without_gpus(self):
        with pytest.raises(ValueError):
            _profile(gpu={"count": 0, "pci_device_ids": []})

    def test_rejects_profile_missing_required_block(self):
        data = copy.deepcopy(SAMPLE_PROFILE)
        del data["cpu"]
        with pytest.raises(ValueError):
            HostProfile(**data)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"future_block": {"something": "new"}},
            {"gpu": {"new_thing": 1}},
            {"launch_determinism": {"new_thing": "x"}},
            {"cpu": {"new_thing": "x"}},
            {"host": {"new_thing": "x"}},
            {"nic": {"new_thing": "x"}},
        ],
    )
    def test_unknown_fields_are_rejected(self, overrides):
        """
        Every block forbids extras, at every level. Nothing free-form reaches the generation
        tooling, and the document matches the OpenAPI schema the load balancers validate against.
        """
        with pytest.raises(ValueError):
            _profile(**overrides)

    def test_older_script_missing_newer_keys_still_validates(self):
        """
        Forbidding extras is one-way: a script OLDER than the API is fine (defaults fill in), only
        a script that ADDS a key before the API knows it breaks.
        """
        lean = {
            "launch_determinism": {"qemu_version": "10.1.0"},
            "gpu": {"pci_device_ids": ["2335"], "count": 8},
            "cpu": {
                "total": 192,
                "sockets": 2,
                "cores_per_socket": 48,
                "threads_per_core": 2,
            },
            "memory": {"total_gb": 2015},
            "numa": {"node_count": 2},
        }
        profile = HostProfile(**lean)
        assert profile.gpu.count == 8
        assert profile.nvswitch.count == 0

    def test_every_key_the_script_emits_is_modeled(self):
        """
        The sample is a complete discover-profile.sh document; with extras forbidden it validates
        only if every key it emits is modeled. This is the test that fails when the script grows a
        field and the schema has not caught up.
        """
        profile = HostProfile(**copy.deepcopy(SAMPLE_PROFILE))
        assert profile.gpu.vbios == ["96.00.89.00.01"]
        assert profile.memory.suggested_total_vm_ram_gb == 1024
        assert profile.numa.cpus_per_node == {"0": "0-47", "1": "48-95"}
        assert profile.nic.eth_class_count == 2
        assert profile.nvswitch.present is True

    @pytest.mark.parametrize("field", ["hostname", "timestamp"])
    def test_machine_identifying_fields_are_never_serialised(self, field):
        """
        They identify one machine, not a host class. Accepted so a real document validates, but
        excluded from every dump -- so they never reach the stored column and cannot leak from the
        public endpoint, whatever serialisation path is used.
        """
        profile = _profile()

        assert getattr(profile, field) is not None
        assert field not in profile.model_dump()
        assert field not in profile.model_dump(by_alias=True)
        assert field not in profile.model_dump(mode="json")
        assert field not in json.loads(profile.model_dump_json())

    def test_omitting_them_entirely_still_validates(self):
        data = copy.deepcopy(SAMPLE_PROFILE)
        del data["hostname"], data["timestamp"]

        assert HostProfile(**data).fingerprint == _profile().fingerprint

    def test_wire_keys_map_to_descriptive_attributes(self):
        """discover-profile.sh's ``host`` / ``launch_determinism`` keys, named for what they hold."""
        profile = _profile()
        assert profile.platform.bios_version == "1.10.2"
        assert profile.platform.os_version_id == "24.04"
        assert profile.qemu.qemu_version == "8.2.2"
        assert profile.qemu.cpu_args == "host"

    def test_restated_fields_are_declared_but_never_read(self):
        """
        ``launch_determinism`` restates values that belong to ``numa`` and ``cpu``. They must be
        declared, or a real submission would fail now that extras are forbidden -- but nothing
        reads them, so a stale copy cannot influence anything. The fingerprint test below pins
        that behavior.
        """
        profile = _profile()
        assert profile.qemu.numa_node_count == 2
        assert profile.qemu.host_cpu_topology == "sockets=2,cores_per_socket=48,threads_per_core=2"
        # numa_topology_eligible is just node_count == 2, hence redundant with the numa block.
        assert profile.qemu.numa_topology_eligible == (profile.numa.node_count == 2)

    @pytest.mark.parametrize(
        "overrides",
        [
            # Values that reach S3 object metadata (HTTP headers) and log lines.
            {"gpu": {"pci_device_ids": ["2335\r\nx-amz-meta-evil: 1"]}},
            {"gpu": {"pci_device_ids": ["2335\nInjected log line"]}},
            {"gpu": {"pci_device_ids": ["not-hex"]}},
            {"gpu": {"pci_device_ids": ["\u00e9\u00e9\u00e9\u00e9"]}},
            {"gpu": {"pci_device_ids": ["2335"] * 65}},
            {"gpu": {"count": 10**9}},
            # A command-line fragment the offline generation job may well interpolate.
            {"launch_determinism": {"cpu_args": "host; rm -rf /"}},
            {"launch_determinism": {"cpu_args": "host $(id)"}},
            {"launch_determinism": {"cpu_args": "host`id`"}},
            {"launch_determinism": {"qemu_version": "$(id)"}},
            {"cpu": {"cpu_processor_id": "'; DROP TABLE servers;--"}},
            {"cpu": {"cpu_vendor": "GenuineIntel\nnope"}},
            {"cpu": {"total": 10**9}},
            {"memory": {"total_gb": 10**12}},
            {"numa": {"node_count": 10**6}},
            {"host": {"bios_version": "1.0\x00truncated"}},
            {"hostname": "host\nname"},
        ],
    )
    def test_hostile_values_are_rejected(self, overrides):
        """
        Modeled fields are bounded and pattern-checked: these are the values that would otherwise
        reach an HTTP header, a log line, or a generated command line.
        """
        with pytest.raises(ValueError):
            _profile(**overrides)

    def test_legitimate_script_values_still_validate(self):
        """The real shapes discover-profile.sh emits must not be caught by the constraints."""
        profile = _profile(
            launch_determinism={
                "qemu_version": "10.1.0",
                "qemu_version_full": "QEMU emulator version 10.1.0 (Debian 1:10.1.0+ds-1)",
                "cpu_args": "host,-avx10",
            },
            cpu={"cpu_vendor": "AuthenticAMD", "cpu_processor_id": "f26c0000fffba91f"},
            host={
                "board_name": "PowerEdge XE9680 (Rev. 1.2)",
                "bios_date": "04/01/2026",
            },
        )
        assert profile.qemu.cpu_args == "host,-avx10"
        assert profile.platform.board_name == "PowerEdge XE9680 (Rev. 1.2)"

    def test_pci_topology_is_bounded_but_permissive(self):
        """
        The lspci tree is modeled, so it is length-bounded and free of control characters other
        than the newlines and tabs it is drawn with -- but its content is otherwise unconstrained,
        since it is vendor text we do not get to predict. It is only ever stored, never used as a
        key, header, or log value.
        """
        profile = _profile(pci_topology="-+-[0000:e0]-\\-00.0 weird $ text\n  \\-01.0-[e1]--")
        assert "weird $ text" in profile.pci_topology

        with pytest.raises(ValueError):
            _profile(pci_topology="tree\x00truncated")
        with pytest.raises(ValueError):
            _profile(pci_topology="x" * (64 * 1024 + 1))

    def test_restated_numa_count_cannot_shift_the_fingerprint(self):
        """A profile whose restated copy disagrees with ``numa`` fingerprints off ``numa``."""
        assert (
            _profile(launch_determinism={"numa_node_count": 8}).fingerprint
            == _profile().fingerprint
        )
        assert _profile(numa={"node_count": 8}).fingerprint != _profile().fingerprint


def _mock_session(created=True, known=False):
    """
    AsyncSession double. `created` is what the INSERT ... RETURNING yields (None = conflict, i.e.
    the host class was already on file); `known` is what the existence SELECT reports.
    """
    db = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value="fp" if created else None)
    # host_profile_state reads (measured_at,) via one_or_none: a bare (None,) row means on file but
    # still pending, no row means never seen.
    result.one_or_none = MagicMock(return_value=(None,) if known else None)
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()
    return db


class TestStorage:
    @pytest.mark.asyncio
    async def test_new_profile_is_recorded(self):
        db = _mock_session(created=True)
        profile = _profile()
        fingerprint, stored = await store_host_profile(db=db, profile=profile, hotkey="5Fhotkey")

        assert stored is True
        assert fingerprint == profile.fingerprint
        db.commit.assert_awaited_once()
        values = db.execute.await_args.args[0].compile().params
        assert values["fingerprint"] == profile.fingerprint
        assert values["miner_hotkey"] == "5Fhotkey"
        # The signature is admission control at the endpoint, not something re-checked later, so
        # the signed bytes are deliberately not retained.
        assert "raw_body" not in values
        assert "signature" not in values

    @pytest.mark.asyncio
    async def test_profile_column_is_the_wire_shape(self):
        """Stored under the keys discover-profile.sh emits, so it is queryable as submitted."""
        db = _mock_session(created=True)

        await store_host_profile(db=db, profile=_profile(), hotkey="5Fhotkey")

        stored_profile = db.execute.await_args.args[0].compile().params["profile"]
        assert stored_profile["launch_determinism"]["qemu_version"] == "8.2.2"
        assert stored_profile["gpu"]["pci_device_ids"] == ["2335"]

    @pytest.mark.asyncio
    async def test_known_host_class_is_not_overwritten(self):
        """ON CONFLICT DO NOTHING: first write wins, whether it is pending or already measured."""
        db = _mock_session(created=False)

        fingerprint, stored = await store_host_profile(db=db, profile=_profile(), hotkey="5Fother")

        assert stored is False
        assert fingerprint == _profile().fingerprint
        db.commit.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("known, expected", [(False, False), (True, True)])
    async def test_known_checks_the_table(self, known, expected):
        db = _mock_session(known=known)

        assert await host_profile_is_known(db, _profile().fingerprint) is expected


class TestRateLimitMinerDependency:
    """The route's single auth+metering dependency (api.rate_limit.rate_limit_miner)."""

    def _dependency(self, **kwargs):
        return rate_limit_miner("host_profile_submit", 10, window_seconds=3600, **kwargs)

    def test_authentication_is_a_subdependency(self):
        """
        Auth must resolve BEFORE metering, which only holds if it is a sub-dependency of this one
        rather than a sibling Depends on the route -- siblings have no guaranteed order, and
        metering an unverified hotkey would let anyone spend a real miner's quota.
        """
        params = inspect.signature(self._dependency()).parameters
        assert isinstance(params["user"].default, DependsMarker)
        assert isinstance(params["db"].default, DependsMarker)

    @pytest.mark.asyncio
    @patch("api.rate_limit.check_rate_limit", new_callable=AsyncMock)
    @patch("api.rate_limit.is_miner_blacklisted", new_callable=AsyncMock)
    async def test_meters_the_authenticated_hotkey(self, blacklisted, check):
        blacklisted.return_value = None
        user = MagicMock()

        result = await self._dependency()(hotkey="5Fhotkey", db=MagicMock(), user=user)

        assert result is user
        assert check.await_args_list[0].args[:2] == ("host_profile_submit", 10)
        assert check.await_args_list[0].kwargs == {
            "window_seconds": 3600,
            "identity": "5Fhotkey",
        }

    @pytest.mark.asyncio
    @patch("api.rate_limit.check_rate_limit", new_callable=AsyncMock)
    @patch("api.rate_limit.is_miner_blacklisted", new_callable=AsyncMock)
    async def test_global_ceiling_is_metered_without_an_identity(self, blacklisted, check):
        blacklisted.return_value = None

        await self._dependency(global_limit=120)(
            hotkey="5Fhotkey", db=MagicMock(), user=MagicMock()
        )

        global_call = check.await_args_list[1]
        assert global_call.args[:2] == ("host_profile_submit:global", 120)
        assert "identity" not in global_call.kwargs

    @pytest.mark.asyncio
    @patch("api.rate_limit.check_rate_limit", new_callable=AsyncMock)
    @patch("api.rate_limit.is_miner_blacklisted", new_callable=AsyncMock)
    async def test_blacklisted_miner_is_rejected_and_spends_no_quota(self, blacklisted, check):
        blacklisted.return_value = "Your hotkey has been blacklisted: spam"

        with pytest.raises(HTTPException) as exc:
            await self._dependency()(hotkey="5Fbad", db=MagicMock(), user=MagicMock())

        assert exc.value.status_code == 403
        check.assert_not_called()


class TestCheckRateLimit:
    """The Redis counter shared by both rate limit dependencies."""

    def _redis(self, count):
        redis = MagicMock()
        redis.incr = AsyncMock(return_value=count)
        redis.expire = AsyncMock()
        return redis

    @pytest.mark.asyncio
    @patch("api.rate_limit.settings")
    async def test_identity_gets_its_own_counter(self, mock_settings):
        from api.rate_limit import check_rate_limit

        redis = self._redis(1)
        mock_settings.redis_client = redis

        await check_rate_limit("host_profile_submit", 10, window_seconds=3600, identity="5Fa")
        await check_rate_limit("host_profile_submit", 10, window_seconds=3600, identity="5Fb")

        key_a = redis.incr.await_args_list[0].args[0]
        key_b = redis.incr.await_args_list[1].args[0]
        assert "5Fa" in key_a and "5Fb" in key_b
        assert key_a != key_b
        # The window TTL follows the window, not a hardcoded minute.
        assert redis.expire.await_args.args[1] == 7200

    @pytest.mark.asyncio
    @patch("api.rate_limit.settings")
    async def test_raises_429_past_the_limit(self, mock_settings):
        from api.rate_limit import check_rate_limit

        mock_settings.redis_client = self._redis(11)

        with pytest.raises(HTTPException) as exc:
            await check_rate_limit("host_profile_submit", 10, window_seconds=3600, identity="5Fa")

        assert exc.value.status_code == 429

    @pytest.mark.asyncio
    @patch("api.rate_limit.settings")
    async def test_zero_limit_disables_metering(self, mock_settings):
        from api.rate_limit import check_rate_limit

        redis = self._redis(1)
        mock_settings.redis_client = redis

        await check_rate_limit("host_profile_submit", 0, window_seconds=3600, identity="5Fa")

        redis.incr.assert_not_called()


class TestEndpoint:
    """The handler body -- auth, blacklist and metering are the dependency's job (above)."""

    async def _submit(self, body=None):
        body = body if body is not None else json.dumps(SAMPLE_PROFILE)
        return await submit_host_profile(
            request=_mock_request(body),
            profile=_profile(),
            db=MagicMock(),
            hotkey="5Fhotkey",
            _=None,
        )

    @pytest.mark.asyncio
    @patch("api.server.router.resolve_host_profile_status", new_callable=AsyncMock)
    async def test_accepts_new_profile(self, resolve):
        resolve.return_value = (_profile().fingerprint, HostProfileStatus.PENDING, True)

        result = await self._submit()

        assert result.stored is True
        assert result.status == HostProfileStatus.PENDING
        assert result.fingerprint == _profile().fingerprint
        assert resolve.await_args.kwargs["hotkey"] == "5Fhotkey"

    @pytest.mark.asyncio
    @patch("api.server.router.resolve_host_profile_status", new_callable=AsyncMock)
    async def test_duplicate_submission_is_a_no_op(self, resolve):
        resolve.return_value = (
            _profile().fingerprint,
            HostProfileStatus.PENDING,
            False,
        )

        result = await self._submit()

        assert result.stored is False
        assert result.status == HostProfileStatus.PENDING

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "profile_status, expect",
        [
            (HostProfileStatus.ACCEPTED, "retained for attestation"),
            (HostProfileStatus.PENDING, "will be generated"),
            (HostProfileStatus.UNKNOWN, "no measurements"),
        ],
    )
    @patch("api.server.router.resolve_host_profile_status", new_callable=AsyncMock)
    async def test_every_status_has_a_detail_message(self, resolve, profile_status, expect):
        resolve.return_value = (_profile().fingerprint, profile_status, False)

        result = await self._submit()

        assert result.status == profile_status
        assert expect in result.detail

    @pytest.mark.asyncio
    @patch("api.server.router.resolve_host_profile_status", new_callable=AsyncMock)
    async def test_oversized_body_is_rejected(self, resolve):
        with pytest.raises(HTTPException) as exc:
            await self._submit(body=b"x" * (256 * 1024 + 1))

        assert exc.value.status_code == 413
        resolve.assert_not_called()

    @pytest.mark.asyncio
    @patch("api.server.router.resolve_host_profile_status", new_callable=AsyncMock)
    async def test_storage_failure_surfaces_as_503(self, resolve):
        resolve.side_effect = Exception("bucket unreachable")

        with pytest.raises(HTTPException) as exc:
            await self._submit()

        assert exc.value.status_code == 503
        assert "bucket unreachable" not in exc.value.detail

    def test_route_declares_the_miner_rate_limit_dependency(self):
        """The metering is visible at the route, not buried in the handler body."""
        params = inspect.signature(submit_host_profile).parameters
        assert isinstance(params["_"].default, DependsMarker)
        assert params["_"].default.dependency.__name__ == "_rate_limit_miner"


class TestPreflightEndpoint:
    """POST /servers/tdx/preflight -- the one-boolean launchability check. Stores nothing; the
    answer is just whether a measurement for the caller's (version, rc) carries the fingerprint.
    """

    async def _preflight(self, version="1.4.0", rc=False, body=None):
        body = body if body is not None else json.dumps(SAMPLE_PROFILE)
        return await tdx_preflight(
            request=_mock_request(body),
            profile=_profile(),
            version=version,
            rc=rc,
            hotkey="5Fhotkey",
            _=None,
        )

    @pytest.mark.asyncio
    @patch("api.server.router.measurements_for_fingerprint")
    async def test_launchable_when_version_rc_covered(self, measurements):
        measurements.return_value = [{"version": "1.4.0", "rc": False}]

        result = await self._preflight(version="1.4.0", rc=False)

        assert result.launchable is True
        assert result.fingerprint == _profile().fingerprint
        assert "can launch" in result.detail

    @pytest.mark.asyncio
    @patch("api.server.router.measurements_for_fingerprint")
    async def test_not_launchable_when_version_absent(self, measurements):
        measurements.return_value = [{"version": "1.3.0", "rc": False}]

        result = await self._preflight(version="1.4.0", rc=False)

        assert result.launchable is False
        assert "Register it" in result.detail

    @pytest.mark.asyncio
    @patch("api.server.router.measurements_for_fingerprint")
    async def test_rc_must_match_not_just_version(self, measurements):
        # A production measurement (rc=False) does not satisfy a debug image asking about rc=True.
        measurements.return_value = [{"version": "1.4.0", "rc": False}]

        result = await self._preflight(version="1.4.0", rc=True)

        assert result.launchable is False

    @pytest.mark.asyncio
    @patch("api.server.router.measurements_for_fingerprint")
    async def test_oversized_body_is_rejected(self, measurements):
        with pytest.raises(HTTPException) as exc:
            await self._preflight(body=b"x" * (256 * 1024 + 1))

        assert exc.value.status_code == 413
        measurements.assert_not_called()

    def test_route_declares_the_miner_rate_limit_dependency(self):
        params = inspect.signature(tdx_preflight).parameters
        assert isinstance(params["_"].default, DependsMarker)
        assert params["_"].default.dependency.__name__ == "_rate_limit_miner"


def _measurement(fingerprint=None, rc=False, name="8xh200", version="1.4.0"):
    return TeeMeasurementConfig(
        version=version,
        name=name,
        mrtd="A" * 96,
        rtmr0="B" * 96,
        rtmr1="C" * 96,
        rtmr2="D" * 96,
        runtime_rtmr3="E" * 96,
        expected_gpus=["h200"],
        gpu_count=8,
        fingerprint=fingerprint,
        rc=rc,
        authorized_hotkeys=["5Fop"] if rc else [],
    )


class TestMeasurementsForFingerprint:
    """The live (version, rc) join between a host class and the images that can attest on it."""

    @patch("api.server.util.settings")
    def test_matching_measurement_is_returned_as_version_rc(self, mock_settings):
        mock_settings.tee_measurements = [_measurement(fingerprint="a" * 64, version="1.4.0")]

        assert measurements_for_fingerprint("a" * 64) == [{"version": "1.4.0", "rc": False}]

    @patch("api.server.util.settings")
    def test_rc_measurements_are_included(self, mock_settings):
        """
        rc is a property of the version, not the topology -- the caller decides which (version, rc)
        it needs, so an rc entry is offered, not hidden.
        """
        mock_settings.tee_measurements = [
            _measurement(fingerprint="a" * 64, rc=True, version="1.5.0")
        ]

        assert measurements_for_fingerprint("a" * 64) == [{"version": "1.5.0", "rc": True}]

    @patch("api.server.util.settings")
    def test_rc_and_release_of_the_same_version_are_distinct_entries(self, mock_settings):
        mock_settings.tee_measurements = [
            _measurement(fingerprint="a" * 64, rc=True, name="8xh200-rc", version="1.5.0"),
            _measurement(fingerprint="a" * 64, name="8xh200", version="1.5.0"),
        ]

        assert measurements_for_fingerprint("a" * 64) == [
            {"version": "1.5.0", "rc": False},
            {"version": "1.5.0", "rc": True},
        ]

    @patch("api.server.util.settings")
    def test_no_version_floor(self, mock_settings):
        """
        No floor here: the caller compares against its own target version. A fingerprinted entry of
        any version is offered (older ones simply never carry a fingerprint to match).
        """
        mock_settings.tee_measurements = [_measurement(fingerprint="a" * 64, version="1.3.1")]

        assert measurements_for_fingerprint("a" * 64) == [{"version": "1.3.1", "rc": False}]

    @patch("api.server.util.settings")
    def test_unmentioned_topology_is_empty(self, mock_settings):
        mock_settings.tee_measurements = [_measurement(fingerprint="a" * 64)]

        assert measurements_for_fingerprint("b" * 64) == []

    @patch("api.server.util.settings")
    def test_entries_without_a_fingerprint_never_match(self, mock_settings):
        """Entries predating the field load fine; they are just unmatchable."""
        mock_settings.tee_measurements = [_measurement(fingerprint=None)]

        assert measurements_for_fingerprint(_profile().fingerprint) == []

    @patch("api.server.util.settings")
    def test_duplicate_pairs_are_collapsed_and_ordered(self, mock_settings):
        mock_settings.tee_measurements = [
            _measurement(fingerprint="a" * 64, name="dup", version="1.5.0"),
            _measurement(fingerprint="a" * 64, name="8xh200", version="1.4.0"),
            _measurement(fingerprint="a" * 64, name="8xh200-again", version="1.4.0"),
        ]

        assert measurements_for_fingerprint("a" * 64) == [
            {"version": "1.4.0", "rc": False},
            {"version": "1.5.0", "rc": False},
        ]

    @patch("api.server.util.settings")
    def test_reflects_the_config_map_immediately(self, mock_settings):
        """Read live, so a newly published measurement takes effect on ConfigMap remount."""
        mock_settings.tee_measurements = []
        assert measurements_for_fingerprint("a" * 64) == []

        mock_settings.tee_measurements = [_measurement(fingerprint="a" * 64)]
        assert measurements_for_fingerprint("a" * 64) == [{"version": "1.4.0", "rc": False}]


class TestStatusResolution:
    """The monotonic retention status (unknown/pending/accepted), read from the profile row."""

    async def _resolve(self, on_file=False, measured_at=None, stored=True):
        profile = _profile()
        db = MagicMock()
        with (
            patch(
                "api.server.util.host_profile_state",
                AsyncMock(return_value=(on_file, measured_at)),
            ) as state,
            patch(
                "api.server.util.store_host_profile",
                AsyncMock(return_value=(profile.fingerprint, stored)),
            ) as store,
        ):
            result = await resolve_host_profile_status(db=db, profile=profile, hotkey="5Fhotkey")
        return result, store, state

    @pytest.mark.asyncio
    async def test_accepted_from_measured_at(self):
        """measured_at is the retention marker: once the reconciler stamps it the class is accepted,
        for good and version-agnostically (rc counts, since the reconciler counts rc).
        """
        (fingerprint, status, _), _, _ = await self._resolve(
            on_file=True, measured_at="2026-08-01T00:00:00Z"
        )

        assert status == HostProfileStatus.ACCEPTED
        assert fingerprint == _profile().fingerprint

    @pytest.mark.asyncio
    async def test_pending_when_on_file_but_unmeasured(self):
        (_, status, stored), store, _ = await self._resolve(on_file=True)

        assert status == HostProfileStatus.PENDING
        assert stored is True
        store.assert_called_once()

    @pytest.mark.asyncio
    async def test_stored_flag_is_passed_through(self):
        (_, _, stored), _, _ = await self._resolve(on_file=True, stored=False)

        assert stored is False

    @pytest.mark.asyncio
    async def test_a_real_submission_always_stores_and_is_never_unknown(self):
        """A submission always parks the profile, so the row exists and PENDING is the floor."""
        (_, status, _), store, _ = await self._resolve(on_file=True)

        assert status != HostProfileStatus.UNKNOWN
        store.assert_called_once()

    @pytest.mark.asyncio
    async def test_accepted_class_we_do_not_hold_is_still_captured(self):
        """
        A fingerprint cannot be inverted back to its topology inputs, so an accepted host class with
        no stored profile could never have its RTMR0 regenerated. A real submission is the chance to
        capture it; store_host_profile no-ops when the row already exists.
        """
        (_, status, stored), store, _ = await self._resolve(
            on_file=True, measured_at="2026-08-01T00:00:00Z", stored=True
        )

        assert status == HostProfileStatus.ACCEPTED
        assert stored is True
        store.assert_called_once()

    def test_status_helper_reports_unknown_only_without_a_row(self):
        """UNKNOWN is a status-derivation state (no row, no measurement); submission can't reach it
        because it always creates the row, but the helper still distinguishes it."""
        from api.server.util import _host_profile_status

        assert _host_profile_status(False, None) == HostProfileStatus.UNKNOWN
        assert _host_profile_status(True, None) == HostProfileStatus.PENDING
        assert _host_profile_status(True, "2026-08-01T00:00:00Z") == HostProfileStatus.ACCEPTED

    @pytest.mark.asyncio
    async def test_returns_the_model_fingerprint(self):
        """The fingerprint echoed back is HostProfile.fingerprint -- the value that keys the row."""
        profile = _profile()
        with (
            patch(
                "api.server.util.host_profile_state",
                AsyncMock(return_value=(True, "2026-08-01T00:00:00Z")),
            ),
            patch(
                "api.server.util.store_host_profile",
                AsyncMock(return_value=(profile.fingerprint, False)),
            ),
        ):
            fingerprint, status, _ = await resolve_host_profile_status(
                db=MagicMock(), profile=profile, hotkey="5Fhotkey"
            )

        assert fingerprint == profile.fingerprint
        assert status == HostProfileStatus.ACCEPTED


class TestHostProfileStatusEndpoint:
    """POST /servers/tdx/host_profiles/status -- is this host class known, and for what images?

    The version-free gate `chutes-cvm host verify` runs: a miner checks a host before it has
    downloaded any image, so nothing about the answer may depend on a version.
    """

    async def _status(self, body=None):
        body = body if body is not None else json.dumps(SAMPLE_PROFILE)
        return await tdx_host_profile_status(
            request=_mock_request(body),
            profile=_profile(),
            db=MagicMock(),
            hotkey="5Fhotkey",
            _=None,
        )

    @pytest.mark.asyncio
    @patch("api.server.router.measurements_for_fingerprint")
    @patch("api.server.router.host_profile_status", new_callable=AsyncMock)
    async def test_measured_class_lists_its_covered_images(self, status, measurements):
        status.return_value = HostProfileStatus.ACCEPTED
        measurements.return_value = [
            {"version": "1.4.0", "rc": False},
            {"version": "1.4.0", "rc": True},
        ]

        result = await self._status()

        assert result.fingerprint == _profile().fingerprint
        assert result.status == HostProfileStatus.ACCEPTED
        assert [(m.version, m.rc) for m in result.measurements] == [
            ("1.4.0", False),
            ("1.4.0", True),
        ]
        assert "1.4.0 (rc)" in result.detail

    @pytest.mark.asyncio
    @patch("api.server.router.measurements_for_fingerprint")
    @patch("api.server.router.host_profile_status", new_callable=AsyncMock)
    async def test_pending_class_is_told_to_wait_not_resubmit(self, status, measurements):
        status.return_value = HostProfileStatus.PENDING
        measurements.return_value = []

        result = await self._status()

        assert result.measurements == []
        assert "awaiting measurement generation" in result.detail
        assert "POST /servers/tdx/host_profiles" not in result.detail

    @pytest.mark.asyncio
    @patch("api.server.router.measurements_for_fingerprint")
    @patch("api.server.router.host_profile_status", new_callable=AsyncMock)
    async def test_unknown_class_is_told_to_register(self, status, measurements):
        status.return_value = HostProfileStatus.UNKNOWN
        measurements.return_value = []

        result = await self._status()

        assert result.status == HostProfileStatus.UNKNOWN
        assert result.measurements == []
        assert "POST /servers/tdx/host_profiles" in result.detail

    @pytest.mark.asyncio
    @patch("api.server.router.measurements_for_fingerprint")
    @patch("api.server.router.host_profile_status", new_callable=AsyncMock)
    async def test_measurements_win_over_a_lagging_row(self, status, measurements):
        """measured_at is stamped by the reconciler and can lag a publish, so a class can read
        PENDING while measurements already cover it -- the list is the launchability signal."""
        status.return_value = HostProfileStatus.PENDING
        measurements.return_value = [{"version": "1.4.0", "rc": False}]

        result = await self._status()

        assert result.status == HostProfileStatus.PENDING
        assert len(result.measurements) == 1
        assert "is measured" in result.detail

    def test_signature_is_version_free(self):
        """Deliberately version-free -- a caller holding no image can still ask. A version
        parameter creeping in here would reintroduce the image dependency `host verify` had."""
        params = inspect.signature(tdx_host_profile_status).parameters

        assert "version" not in params
        assert "rc" not in params

    @pytest.mark.asyncio
    @patch("api.server.router.measurements_for_fingerprint")
    @patch("api.server.router.host_profile_status", new_callable=AsyncMock)
    async def test_oversized_body_is_rejected_before_any_lookup(self, status, measurements):
        with pytest.raises(HTTPException) as exc:
            await self._status(body=b"x" * (256 * 1024 + 1))

        assert exc.value.status_code == 413
        status.assert_not_called()
        measurements.assert_not_called()

    @pytest.mark.asyncio
    @patch("api.server.router.host_profile_status", new_callable=AsyncMock)
    async def test_lookup_failure_surfaces_as_503(self, status):
        status.side_effect = Exception("db unreachable")

        with pytest.raises(HTTPException) as exc:
            await self._status()

        assert exc.value.status_code == 503
        assert "db unreachable" not in exc.value.detail

    def test_route_declares_the_miner_rate_limit_dependency(self):
        params = inspect.signature(tdx_host_profile_status).parameters
        assert isinstance(params["_"].default, DependsMarker)
        assert params["_"].default.dependency.__name__ == "_rate_limit_miner"
