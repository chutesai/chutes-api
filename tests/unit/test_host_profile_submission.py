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
from api.server.router import submit_host_profile
from api.server.schemas import HostProfile
from api.server.util import (
    host_profile_is_known,
    resolve_host_profile_status,
    store_host_profile,
    host_profile_measurement_status,
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
    "numa": {"node_count": 2, "nodes": [0, 1], "cpus_per_node": {"0": "0-47", "1": "48-95"}},
    "nic": {
        "ib_class_count": 8,
        "eth_class_count": 2,
        "ib_devices": ["0000:1a:00.0"],
        "bridge_pfs": [],
        "passthrough_candidates": ["0000:1a:00.0", "0000:42:00.0"],
        "passthrough_numa_nodes": [0, 1],
    },
    "nvswitch": {"present": True, "count": 4, "devices": ["0000:0a:00.0"], "numa_nodes": [0]},
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
            host={"board_name": "PowerEdge XE9680 (Rev. 1.2)", "bios_date": "04/01/2026"},
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
    db.execute = AsyncMock(return_value=result)
    db.scalar = AsyncMock(return_value="fp" if known else None)
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
        assert check.await_args_list[0].kwargs == {"window_seconds": 3600, "identity": "5Fhotkey"}

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

    async def _submit(self, body=None, dry_run=False):
        body = body if body is not None else json.dumps(SAMPLE_PROFILE)
        return await submit_host_profile(
            request=_mock_request(body),
            profile=_profile(),
            db=MagicMock(),
            hotkey="5Fhotkey",
            dry_run=dry_run,
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
        assert resolve.await_args.kwargs["dry_run"] is False

    @pytest.mark.asyncio
    @patch("api.server.router.resolve_host_profile_status", new_callable=AsyncMock)
    async def test_duplicate_submission_is_a_no_op(self, resolve):
        resolve.return_value = (_profile().fingerprint, HostProfileStatus.PENDING, False)

        result = await self._submit()

        assert result.stored is False
        assert result.status == HostProfileStatus.PENDING

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "profile_status, expect",
        [
            (HostProfileStatus.ACCEPTED, "can launch"),
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
    async def test_dry_run_is_passed_through(self, resolve):
        resolve.return_value = (_profile().fingerprint, HostProfileStatus.UNKNOWN, False)

        result = await self._submit(dry_run=True)

        assert resolve.await_args.kwargs["dry_run"] is True
        assert result.stored is False

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
        authorized_signing_keys=["pem"] if rc else [],
    )


class TestMeasurementStatus:
    """What the measurement config says about a host class -- the sole truth for attestability."""

    @patch("api.server.util.settings")
    def test_published_measurement_is_accepted(self, mock_settings):
        mock_settings.tee_measurements = [_measurement(fingerprint="a" * 64)]

        assert host_profile_measurement_status("a" * 64) is HostProfileStatus.ACCEPTED

    @patch("api.server.util.settings")
    def test_rc_only_measurement_does_not_qualify(self, mock_settings):
        """
        rc is a property of a VERSION, not a topology -- there is no measured-but-rc state. An
        rc-only fingerprint returns None here and reports PENDING off the bucket instead.
        """
        mock_settings.tee_measurements = [_measurement(fingerprint="a" * 64, rc=True)]

        assert host_profile_measurement_status("a" * 64) is None

    @patch("api.server.util.settings")
    def test_published_wins_over_rc_for_the_same_topology(self, mock_settings):
        """A topology carried by both an rc and a published measurement is attestable."""
        mock_settings.tee_measurements = [
            _measurement(fingerprint="a" * 64, rc=True, name="8xh200-rc"),
            _measurement(fingerprint="a" * 64),
        ]

        assert host_profile_measurement_status("a" * 64) is HostProfileStatus.ACCEPTED

    @patch("api.server.util.settings")
    def test_unmentioned_topology_has_no_status(self, mock_settings):
        mock_settings.tee_measurements = [_measurement(fingerprint="a" * 64)]

        assert host_profile_measurement_status("b" * 64) is None

    @patch("api.server.util.settings")
    def test_entries_without_a_fingerprint_never_match(self, mock_settings):
        """Backward compat: entries predating the field load fine, they are just unmatchable."""
        mock_settings.tee_measurements = [_measurement(fingerprint=None)]

        assert host_profile_measurement_status(_profile().fingerprint) is None

    @pytest.mark.parametrize("version", ["1.3.0", "1.3.1", "0.9.0"])
    @patch("api.server.util.settings")
    def test_pre_1_4_0_measurements_do_not_count(self, mock_settings, version):
        """
        The caller runs the 1.4.0 CLI. A host class measured only on an older version cannot
        launch for them, so reporting accepted off that entry would be a lie -- and it is why the
        24 pre-1.4.0 entries need no fingerprint backfill.
        """
        mock_settings.tee_measurements = [_measurement(fingerprint="a" * 64, version=version)]

        assert host_profile_measurement_status("a" * 64) is None

    @patch("api.server.util.settings")
    def test_1_4_0_and_newer_count(self, mock_settings):
        mock_settings.tee_measurements = [_measurement(fingerprint="a" * 64, version="1.4.0")]
        assert host_profile_measurement_status("a" * 64) is HostProfileStatus.ACCEPTED

        mock_settings.tee_measurements = [_measurement(fingerprint="a" * 64, version="1.5.2")]
        assert host_profile_measurement_status("a" * 64) is HostProfileStatus.ACCEPTED

    @patch("api.server.util.settings")
    def test_gated_old_entry_does_not_mask_a_current_one(self, mock_settings):
        """A topology carried by both an ignored 1.3.1 entry and a 1.4.0 entry is accepted."""
        mock_settings.tee_measurements = [
            _measurement(fingerprint="a" * 64, version="1.3.1", name="8xh200-old"),
            _measurement(fingerprint="a" * 64, version="1.4.0"),
        ]

        assert host_profile_measurement_status("a" * 64) is HostProfileStatus.ACCEPTED

    @patch("api.server.util.settings")
    def test_malformed_version_is_treated_as_too_old(self, mock_settings):
        """semcomp floors an unparseable version at 0.0.0, so it fails the gate rather than raising."""
        mock_settings.tee_measurements = [
            _measurement(fingerprint="a" * 64, version="not-a-version")
        ]

        assert host_profile_measurement_status("a" * 64) is None

    @patch("api.server.util.settings")
    def test_reflects_the_config_map_immediately(self, mock_settings):
        """Read live, so a newly published measurement takes effect on ConfigMap remount."""
        mock_settings.tee_measurements = []
        assert host_profile_measurement_status("a" * 64) is None

        mock_settings.tee_measurements = [_measurement(fingerprint="a" * 64)]
        assert host_profile_measurement_status("a" * 64) is HostProfileStatus.ACCEPTED


class TestStatusResolution:
    """accepted / pending / unknown, and what each does to the bucket."""

    async def _resolve(self, dry_run=False, config_status=None, known=False, stored=True):
        profile = _profile()
        db = MagicMock()
        with (
            patch(
                "api.server.util.host_profile_measurement_status",
                MagicMock(return_value=config_status),
            ),
            patch("api.server.util.host_profile_is_known", AsyncMock(return_value=known)) as head,
            patch(
                "api.server.util.store_host_profile",
                AsyncMock(return_value=(profile.fingerprint, stored)),
            ) as store,
        ):
            result = await resolve_host_profile_status(
                db=db, profile=profile, hotkey="5Fhotkey", dry_run=dry_run
            )
        return result, store, head

    @pytest.mark.asyncio
    async def test_accepted_comes_from_the_config(self):
        (fingerprint, profile_status, _), _, head = await self._resolve(
            config_status=HostProfileStatus.ACCEPTED
        )

        assert profile_status == HostProfileStatus.ACCEPTED
        assert fingerprint == _profile().fingerprint
        # Status never depends on the bucket when the config has an answer.
        head.assert_not_called()

    @pytest.mark.asyncio
    async def test_rc_only_topology_falls_through_to_the_bucket(self):
        """
        host_profile_measurement_status returns None for an rc-only fingerprint, so the bucket decides:
        held -> PENDING. There is no separate measured-but-rc status.
        """
        (_, profile_status, _), _, _ = await self._resolve(config_status=None, known=True)

        assert profile_status == HostProfileStatus.PENDING

    @pytest.mark.asyncio
    async def test_accepted_is_not_derived_from_the_measured_prefix(self):
        """
        An object under measured/ means generated, NOT attestable. Only the config can say
        accepted, or a measured-but-rc topology would wrongly report it.
        """
        (_, profile_status, _), _, _ = await self._resolve(config_status=None, known=True)

        assert profile_status == HostProfileStatus.PENDING

    @pytest.mark.asyncio
    async def test_pending_when_stored(self):
        (_, profile_status, stored), store, _ = await self._resolve()

        assert profile_status == HostProfileStatus.PENDING
        assert stored is True
        store.assert_called_once()

    @pytest.mark.asyncio
    async def test_pending_when_already_known(self):
        (_, profile_status, stored), _, _ = await self._resolve(stored=False)

        assert profile_status == HostProfileStatus.PENDING
        assert stored is False

    @pytest.mark.asyncio
    async def test_a_real_submission_is_never_unknown(self):
        """Without dry_run the profile gets parked, so pending is the floor."""
        (_, profile_status, _), _, _ = await self._resolve(dry_run=False, known=False)

        assert profile_status != HostProfileStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_measured_topology_we_do_not_hold_is_still_captured(self):
        """
        A fingerprint cannot be inverted back to its topology inputs, so an accepted host class
        with no stored profile could never have its RTMR0 regenerated. A real submission is the
        chance to capture it; store_host_profile no-ops when either prefix already holds it.
        """
        (_, profile_status, stored), store, _ = await self._resolve(
            config_status=HostProfileStatus.ACCEPTED, stored=True
        )

        assert profile_status == HostProfileStatus.ACCEPTED
        assert stored is True
        store.assert_called_once()

    @pytest.mark.asyncio
    async def test_dry_run_unknown_when_neither(self):
        (_, profile_status, stored), store, _ = await self._resolve(dry_run=True, known=False)

        assert profile_status == HostProfileStatus.UNKNOWN
        assert stored is False
        store.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("known", [True, False])
    async def test_dry_run_never_writes(self, known):
        _, store, _ = await self._resolve(dry_run=True, known=known)

        store.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_pending_when_bucket_has_it(self):
        (_, profile_status, stored), store, _ = await self._resolve(dry_run=True, known=True)

        assert profile_status == HostProfileStatus.PENDING
        assert stored is False
        store.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_accepted_does_not_touch_the_bucket(self):
        (_, profile_status, _), store, head = await self._resolve(
            dry_run=True, config_status=HostProfileStatus.ACCEPTED
        )

        assert profile_status == HostProfileStatus.ACCEPTED
        store.assert_not_called()
        head.assert_not_called()

    @pytest.mark.asyncio
    async def test_matching_uses_the_model_fingerprint(self):
        """
        The value matched against measurements is HostProfile.fingerprint -- the same value that
        keys the bucket object. A second implementation here would silently break matching.
        """
        profile = _profile()
        seen = {}

        def _status(fingerprint):
            seen["asked"] = fingerprint
            return HostProfileStatus.ACCEPTED

        with (
            patch("api.server.util.host_profile_measurement_status", _status),
            patch(
                "api.server.util.store_host_profile",
                AsyncMock(return_value=(profile.fingerprint, False)),
            ),
        ):
            fingerprint, profile_status, _ = await resolve_host_profile_status(
                db=MagicMock(), profile=profile, hotkey="5Fhotkey"
            )

        assert seen["asked"] == profile.fingerprint
        assert fingerprint == profile.fingerprint
        assert profile_status == HostProfileStatus.ACCEPTED
