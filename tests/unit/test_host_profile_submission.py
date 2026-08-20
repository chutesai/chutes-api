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

from api.rate_limit import rate_limit_miner
from api.server.router import submit_host_profile
from api.server.schemas import HostProfile
from api.server.util import store_host_profile


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


def _mock_s3(exists: bool):
    s3 = MagicMock()
    if exists:
        s3.head_object = AsyncMock(return_value={"ETag": "abc"})
    else:
        error = Exception("not found")
        error.response = {"Error": {"Code": "404"}}
        s3.head_object = AsyncMock(side_effect=error)
    s3.put_object = AsyncMock()
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=s3)
    client.__aexit__ = AsyncMock(return_value=False)
    return client, s3


def _mock_storage_settings(mock_settings, bucket="host-profiles-bucket", exists=False):
    client, s3 = _mock_s3(exists=exists)
    mock_settings.s3_client = MagicMock(return_value=client)
    mock_settings.host_profile_bucket = bucket
    mock_settings.host_profile_prefix = "host-profiles"
    return s3


class TestStorage:
    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_new_profile_is_written_verbatim(self, mock_settings):
        s3 = _mock_storage_settings(mock_settings)
        profile = _profile()
        raw = json.dumps(SAMPLE_PROFILE)

        fingerprint, stored = await store_host_profile(
            profile=profile,
            raw_body=raw,
            hotkey="5Fhotkey",
            nonce="1755690000",
            signature="ab" * 64,
        )

        assert stored is True
        assert fingerprint == profile.fingerprint
        kwargs = s3.put_object.call_args.kwargs
        assert kwargs["Bucket"] == "host-profiles-bucket"
        assert kwargs["Key"] == f"host-profiles/{profile.fingerprint}.json"
        assert kwargs["Body"] == raw
        assert kwargs["Metadata"]["hotkey"] == "5Fhotkey"
        assert kwargs["Metadata"]["signature"] == "ab" * 64
        assert kwargs["Metadata"]["fingerprint"] == profile.fingerprint
        assert kwargs["Metadata"]["gpu-count"] == "8"

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_key_comes_from_the_model_fingerprint(self, mock_settings):
        """The storage key is the model's fingerprint -- no second implementation to drift."""
        s3 = _mock_storage_settings(mock_settings)
        profile = _profile(gpu={"count": 4, "pci_device_ids": ["2901"]})

        await store_host_profile(
            profile=profile,
            raw_body=b"{}",
            hotkey="5Fhotkey",
            nonce="1755690000",
            signature="ab" * 64,
        )

        assert s3.put_object.call_args.kwargs["Key"] == f"host-profiles/{profile.fingerprint}.json"

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_falls_back_to_storage_bucket(self, mock_settings):
        s3 = _mock_storage_settings(mock_settings, bucket=None)
        mock_settings.storage_bucket = "chutes"

        await store_host_profile(
            profile=_profile(),
            raw_body=b"{}",
            hotkey="5Fhotkey",
            nonce="1755690000",
            signature="ab" * 64,
        )

        assert s3.put_object.call_args.kwargs["Bucket"] == "chutes"

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_existing_profile_is_not_overwritten(self, mock_settings):
        s3 = _mock_storage_settings(mock_settings, exists=True)

        fingerprint, stored = await store_host_profile(
            profile=_profile(),
            raw_body=b"{}",
            hotkey="5Fother",
            nonce="1755690000",
            signature="ab" * 64,
        )

        assert stored is False
        assert fingerprint == _profile().fingerprint
        s3.put_object.assert_not_called()

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_unexpected_s3_error_propagates(self, mock_settings):
        s3 = _mock_storage_settings(mock_settings)
        error = Exception("access denied")
        error.response = {"Error": {"Code": "403"}}
        s3.head_object = AsyncMock(side_effect=error)

        with pytest.raises(Exception, match="access denied"):
            await store_host_profile(
                profile=_profile(),
                raw_body=b"{}",
                hotkey="5Fhotkey",
                nonce="1755690000",
                signature="ab" * 64,
            )
        s3.put_object.assert_not_called()


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

    async def _submit(self, body=None):
        body = body if body is not None else json.dumps(SAMPLE_PROFILE)
        return await submit_host_profile(
            request=_mock_request(body),
            profile=_profile(),
            hotkey="5Fhotkey",
            nonce="1755690000",
            signature="ab" * 64,
            _=None,
        )

    @pytest.mark.asyncio
    @patch("api.server.router.store_host_profile", new_callable=AsyncMock)
    async def test_accepts_new_profile(self, store):
        store.return_value = (_profile().fingerprint, True)

        result = await self._submit()

        assert result.stored is True
        assert result.fingerprint == _profile().fingerprint
        # The exact signed bytes are stored, not a re-serialization of the parsed model.
        assert store.await_args.kwargs["raw_body"] == json.dumps(SAMPLE_PROFILE)
        assert store.await_args.kwargs["hotkey"] == "5Fhotkey"
        assert store.await_args.kwargs["signature"] == "ab" * 64

    @pytest.mark.asyncio
    @patch("api.server.router.store_host_profile", new_callable=AsyncMock)
    async def test_duplicate_submission_is_a_no_op(self, store):
        store.return_value = (_profile().fingerprint, False)

        result = await self._submit()

        assert result.stored is False
        assert "already submitted" in result.detail

    @pytest.mark.asyncio
    @patch("api.server.router.store_host_profile", new_callable=AsyncMock)
    async def test_oversized_body_is_rejected(self, store):
        with pytest.raises(HTTPException) as exc:
            await self._submit(body=b"x" * (256 * 1024 + 1))

        assert exc.value.status_code == 413
        store.assert_not_called()

    @pytest.mark.asyncio
    @patch("api.server.router.store_host_profile", new_callable=AsyncMock)
    async def test_storage_failure_surfaces_as_503(self, store):
        store.side_effect = Exception("bucket unreachable")

        with pytest.raises(HTTPException) as exc:
            await self._submit()

        assert exc.value.status_code == 503
        assert "bucket unreachable" not in exc.value.detail

    def test_route_declares_the_miner_rate_limit_dependency(self):
        """The metering is visible at the route, not buried in the handler body."""
        params = inspect.signature(submit_host_profile).parameters
        assert isinstance(params["_"].default, DependsMarker)
        assert params["_"].default.dependency.__name__ == "_rate_limit_miner"
