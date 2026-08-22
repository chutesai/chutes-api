"""
Unit tests for the host profile bucket lifecycle: the public topologies endpoint, promotion of
pending/ -> measured/, config-driven reconcile, and new-submission notification.
"""

import copy
import pytest
import orjson as json
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

import host_profile_reconciler
from api.config import TeeMeasurementConfig
from api.server.router import get_topologies
from api.server.util import (
    list_measured_topologies,
    list_pending_fingerprints,
    promote_host_profile,
    reconcile_host_profiles,
)
from tests.unit.test_host_profile_submission import SAMPLE_PROFILE, _profile

FP_A = "a" * 64
FP_B = "b" * 64


def _measurement(fingerprint=None, rc=False, version="1.4.0", name="8xh200"):
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


def _mock_s3(objects=None, bodies=None):
    """
    S3 double. `objects` maps prefix name ("pending"/"measured") -> list of fingerprints.
    `bodies` maps fingerprint -> stored document (defaults to SAMPLE_PROFILE).
    """
    objects = objects or {}
    bodies = bodies or {}
    s3 = MagicMock()

    async def list_objects_v2(Bucket=None, Prefix=None, ContinuationToken=None):
        which = Prefix.rstrip("/").rsplit("/", 1)[-1]
        contents = [{"Key": f"{Prefix}{fp}.json"} for fp in objects.get(which, [])]
        return {"Contents": contents, "IsTruncated": False}

    async def head_object(Bucket=None, Key=None):
        which = Key.rsplit("/", 2)[-2]
        fingerprint = Key.rsplit("/", 1)[-1][: -len(".json")]
        if fingerprint in objects.get(which, []):
            return {"ETag": "abc"}
        error = Exception("not found")
        error.response = {"Error": {"Code": "404"}}
        raise error

    async def get_object(Bucket=None, Key=None):
        fingerprint = Key.rsplit("/", 1)[-1][: -len(".json")]
        body = MagicMock()
        body.read = AsyncMock(return_value=json.dumps(bodies.get(fingerprint, SAMPLE_PROFILE)))
        return {"Body": body}

    s3.list_objects_v2 = AsyncMock(side_effect=list_objects_v2)
    s3.head_object = AsyncMock(side_effect=head_object)
    s3.get_object = AsyncMock(side_effect=get_object)
    s3.copy_object = AsyncMock()
    s3.delete_object = AsyncMock()
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=s3)
    client.__aexit__ = AsyncMock(return_value=False)
    return client, s3


def _mock_settings(mock_settings, objects=None, bodies=None, measurements=None):
    client, s3 = _mock_s3(objects=objects, bodies=bodies)
    mock_settings.s3_client = MagicMock(return_value=client)
    mock_settings.host_profile_bucket = "chutes"
    mock_settings.host_profile_prefix = "host-profiles"
    mock_settings.tee_measurements = measurements or []
    return s3


class TestListing:
    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_lists_pending_fingerprints(self, mock_settings):
        _mock_settings(mock_settings, objects={"pending": [FP_A, FP_B]})

        assert sorted(await list_pending_fingerprints()) == [FP_A, FP_B]

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_pending_and_measured_are_separate(self, mock_settings):
        _mock_settings(mock_settings, objects={"pending": [FP_A], "measured": [FP_B]})

        assert await list_pending_fingerprints() == [FP_A]
        assert [t["fingerprint"] for t in await list_measured_topologies()] == [FP_B]

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_pagination_is_followed(self, mock_settings):
        """A truncated listing must not silently drop topologies."""
        client, s3 = _mock_s3()
        pages = [
            {
                "Contents": [{"Key": f"host-profiles/measured/{FP_A}.json"}],
                "IsTruncated": True,
                "NextContinuationToken": "next",
            },
            {"Contents": [{"Key": f"host-profiles/measured/{FP_B}.json"}], "IsTruncated": False},
        ]
        s3.list_objects_v2 = AsyncMock(side_effect=pages)
        mock_settings.s3_client = MagicMock(return_value=client)
        mock_settings.host_profile_bucket = "chutes"
        mock_settings.host_profile_prefix = "host-profiles"

        assert [t["fingerprint"] for t in await list_measured_topologies()] == [FP_A, FP_B]
        assert s3.list_objects_v2.await_args.kwargs["ContinuationToken"] == "next"

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_unparseable_object_is_skipped_not_fatal(self, mock_settings):
        """One bad object must not take down the public endpoint."""
        _mock_settings(
            mock_settings,
            objects={"measured": [FP_A, FP_B]},
            bodies={FP_A: {"garbage": True}},
        )

        assert [t["fingerprint"] for t in await list_measured_topologies()] == [FP_B]


class TestPublishedTopology:
    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_machine_identifying_fields_are_stripped(self, mock_settings):
        _mock_settings(mock_settings, objects={"measured": [FP_A]})

        profile = (await list_measured_topologies())[0]["profile"]

        assert "hostname" not in profile
        assert "timestamp" not in profile

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_generic_host_class_data_is_published(self, mock_settings):
        """A verifier needs the full RTMR0 inputs, so everything else stays."""
        _mock_settings(mock_settings, objects={"measured": [FP_A]})

        profile = (await list_measured_topologies())[0]["profile"]

        # Wire shape, not model field names -- a verifier feeds this straight back in.
        assert profile["launch_determinism"]["qemu_version"] == "8.2.2"
        assert profile["gpu"]["pci_device_ids"] == ["2335"]
        assert profile["cpu"]["cpu_processor_id"] == "f26c0000fffba91f"
        assert profile["memory"]["total_gb"] == 2015
        assert profile["numa"]["node_count"] == 2
        assert profile["host"]["bios_version"] == "1.10.2"
        assert "pci_topology" in profile

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_submitter_identity_never_leaks(self, mock_settings):
        """hotkey/nonce/signature live in object metadata, which is never read on this path."""
        s3 = _mock_settings(mock_settings, objects={"measured": [FP_A]})

        published = json.dumps(await list_measured_topologies()).decode()

        for secret in ("hotkey", "nonce", "signature", "5F"):
            assert secret not in published
        # Nothing on this path asks S3 for metadata at all.
        assert s3.head_object.await_count == 0

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_fingerprint_joins_to_the_measurement(self, mock_settings):
        """The published fingerprint is the object key, which is what /tee/measurements carries."""
        _mock_settings(mock_settings, objects={"measured": [FP_A]})

        assert (await list_measured_topologies())[0]["fingerprint"] == FP_A


class TestTopologiesEndpoint:
    @pytest.mark.asyncio
    @patch("api.server.router.list_measured_topologies", new_callable=AsyncMock)
    @patch("api.server.router.settings")
    async def test_returns_and_caches(self, mock_settings, listing):
        mock_settings.redis_client.get = AsyncMock(return_value=None)
        mock_settings.redis_client.set = AsyncMock()
        listing.return_value = [{"fingerprint": FP_A, "profile": {"gpu": {"count": 8}}}]

        result = await get_topologies(_=None)

        assert result[0]["fingerprint"] == FP_A
        key, payload = mock_settings.redis_client.set.await_args.args
        assert key == "tdx_topologies"
        assert json.loads(payload)[0]["fingerprint"] == FP_A

    @pytest.mark.asyncio
    @patch("api.server.router.list_measured_topologies", new_callable=AsyncMock)
    @patch("api.server.router.settings")
    async def test_cache_hit_skips_the_bucket(self, mock_settings, listing):
        cached = [{"fingerprint": FP_A, "profile": {}}]
        mock_settings.redis_client.get = AsyncMock(return_value=json.dumps(cached))

        assert await get_topologies(_=None) == cached
        listing.assert_not_called()

    @pytest.mark.asyncio
    @patch("api.server.router.list_measured_topologies", new_callable=AsyncMock)
    @patch("api.server.router.settings")
    async def test_bucket_failure_is_503(self, mock_settings, listing):
        mock_settings.redis_client.get = AsyncMock(return_value=None)
        listing.side_effect = Exception("bucket unreachable")

        with pytest.raises(HTTPException) as exc:
            await get_topologies(_=None)

        assert exc.value.status_code == 503
        assert "bucket unreachable" not in exc.value.detail

    def test_route_is_public_and_rate_limited(self):
        """Unauthenticated by design, so it carries the anonymous cap."""
        import inspect
        from fastapi.params import Depends as DependsMarker

        params = inspect.signature(get_topologies).parameters
        assert list(params) == ["_"]
        assert isinstance(params["_"].default, DependsMarker)
        assert params["_"].default.dependency.__name__ == "_rate_limit"


class TestPromotion:
    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_moves_pending_to_measured(self, mock_settings):
        s3 = _mock_settings(mock_settings, objects={"pending": [FP_A]})

        assert await promote_host_profile(FP_A) is True
        copy_kwargs = s3.copy_object.await_args.kwargs
        assert copy_kwargs["Key"] == f"host-profiles/measured/{FP_A}.json"
        assert copy_kwargs["CopySource"]["Key"] == f"host-profiles/pending/{FP_A}.json"
        assert s3.delete_object.await_args.kwargs["Key"] == f"host-profiles/pending/{FP_A}.json"

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_is_idempotent(self, mock_settings):
        """Already promoted (or never submitted): a no-op, not an error."""
        s3 = _mock_settings(mock_settings, objects={"measured": [FP_A]})

        assert await promote_host_profile(FP_A) is False
        s3.copy_object.assert_not_called()
        s3.delete_object.assert_not_called()

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_copies_before_deleting(self, mock_settings):
        """A crash between the two must leave the object in both prefixes, never in neither."""
        order = []
        s3 = _mock_settings(mock_settings, objects={"pending": [FP_A]})
        s3.copy_object = AsyncMock(side_effect=lambda **kw: order.append("copy"))
        s3.delete_object = AsyncMock(side_effect=lambda **kw: order.append("delete"))

        await promote_host_profile(FP_A)

        assert order == ["copy", "delete"]


class TestReconcile:
    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_promotes_published_fingerprints(self, mock_settings):
        _mock_settings(
            mock_settings,
            objects={"pending": [FP_A, FP_B]},
            measurements=[_measurement(fingerprint=FP_A)],
        )

        assert await reconcile_host_profiles() == [FP_A]

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_leaves_unpublished_profiles_pending(self, mock_settings):
        s3 = _mock_settings(mock_settings, objects={"pending": [FP_A]}, measurements=[])

        assert await reconcile_host_profiles() == []
        s3.copy_object.assert_not_called()

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_rc_measurements_still_promote(self, mock_settings):
        """
        Reconcile follows the config, not attestability: an rc entry means the topology WAS
        generated, so its profile belongs in the retained set even though it cannot launch yet.
        """
        _mock_settings(
            mock_settings,
            objects={"pending": [FP_A]},
            measurements=[_measurement(fingerprint=FP_A, rc=True)],
        )

        assert await reconcile_host_profiles() == [FP_A]

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_entries_without_a_fingerprint_are_ignored(self, mock_settings):
        _mock_settings(mock_settings, objects={"pending": [FP_A]}, measurements=[_measurement()])

        assert await reconcile_host_profiles() == []

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_second_run_is_a_no_op(self, mock_settings):
        """Idempotent: the object has moved, so the next run promotes nothing."""
        _mock_settings(
            mock_settings,
            objects={"measured": [FP_A]},
            measurements=[_measurement(fingerprint=FP_A)],
        )

        assert await reconcile_host_profiles() == []


class TestNotify:
    def _redis(self, already=()):
        redis = MagicMock()
        redis.sismember = AsyncMock(side_effect=lambda key, fp: fp in already)
        redis.sadd = AsyncMock()
        return redis

    @pytest.mark.asyncio
    @patch("host_profile_reconciler.send_discord_alert", new_callable=AsyncMock)
    @patch("host_profile_reconciler.list_pending_fingerprints", new_callable=AsyncMock)
    @patch("host_profile_reconciler.settings")
    async def test_alerts_once_per_new_submission(self, mock_settings, pending, alert):
        mock_settings.redis_client = self._redis()
        mock_settings.host_profile_prefix = "host-profiles"
        pending.return_value = [FP_A, FP_B]
        alert.return_value = True

        assert await host_profile_reconciler.notify_new_submissions() == [FP_A, FP_B]
        assert alert.await_count == 2
        assert FP_A in alert.await_args_list[0].args[0]

    @pytest.mark.asyncio
    @patch("host_profile_reconciler.send_discord_alert", new_callable=AsyncMock)
    @patch("host_profile_reconciler.list_pending_fingerprints", new_callable=AsyncMock)
    @patch("host_profile_reconciler.settings")
    async def test_already_notified_is_skipped(self, mock_settings, pending, alert):
        mock_settings.redis_client = self._redis(already={FP_A})
        mock_settings.host_profile_prefix = "host-profiles"
        pending.return_value = [FP_A, FP_B]
        alert.return_value = True

        assert await host_profile_reconciler.notify_new_submissions() == [FP_B]
        assert alert.await_count == 1

    @pytest.mark.asyncio
    @patch("host_profile_reconciler.send_discord_alert", new_callable=AsyncMock)
    @patch("host_profile_reconciler.list_pending_fingerprints", new_callable=AsyncMock)
    @patch("host_profile_reconciler.settings")
    async def test_failed_alert_is_retried_next_run(self, mock_settings, pending, alert):
        """Marking before the webhook succeeds would silently swallow the only notification."""
        redis = self._redis()
        mock_settings.redis_client = redis
        mock_settings.host_profile_prefix = "host-profiles"
        pending.return_value = [FP_A]
        alert.return_value = False

        await host_profile_reconciler.notify_new_submissions()

        redis.sadd.assert_not_called()

    @pytest.mark.asyncio
    @patch("host_profile_reconciler.send_discord_alert", new_callable=AsyncMock)
    @patch("host_profile_reconciler.list_pending_fingerprints", new_callable=AsyncMock)
    @patch("host_profile_reconciler.settings")
    async def test_never_triggers_generation(self, mock_settings, pending, alert):
        """The job notifies and reconciles only -- a human decides when to generate."""
        mock_settings.redis_client = self._redis()
        mock_settings.host_profile_prefix = "host-profiles"
        pending.return_value = [FP_A]
        alert.return_value = True

        await host_profile_reconciler.notify_new_submissions()

        body = alert.await_args.args[0]
        assert "Awaiting measurement generation" in body


class TestProfileIsUnchangedByPublication:
    """Publication must not alter the document the fingerprint was computed from."""

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_published_profile_still_fingerprints_the_same(self, mock_settings):
        _mock_settings(mock_settings, objects={"measured": [FP_A]})
        from api.server.schemas import HostProfile

        published = (await list_measured_topologies())[0]["profile"]
        restored = HostProfile(**copy.deepcopy(published))

        assert restored.fingerprint == _profile().fingerprint
