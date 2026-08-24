"""
Unit tests for the host profile lifecycle: the public listing endpoint, config-driven reconcile,
retention, and new-submission notification.
"""

import copy
import pytest
import orjson as json
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

import host_profile_reconciler
from api.config import TeeMeasurementConfig
from api.server.router import list_host_profiles
from api.server.schemas import HostProfile
from api.server.util import (
    list_measured_host_profiles,
    list_pending_profiles,
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


def _record(fingerprint=FP_A, profile=None, notified_at=None, measured_at=None):
    record = MagicMock()
    record.fingerprint = fingerprint
    record.profile = copy.deepcopy(profile if profile is not None else SAMPLE_PROFILE)
    record.notified_at = notified_at
    record.measured_at = measured_at
    return record


def _session(rows=None, scalars=None):
    """
    AsyncSession double. `rows` feeds result.all() (the (fingerprint, profile) select);
    `scalars` feeds result.scalars().all() (RETURNING and the ORM select).
    """
    db = MagicMock()
    result = MagicMock()
    result.all = MagicMock(return_value=rows or [])
    scalar_result = MagicMock()
    scalar_result.all = MagicMock(return_value=scalars or [])
    result.scalars = MagicMock(return_value=scalar_result)
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()
    return db


class TestPublishedProfile:
    @pytest.mark.asyncio
    async def test_machine_identifying_fields_are_stripped(self):
        db = _session(rows=[(FP_A, copy.deepcopy(SAMPLE_PROFILE))])

        profile = (await list_measured_host_profiles(db))[0]["profile"]

        assert "hostname" not in profile
        assert "timestamp" not in profile

    @pytest.mark.asyncio
    async def test_generic_host_class_data_is_published(self):
        """A verifier needs the full RTMR0 inputs, so everything else stays."""
        db = _session(rows=[(FP_A, copy.deepcopy(SAMPLE_PROFILE))])

        profile = (await list_measured_host_profiles(db))[0]["profile"]

        # Wire shape, not model field names -- a verifier feeds this straight back in.
        assert profile["launch_determinism"]["qemu_version"] == "8.2.2"
        assert profile["gpu"]["pci_device_ids"] == ["2335"]
        assert profile["cpu"]["cpu_processor_id"] == "f26c0000fffba91f"
        assert profile["memory"]["total_gb"] == 2015
        assert profile["numa"]["node_count"] == 2
        assert profile["host"]["bios_version"] == "1.10.2"
        assert "pci_topology" in profile

    @pytest.mark.asyncio
    async def test_submitter_identity_never_leaks(self):
        """hotkey/nonce/signature are columns this query never selects."""
        db = _session(rows=[(FP_A, copy.deepcopy(SAMPLE_PROFILE))])

        published = json.dumps(await list_measured_host_profiles(db)).decode()

        for secret in ("hotkey", "nonce", "signature", "5F"):
            assert secret not in published

    @pytest.mark.asyncio
    async def test_fingerprint_joins_to_the_measurement(self):
        db = _session(rows=[(FP_A, copy.deepcopy(SAMPLE_PROFILE))])

        assert (await list_measured_host_profiles(db))[0]["fingerprint"] == FP_A

    @pytest.mark.asyncio
    async def test_published_profile_still_fingerprints_the_same(self):
        """Publication must not alter the document the fingerprint was computed from."""
        db = _session(rows=[(FP_A, _profile().model_dump(by_alias=True))])

        published = (await list_measured_host_profiles(db))[0]["profile"]

        assert HostProfile(**published).fingerprint == _profile().fingerprint


class TestPublishedHostProfilesEndpoint:
    @pytest.mark.asyncio
    @patch("api.server.router.list_measured_host_profiles", new_callable=AsyncMock)
    @patch("api.server.router.settings")
    async def test_returns_and_caches(self, mock_settings, listing):
        mock_settings.redis_client.get = AsyncMock(return_value=None)
        mock_settings.redis_client.set = AsyncMock()
        listing.return_value = [{"fingerprint": FP_A, "profile": {"gpu": {"count": 8}}}]

        result = await list_host_profiles(db=MagicMock(), _=None)

        assert result[0]["fingerprint"] == FP_A
        key, payload = mock_settings.redis_client.set.await_args.args
        assert key == "tdx_host_profiles"
        assert json.loads(payload)[0]["fingerprint"] == FP_A

    @pytest.mark.asyncio
    @patch("api.server.router.list_measured_host_profiles", new_callable=AsyncMock)
    @patch("api.server.router.settings")
    async def test_cache_hit_skips_the_query(self, mock_settings, listing):
        cached = [{"fingerprint": FP_A, "profile": {}}]
        mock_settings.redis_client.get = AsyncMock(return_value=json.dumps(cached))

        assert await list_host_profiles(db=MagicMock(), _=None) == cached
        listing.assert_not_called()

    @pytest.mark.asyncio
    @patch("api.server.router.list_measured_host_profiles", new_callable=AsyncMock)
    @patch("api.server.router.settings")
    async def test_query_failure_is_503(self, mock_settings, listing):
        mock_settings.redis_client.get = AsyncMock(return_value=None)
        listing.side_effect = Exception("db unreachable")

        with pytest.raises(HTTPException) as exc:
            await list_host_profiles(db=MagicMock(), _=None)

        assert exc.value.status_code == 503
        assert "db unreachable" not in exc.value.detail

    def test_route_is_public_and_rate_limited(self):
        """Unauthenticated by design, so it carries the anonymous cap."""
        import inspect
        from fastapi.params import Depends as DependsMarker

        params = inspect.signature(list_host_profiles).parameters
        assert "hotkey" not in params
        assert isinstance(params["_"].default, DependsMarker)
        assert params["_"].default.dependency.__name__ == "_rate_limit"


class TestReconcile:
    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_promotes_published_fingerprints(self, mock_settings):
        mock_settings.tee_measurements = [_measurement(fingerprint=FP_A)]
        db = _session(scalars=[FP_A])

        assert await reconcile_host_profiles(db) == [FP_A]
        db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_no_published_fingerprints_is_a_no_op(self, mock_settings):
        mock_settings.tee_measurements = [_measurement()]
        db = _session()

        assert await reconcile_host_profiles(db) == []
        db.execute.assert_not_called()

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_rc_measurements_still_promote(self, mock_settings):
        """
        Reconcile follows the config, not attestability: an rc entry means the topology WAS
        generated, so its profile belongs in the retained set even though it cannot launch yet.
        """
        mock_settings.tee_measurements = [_measurement(fingerprint=FP_A, rc=True)]
        db = _session(scalars=[FP_A])

        assert await reconcile_host_profiles(db) == [FP_A]

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_only_pending_rows_are_touched(self, mock_settings):
        """Already-measured rows must not have measured_at rewritten on every run."""
        mock_settings.tee_measurements = [_measurement(fingerprint=FP_A)]
        db = _session(scalars=[])

        assert await reconcile_host_profiles(db) == []
        clause = str(db.execute.await_args.args[0])
        assert "measured_at IS NULL" in clause

    @pytest.mark.asyncio
    @patch("api.server.util.settings")
    async def test_second_run_promotes_nothing(self, mock_settings):
        mock_settings.tee_measurements = [_measurement(fingerprint=FP_A)]
        db = _session(scalars=[])

        assert await reconcile_host_profiles(db) == []
        db.commit.assert_not_called()


class TestNothingIsEverDeleted:
    """
    Retention is the point of the table: a profile nobody generated for is still the record that
    someone asked, and a measured row is the only copy of its topology's inputs (a fingerprint
    cannot be inverted). Neither is ever removed.
    """

    def test_no_expiry_helper_exists(self):
        import api.server.util as util

        assert not hasattr(util, "expire_pending_profiles")

    @pytest.mark.asyncio
    @patch("host_profile_reconciler.reconcile_host_profiles", new_callable=AsyncMock)
    @patch("host_profile_reconciler.notify_new_submissions", new_callable=AsyncMock)
    @patch("host_profile_reconciler.get_session")
    async def test_the_job_only_notifies_and_reconciles(self, session, notify, reconcile):
        """No third step: nothing on this schedule removes a profile."""
        db = _session()
        session.return_value.__aenter__ = AsyncMock(return_value=db)
        session.return_value.__aexit__ = AsyncMock(return_value=False)
        notify.return_value = []
        reconcile.return_value = []

        await host_profile_reconciler.main()

        notify.assert_awaited_once()
        reconcile.assert_awaited_once()
        # A delete would have to go through the session; nothing issued one.
        db.execute.assert_not_called()


class TestPendingListing:
    @pytest.mark.asyncio
    async def test_lists_pending_only(self):
        db = _session(scalars=[_record(FP_A), _record(FP_B)])

        records = await list_pending_profiles(db)

        assert [r.fingerprint for r in records] == [FP_A, FP_B]
        assert "measured_at IS NULL" in str(db.execute.await_args.args[0])


class TestNotify:
    @pytest.mark.asyncio
    @patch("host_profile_reconciler.send_discord_alert", new_callable=AsyncMock)
    @patch("host_profile_reconciler.list_pending_profiles", new_callable=AsyncMock)
    async def test_alerts_once_per_new_submission(self, pending, alert):
        pending.return_value = [_record(FP_A), _record(FP_B)]
        alert.return_value = True

        assert await host_profile_reconciler.notify_new_submissions(_session()) == [FP_A, FP_B]
        assert alert.await_count == 2
        assert FP_A in alert.await_args_list[0].args[0]

    @pytest.mark.asyncio
    @patch("host_profile_reconciler.send_discord_alert", new_callable=AsyncMock)
    @patch("host_profile_reconciler.list_pending_profiles", new_callable=AsyncMock)
    async def test_already_notified_is_skipped(self, pending, alert):
        pending.return_value = [_record(FP_A, notified_at="2026-08-21"), _record(FP_B)]
        alert.return_value = True

        assert await host_profile_reconciler.notify_new_submissions(_session()) == [FP_B]
        assert alert.await_count == 1

    @pytest.mark.asyncio
    @patch("host_profile_reconciler.send_discord_alert", new_callable=AsyncMock)
    @patch("host_profile_reconciler.list_pending_profiles", new_callable=AsyncMock)
    async def test_failed_alert_is_retried_next_run(self, pending, alert):
        """Stamping before the webhook succeeds would swallow the only notification."""
        pending.return_value = [_record(FP_A)]
        alert.return_value = False
        db = _session()

        assert await host_profile_reconciler.notify_new_submissions(db) == []
        db.execute.assert_not_called()

    @pytest.mark.asyncio
    @patch("host_profile_reconciler.send_discord_alert", new_callable=AsyncMock)
    @patch("host_profile_reconciler.list_pending_profiles", new_callable=AsyncMock)
    async def test_alert_names_the_hardware(self, pending, alert):
        """The point of the DB move: triage without opening the raw document."""
        pending.return_value = [_record(FP_A)]
        alert.return_value = True

        await host_profile_reconciler.notify_new_submissions(_session())

        body = alert.await_args.args[0]
        assert "2335" in body and "8" in body

    @pytest.mark.asyncio
    @patch("host_profile_reconciler.send_discord_alert", new_callable=AsyncMock)
    @patch("host_profile_reconciler.list_pending_profiles", new_callable=AsyncMock)
    async def test_never_triggers_generation(self, pending, alert):
        """The job notifies and reconciles only -- a human decides when to generate."""
        pending.return_value = [_record(FP_A)]
        alert.return_value = True

        await host_profile_reconciler.notify_new_submissions(_session())

        assert "Awaiting measurement generation" in alert.await_args.args[0]
