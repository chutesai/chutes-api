"""
Unit tests for the public GET /servers/tee/measurements endpoint.
"""

import pytest
import orjson as json
from unittest.mock import AsyncMock, MagicMock, patch

from api.config import TeeMeasurementConfig
from api.server.router import get_tee_measurements
from api.server.schemas import TeeMeasurementResponse


def _make_measurement(**overrides):
    defaults = dict(
        version="1",
        name="8xh200",
        mrtd="A" * 96,
        boot_rtmrs={"RTMR0": "B" * 96, "RTMR1": "C" * 96, "RTMR2": "D" * 96, "RTMR3": "E" * 96},
        runtime_rtmrs={"RTMR0": "B" * 96, "RTMR1": "C" * 96, "RTMR2": "F" * 96, "RTMR3": "A" * 96},
        expected_gpus=["h200"],
        gpu_count=8,
    )
    defaults.update(overrides)
    return TeeMeasurementConfig(**defaults)


def _make_mock_settings(measurements, cached=None):
    mock_settings = MagicMock()
    mock_settings.tee_measurements = measurements
    mock_settings.redis_client.get = AsyncMock(return_value=cached)
    mock_settings.redis_client.set = AsyncMock()
    return mock_settings


@pytest.mark.asyncio
@patch("api.server.router.settings")
async def test_returns_all_measurements(mock_settings):
    m1 = _make_measurement(name="8xh200", version="1", expected_gpus=["h200"], gpu_count=8)
    m2 = _make_measurement(name="8xb200", version="2", expected_gpus=["b200"], gpu_count=8)
    mock_settings.tee_measurements = [m1, m2]
    mock_settings.redis_client.get = AsyncMock(return_value=None)
    mock_settings.redis_client.set = AsyncMock()

    result = await get_tee_measurements()

    assert len(result) == 2
    assert all(isinstance(r, TeeMeasurementResponse) for r in result)


@pytest.mark.asyncio
@patch("api.server.router.settings")
async def test_measurement_fields_are_correct(mock_settings):
    m = _make_measurement()
    mock_settings.tee_measurements = [m]
    mock_settings.redis_client.get = AsyncMock(return_value=None)
    mock_settings.redis_client.set = AsyncMock()

    result = await get_tee_measurements()

    assert len(result) == 1
    r = result[0]
    assert r.version == "1"
    assert r.name == "8xh200"
    assert r.mrtd == "A" * 96
    assert r.boot_rtmrs == m.boot_rtmrs
    assert r.runtime_rtmrs == m.runtime_rtmrs
    assert r.expected_gpus == ["h200"]
    assert r.gpu_count == 8
    assert isinstance(r.gpu_count, int)


@pytest.mark.asyncio
@patch("api.server.router.settings")
async def test_returns_empty_list_when_no_measurements(mock_settings):
    mock_settings.tee_measurements = []
    mock_settings.redis_client.get = AsyncMock(return_value=None)
    mock_settings.redis_client.set = AsyncMock()

    result = await get_tee_measurements()

    assert result == []


@pytest.mark.asyncio
@patch("api.server.router.settings")
async def test_result_is_written_to_cache(mock_settings):
    m = _make_measurement()
    mock_settings.tee_measurements = [m]
    mock_settings.redis_client.get = AsyncMock(return_value=None)
    mock_settings.redis_client.set = AsyncMock()

    await get_tee_measurements()

    mock_settings.redis_client.set.assert_awaited_once()
    call_args = mock_settings.redis_client.set.call_args
    assert call_args.args[0] == "tee_measurements"
    assert call_args.kwargs["ex"] == 3600


@pytest.mark.asyncio
@patch("api.server.router.settings")
async def test_cache_hit_skips_settings_and_write(mock_settings):
    m = _make_measurement()
    cached_data = [m.__dict__]
    mock_settings.redis_client.get = AsyncMock(return_value=json.dumps(cached_data))
    mock_settings.redis_client.set = AsyncMock()

    result = await get_tee_measurements()

    # Returned the cached payload directly; no write back to Redis
    mock_settings.redis_client.set.assert_not_awaited()
    # tee_measurements should never have been accessed
    mock_settings.tee_measurements.__iter__.assert_not_called() if hasattr(
        mock_settings.tee_measurements, "__iter__"
    ) else None
    assert result is not None
