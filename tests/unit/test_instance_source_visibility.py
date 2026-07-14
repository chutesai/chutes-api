import base64
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


with patch("ctypes.CDLL", return_value=MagicMock()):
    from api.instance.router import _validate_launch_config_env


@pytest.mark.asyncio
async def test_legacy_launch_mismatch_does_not_disclose_hidden_source(monkeypatch):
    secret_source = "print('private chute source: do not disclose')"
    submitted_source = "print('different source')"
    encoded_submission = base64.b64encode(submitted_source.encode()).decode()

    monkeypatch.setenv("ENVDUMP_UNLOCK", "test-key")

    db = AsyncMock()
    launch_config = MagicMock(
        config_id="config-id",
        env_key="env-key",
        miner_hotkey="miner-hotkey",
    )
    chute = MagicMock(
        chutes_version="0.3.60",
        code=secret_source,
    )
    args = MagicMock(env="encrypted-env", code="encrypted-code")

    with (
        patch.dict(sys.modules, {"chutes.envdump": MagicMock()}),
        patch(
            "api.instance.router.asyncio.to_thread",
            new=AsyncMock(
                side_effect=[
                    {"env": {}},
                    {"content": encoded_submission},
                ]
            ),
        ),
        patch("api.instance.router.verify_expected_command", new_callable=AsyncMock),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _validate_launch_config_env(
                db,
                launch_config,
                chute,
                args,
                "ENVDUMP: test",
            )

    assert exc_info.value.status_code == 403
    assert "Incorrect code supplied" in exc_info.value.detail
    assert secret_source not in exc_info.value.detail
    assert secret_source not in launch_config.verification_error
    db.commit.assert_awaited_once()
