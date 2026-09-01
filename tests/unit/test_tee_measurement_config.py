"""
Unit tests for TEE measurement config loading (api.config._load_tee_measurements),
focused on the release-candidate (rc) allowlist invariant: an rc version must declare a non-empty
`authorized_hotkeys` or it is dropped as unusable. That single allowlist now covers every path --
the guest initramfs proves possession of its miner hotkey too, so the separate operator RSA
`authorized_signing_keys` allowlist is gone.
"""

import copy
from pathlib import Path

import pytest
import yaml

from api.config import Settings, TeeMeasurementConfig


PUBLISHED_VERSION = {
    "version": "1.4.0",
    "mrtd": "a" * 96,
    "rtmr1": "c" * 96,
    "rtmr2": "d" * 96,
    "runtime_rtmr3": "e" * 96,
    "hardware": [
        {
            "name": "8xh200",
            "rtmr0": "b" * 96,
            "expected_gpus": ["h200"],
            "gpu_count": 8,
        }
    ],
}


def _rc_version(*, authorized_hotkeys=("5Auth1",), version="1.5.0-rc"):
    """A loadable rc version by default. Pass None/[] to exercise the drop path."""
    cfg = {
        "version": version,
        "mrtd": "1" * 96,
        "rtmr1": "2" * 96,
        "rtmr2": "3" * 96,
        "runtime_rtmr3": "4" * 96,
        "hardware": [
            {
                "name": "8xh200",
                "rtmr0": "5" * 96,
                "expected_gpus": ["h200"],
                "gpu_count": 8,
            }
        ],
        "rc": True,
    }
    if authorized_hotkeys is not None:
        cfg["authorized_hotkeys"] = list(authorized_hotkeys)
    return cfg


def _settings_for(measurements, tmp_path: Path) -> Settings:
    """A real-ish Settings whose tee_measurement_config_path points at a written YAML file."""
    path = tmp_path / "tee_measurements.yaml"
    path.write_text(yaml.safe_dump({"measurements": measurements}))
    s = Settings.__new__(Settings)
    object.__setattr__(s, "tee_measurement_config_path", path)
    return s


def test_published_measurement_loads_without_allowlist(tmp_path):
    s = _settings_for([PUBLISHED_VERSION], tmp_path)
    loaded = s._load_tee_measurements()
    assert len(loaded) == 1
    assert loaded[0].rc is False
    assert loaded[0].authorized_hotkeys == []


def test_rc_measurement_loads_with_allowlist(tmp_path):
    s = _settings_for(
        [_rc_version(authorized_hotkeys=["5Auth1", "5Auth2"])],
        tmp_path,
    )
    loaded = s._load_tee_measurements()
    assert len(loaded) == 1
    assert loaded[0].rc is True
    assert loaded[0].authorized_hotkeys == ["5Auth1", "5Auth2"]


# authorized_hotkeys invariant -- the one allowlist, enforced on every path.


def test_rc_measurement_without_authorized_hotkeys_is_dropped(tmp_path):
    s = _settings_for([_rc_version(authorized_hotkeys=None)], tmp_path)
    assert s._load_tee_measurements() == []


def test_rc_measurement_with_empty_authorized_hotkeys_is_dropped(tmp_path):
    s = _settings_for([_rc_version(authorized_hotkeys=[])], tmp_path)
    assert s._load_tee_measurements() == []


def test_rc_measurement_with_blank_authorized_hotkeys_is_dropped(tmp_path):
    """Whitespace-only entries strip to an empty allowlist -> dropped."""
    s = _settings_for([_rc_version(authorized_hotkeys=["   ", ""])], tmp_path)
    assert s._load_tee_measurements() == []


# Isolation: one bad rc version must not take down others.


def test_bad_rc_entry_does_not_drop_published_measurements(tmp_path):
    s = _settings_for(
        [PUBLISHED_VERSION, _rc_version(authorized_hotkeys=None)],
        tmp_path,
    )
    loaded = s._load_tee_measurements()
    assert {m.version for m in loaded} == {"1.4.0"}


def test_allowlist_whitespace_is_stripped(tmp_path):
    s = _settings_for(
        [_rc_version(authorized_hotkeys=["  5Auth1  ", "5Auth2"])],
        tmp_path,
    )
    loaded = s._load_tee_measurements()
    assert loaded[0].authorized_hotkeys == ["5Auth1", "5Auth2"]


def test_teemeasurementconfig_defaults_allowlist_empty():
    """The dataclass defaults keep published-measurement construction unchanged."""
    cfg = TeeMeasurementConfig(
        version="1",
        name="x",
        mrtd="a" * 96,
        rtmr0="b" * 96,
        rtmr1="c" * 96,
        rtmr2="d" * 96,
        runtime_rtmr3="e" * 96,
        expected_gpus=["h200"],
        gpu_count=8,
    )
    assert cfg.authorized_hotkeys == []
    assert cfg.rc is False


def _with_fingerprint(value):
    """PUBLISHED_VERSION with a fingerprint on its single hardware entry."""
    version = copy.deepcopy(PUBLISHED_VERSION)
    version["hardware"][0]["fingerprint"] = value
    return version


def test_hardware_fingerprint_loads(tmp_path):
    s = _settings_for([_with_fingerprint("a" * 64)], tmp_path)
    assert s._load_tee_measurements()[0].fingerprint == "a" * 64


def test_hardware_fingerprint_is_normalised(tmp_path):
    s = _settings_for([_with_fingerprint("  " + "A" * 64 + "  ")], tmp_path)
    assert s._load_tee_measurements()[0].fingerprint == "a" * 64


def test_hardware_fingerprint_is_optional(tmp_path):
    """Entries predating the field must still load -- they are simply unmatchable."""
    s = _settings_for([PUBLISHED_VERSION], tmp_path)
    loaded = s._load_tee_measurements()
    assert len(loaded) == 1
    assert loaded[0].fingerprint is None


@pytest.mark.parametrize("value", [None, "", "   "])
def test_blank_hardware_fingerprint_is_treated_as_absent(tmp_path, value):
    s = _settings_for([_with_fingerprint(value)], tmp_path)
    assert s._load_tee_measurements()[0].fingerprint is None


@pytest.mark.parametrize(
    "value",
    ["a" * 63, "a" * 65, "g" * 64, "0x" + "a" * 62, "a" * 32 + "-" + "a" * 31],
)
def test_malformed_hardware_fingerprint_is_rejected(tmp_path, value):
    """A wrong fingerprint is worse than none: it would match the wrong host class."""
    s = _settings_for([_with_fingerprint(value)], tmp_path)
    with pytest.raises(ValueError, match="fingerprint"):
        s._load_tee_measurements()


def test_fingerprint_defaults_to_none_on_the_dataclass():
    cfg = TeeMeasurementConfig(
        version="1",
        name="x",
        mrtd="a" * 96,
        rtmr0="b" * 96,
        rtmr1="c" * 96,
        rtmr2="d" * 96,
        runtime_rtmr3="e" * 96,
        expected_gpus=["h200"],
        gpu_count=8,
    )
    assert cfg.fingerprint is None
