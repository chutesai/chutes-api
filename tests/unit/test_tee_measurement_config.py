"""
Unit tests for TEE measurement config loading (api.config._load_tee_measurements),
focused on the release-candidate (rc) allowlist invariants: an rc version must declare non-empty,
valid `authorized_hotkeys` (register/runtime) AND `authorized_signing_keys` (boot/provision, RSA
operator public keys) or it is dropped as unusable.
"""

from pathlib import Path

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from api.config import Settings, TeeMeasurementConfig


def _make_pubkey_pem() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )


_VALID_PUBKEY_PEM = _make_pubkey_pem()


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


def _rc_version(
    *,
    authorized_hotkeys=("5Auth1",),
    authorized_signing_keys=(_VALID_PUBKEY_PEM,),
    version="1.5.0-rc",
):
    """A loadable rc version by default (both allowlists valid). Pass None/[]/bad to a field to
    exercise the drop paths."""
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
    if authorized_signing_keys is not None:
        cfg["authorized_signing_keys"] = list(authorized_signing_keys)
    return cfg


def _settings_for(measurements, tmp_path: Path) -> Settings:
    """A real-ish Settings whose tee_measurement_config_path points at a written YAML file."""
    path = tmp_path / "tee_measurements.yaml"
    path.write_text(yaml.safe_dump({"measurements": measurements}))
    s = Settings.__new__(Settings)
    object.__setattr__(s, "tee_measurement_config_path", path)
    return s


def test_published_measurement_loads_without_allowlists(tmp_path):
    s = _settings_for([PUBLISHED_VERSION], tmp_path)
    loaded = s._load_tee_measurements()
    assert len(loaded) == 1
    assert loaded[0].rc is False
    assert loaded[0].authorized_hotkeys == []
    assert loaded[0].authorized_signing_keys == []


def test_rc_measurement_loads_with_both_allowlists(tmp_path):
    s = _settings_for(
        [_rc_version(authorized_hotkeys=["5Auth1", "5Auth2"])],
        tmp_path,
    )
    loaded = s._load_tee_measurements()
    assert len(loaded) == 1
    assert loaded[0].rc is True
    assert loaded[0].authorized_hotkeys == ["5Auth1", "5Auth2"]
    # The loader strips entries (incl. the PEM's trailing newline); a stripped PEM still loads.
    assert loaded[0].authorized_signing_keys == [_VALID_PUBKEY_PEM.strip()]
    serialization.load_pem_public_key(loaded[0].authorized_signing_keys[0].encode())


# authorized_hotkeys invariant (register/runtime path).


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


# authorized_signing_keys invariant (boot/provision path).


def test_rc_measurement_without_signing_keys_is_dropped(tmp_path):
    s = _settings_for([_rc_version(authorized_signing_keys=None)], tmp_path)
    assert s._load_tee_measurements() == []


def test_rc_measurement_with_empty_signing_keys_is_dropped(tmp_path):
    s = _settings_for([_rc_version(authorized_signing_keys=[])], tmp_path)
    assert s._load_tee_measurements() == []


def test_rc_measurement_with_unparseable_signing_key_is_dropped(tmp_path):
    """A signing key that doesn't load as a PEM public key makes the whole rc version unusable."""
    s = _settings_for(
        [_rc_version(authorized_signing_keys=[_VALID_PUBKEY_PEM, "-----NOT A KEY-----"])],
        tmp_path,
    )
    assert s._load_tee_measurements() == []


# Isolation: one bad rc version must not take down others.


def test_bad_rc_entry_does_not_drop_published_measurements(tmp_path):
    s = _settings_for(
        [PUBLISHED_VERSION, _rc_version(authorized_signing_keys=None)],
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


def test_teemeasurementconfig_defaults_allowlists_empty():
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
    assert cfg.authorized_signing_keys == []
    assert cfg.rc is False
