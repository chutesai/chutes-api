"""
Unit tests for LUKS passphrase rotation and Fernet encrypt/decrypt flow.

Covers:
- Fernet round-trip symmetry and key-mismatch errors
- First-boot provisioning: passphrase must survive confirm(rotated=false)
- Restart rotation: decrypt existing → generate pending → confirm promotes
- Full lifecycle: first boot → confirm → restart → confirm
- Legacy sync flow (sync_server_luks_passphrases)
"""

import secrets
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet, InvalidToken

from api.server.util import (
    encrypt_passphrase,
    decrypt_passphrase,
    generate_cache_passphrase,
    rotate_luks_passphrases,
    sync_server_luks_passphrases,
    _get_fernet,
)
from api.server.exceptions import InvalidTdxConfiguration


# ---------------------------------------------------------------------------
# Fernet encrypt / decrypt round-trip
# ---------------------------------------------------------------------------


class TestFernetRoundTrip:
    """Verify encrypt_passphrase ↔ decrypt_passphrase symmetry."""

    @patch("api.server.util.settings")
    def test_round_trip(self, mock_settings):
        key = Fernet.generate_key()
        mock_settings.fernet_key = Fernet(key)

        plaintext = generate_cache_passphrase()
        ciphertext = encrypt_passphrase(plaintext)
        assert decrypt_passphrase(ciphertext) == plaintext

    @patch("api.server.util.settings")
    def test_different_plaintexts_produce_different_ciphertexts(self, mock_settings):
        mock_settings.fernet_key = Fernet(Fernet.generate_key())

        ct1 = encrypt_passphrase("passphrase_a")
        ct2 = encrypt_passphrase("passphrase_b")
        assert ct1 != ct2

    @patch("api.server.util.settings")
    def test_ciphertext_is_not_plaintext(self, mock_settings):
        mock_settings.fernet_key = Fernet(Fernet.generate_key())

        plaintext = "supersecret"
        ciphertext = encrypt_passphrase(plaintext)
        assert ciphertext != plaintext
        assert plaintext not in ciphertext


# ---------------------------------------------------------------------------
# Key mismatch — the cluster-migration failure scenario
# ---------------------------------------------------------------------------


class TestFernetKeyMismatch:
    """Simulates what happens when CACHE_PASSPHRASE_KEY changes between clusters."""

    def test_decrypt_with_wrong_key_raises(self):
        """Decrypting a ciphertext with a different Fernet key raises InvalidToken."""
        key_old = Fernet.generate_key()
        key_new = Fernet.generate_key()
        assert key_old != key_new

        plaintext = generate_cache_passphrase()
        ciphertext = Fernet(key_old).encrypt(plaintext.encode()).decode()

        with pytest.raises(InvalidToken):
            Fernet(key_new).decrypt(ciphertext.encode())

    @patch("api.server.util.settings")
    def test_encrypt_old_key_decrypt_new_key_raises(self, mock_settings):
        """End-to-end: encrypt with old settings, swap key, decrypt fails."""
        old_key = Fernet.generate_key()
        new_key = Fernet.generate_key()

        # Encrypt under old key
        mock_settings.fernet_key = Fernet(old_key)
        plaintext = "my_luks_passphrase"
        ciphertext = encrypt_passphrase(plaintext)

        # Swap to new key — simulates cluster migration with different secret
        mock_settings.fernet_key = Fernet(new_key)
        with pytest.raises(InvalidToken):
            decrypt_passphrase(ciphertext)


# ---------------------------------------------------------------------------
# _get_fernet edge cases
# ---------------------------------------------------------------------------


class TestGetFernet:
    @patch("api.server.util.settings")
    def test_missing_key_raises(self, mock_settings):
        mock_settings.fernet_key = None
        with pytest.raises(InvalidTdxConfiguration, match="CACHE_PASSPHRASE_KEY"):
            _get_fernet()

    @patch("api.server.util.settings")
    def test_valid_key_returns_fernet(self, mock_settings):
        mock_settings.fernet_key = Fernet(Fernet.generate_key())
        assert isinstance(_get_fernet(), Fernet)


# ---------------------------------------------------------------------------
# generate_cache_passphrase
# ---------------------------------------------------------------------------


class TestGenerateCachePassphrase:
    def test_length(self):
        pp = generate_cache_passphrase()
        assert len(pp) == 128

    def test_hex(self):
        pp = generate_cache_passphrase()
        int(pp, 16)  # raises if not valid hex

    def test_unique(self):
        assert generate_cache_passphrase() != generate_cache_passphrase()


# ---------------------------------------------------------------------------
# rotate_luks_passphrases (DB interactions mocked)
# ---------------------------------------------------------------------------


def _make_vm_config(volume_passphrases=None, k3s_encryption_key=None):
    """Create a mock VmCacheConfig for testing."""
    config = MagicMock()
    config.volume_passphrases = volume_passphrases or {}
    config.k3s_encryption_key = k3s_encryption_key
    config.last_boot_at = None
    return config


@pytest.mark.asyncio
class TestRotateLuksPassphrases:
    """Tests for rotate_luks_passphrases exercising first-boot and restart paths."""

    @patch("api.server.util.settings")
    @patch("api.server.util._get_vm_cache_config")
    @patch("api.server.util._create_vm_cache_config")
    async def test_first_boot_no_existing_config(
        self, mock_create, mock_get, mock_settings
    ):
        """First boot: no VmCacheConfig row exists → creates one, current=None."""
        mock_settings.fernet_key = Fernet(Fernet.generate_key())

        mock_get.return_value = None
        new_config = _make_vm_config()
        mock_create.return_value = new_config

        db = AsyncMock()

        volumes, vm_config = await rotate_luks_passphrases(
            db, "hotkey1", "vm1", ["storage", "tdx-cache"]
        )

        assert "storage" in volumes
        assert "tdx-cache" in volumes
        assert volumes["storage"].current is None  # first boot
        assert volumes["storage"].is_first_boot is True
        assert volumes["tdx-cache"].current is None
        assert len(volumes["storage"].next) == 128  # hex passphrase

    @patch("api.server.util.settings")
    @patch("api.server.util._get_vm_cache_config")
    @patch("api.server.util._create_vm_cache_config")
    async def test_first_boot_persists_passphrase_as_current(
        self, mock_create, mock_get, mock_settings
    ):
        """First boot must store the passphrase as both current and pending.

        The VM promotes next→current client-side (because current=null) and
        confirms with rotated=false.  If the passphrase is only stored as
        pending, confirm discards it and the key used to luksFormat is lost.
        """
        fernet = Fernet(Fernet.generate_key())
        mock_settings.fernet_key = fernet

        mock_get.return_value = None
        new_config = _make_vm_config()
        mock_create.return_value = new_config
        db = AsyncMock()

        volumes, vm_config = await rotate_luks_passphrases(
            db, "hotkey1", "vm1", ["storage"]
        )

        stored = vm_config.volume_passphrases
        # Both current and pending must exist after first-boot rotate
        assert "storage" in stored, "first-boot passphrase not stored as current"
        assert "pending_storage" in stored
        # Both must decrypt to the same passphrase (the one returned as next)
        assert fernet.decrypt(stored["storage"].encode()).decode() == volumes["storage"].next
        assert fernet.decrypt(stored["pending_storage"].encode()).decode() == volumes["storage"].next

    @patch("api.server.util.settings")
    @patch("api.server.util._get_vm_cache_config")
    async def test_restart_existing_config(self, mock_get, mock_settings):
        """Restart: existing VmCacheConfig → decrypts current, generates next."""
        fernet_key = Fernet.generate_key()
        fernet = Fernet(fernet_key)
        mock_settings.fernet_key = fernet

        original_passphrase = generate_cache_passphrase()
        encrypted = fernet.encrypt(original_passphrase.encode()).decode()

        mock_get.return_value = _make_vm_config(
            volume_passphrases={"storage": encrypted}
        )
        db = AsyncMock()

        volumes, vm_config = await rotate_luks_passphrases(
            db, "hotkey1", "vm1", ["storage"]
        )

        assert volumes["storage"].current == original_passphrase
        assert volumes["storage"].is_first_boot is False
        assert volumes["storage"].next != original_passphrase

    @patch("api.server.util.settings")
    @patch("api.server.util._get_vm_cache_config")
    async def test_restart_does_not_overwrite_current(self, mock_get, mock_settings):
        """On restart, {vol} (current) must not be overwritten by rotate."""
        fernet = Fernet(Fernet.generate_key())
        mock_settings.fernet_key = fernet

        original = generate_cache_passphrase()
        encrypted = fernet.encrypt(original.encode()).decode()
        mock_get.return_value = _make_vm_config(
            volume_passphrases={"storage": encrypted}
        )
        db = AsyncMock()

        volumes, vm_config = await rotate_luks_passphrases(
            db, "hotkey1", "vm1", ["storage"]
        )

        stored = vm_config.volume_passphrases
        # Current must be the original passphrase, not overwritten
        assert fernet.decrypt(stored["storage"].encode()).decode() == original
        # Pending must be the new passphrase
        assert fernet.decrypt(stored["pending_storage"].encode()).decode() == volumes["storage"].next
        assert volumes["storage"].next != original

    @patch("api.server.util.settings")
    @patch("api.server.util._get_vm_cache_config")
    async def test_restart_wrong_fernet_key_raises(self, mock_get, mock_settings):
        """Restart with wrong Fernet key: decrypt_passphrase raises InvalidToken."""
        old_fernet = Fernet(Fernet.generate_key())
        new_fernet = Fernet(Fernet.generate_key())

        original_passphrase = generate_cache_passphrase()
        encrypted_with_old = old_fernet.encrypt(original_passphrase.encode()).decode()

        mock_settings.fernet_key = new_fernet
        mock_get.return_value = _make_vm_config(
            volume_passphrases={"storage": encrypted_with_old}
        )
        db = AsyncMock()

        with pytest.raises(InvalidToken):
            await rotate_luks_passphrases(db, "hotkey1", "vm1", ["storage"])

    @patch("api.server.util.settings")
    @patch("api.server.util._get_vm_cache_config")
    async def test_stale_pending_is_discarded(self, mock_get, mock_settings):
        """Stale pending_{vol} from prior unconfirmed rotation is cleaned up."""
        fernet = Fernet(Fernet.generate_key())
        mock_settings.fernet_key = fernet

        current = generate_cache_passphrase()
        stale_pending = generate_cache_passphrase()
        mock_get.return_value = _make_vm_config(
            volume_passphrases={
                "storage": fernet.encrypt(current.encode()).decode(),
                "pending_storage": fernet.encrypt(stale_pending.encode()).decode(),
            }
        )
        db = AsyncMock()

        volumes, vm_config = await rotate_luks_passphrases(
            db, "hotkey1", "vm1", ["storage"]
        )

        stored = vm_config.volume_passphrases
        assert "pending_storage" in stored
        decrypted_pending = fernet.decrypt(stored["pending_storage"].encode()).decode()
        assert decrypted_pending == volumes["storage"].next
        assert decrypted_pending != stale_pending

    @patch("api.server.util.settings")
    @patch("api.server.util._get_vm_cache_config")
    async def test_multiple_volumes(self, mock_get, mock_settings):
        """Rotation handles storage + tdx-cache together."""
        fernet = Fernet(Fernet.generate_key())
        mock_settings.fernet_key = fernet

        storage_pp = generate_cache_passphrase()
        cache_pp = generate_cache_passphrase()
        mock_get.return_value = _make_vm_config(
            volume_passphrases={
                "storage": fernet.encrypt(storage_pp.encode()).decode(),
                "tdx-cache": fernet.encrypt(cache_pp.encode()).decode(),
            }
        )
        db = AsyncMock()

        volumes, _ = await rotate_luks_passphrases(
            db, "hotkey1", "vm1", ["storage", "tdx-cache"]
        )

        assert volumes["storage"].current == storage_pp
        assert volumes["tdx-cache"].current == cache_pp
        assert volumes["storage"].next != storage_pp
        assert volumes["tdx-cache"].next != cache_pp

    @patch("api.server.util.settings")
    @patch("api.server.util._get_vm_cache_config")
    async def test_mixed_existing_and_new_volumes(self, mock_get, mock_settings):
        """One volume has an existing passphrase, the other is new.

        The existing volume's current must not be overwritten; the new
        volume must get the first-boot treatment (stored as current).
        """
        fernet = Fernet(Fernet.generate_key())
        mock_settings.fernet_key = fernet

        existing_pp = generate_cache_passphrase()
        mock_get.return_value = _make_vm_config(
            volume_passphrases={
                "storage": fernet.encrypt(existing_pp.encode()).decode(),
            }
        )
        db = AsyncMock()

        volumes, vm_config = await rotate_luks_passphrases(
            db, "hotkey1", "vm1", ["storage", "tdx-cache"]
        )

        stored = vm_config.volume_passphrases

        # storage: existing → current is the original, not overwritten
        assert volumes["storage"].current == existing_pp
        assert volumes["storage"].is_first_boot is False
        assert fernet.decrypt(stored["storage"].encode()).decode() == existing_pp

        # tdx-cache: new → current is None, but stored as current for safety
        assert volumes["tdx-cache"].current is None
        assert volumes["tdx-cache"].is_first_boot is True
        assert "tdx-cache" in stored, "new volume not persisted as current"
        assert fernet.decrypt(stored["tdx-cache"].encode()).decode() == volumes["tdx-cache"].next

        # Both have pending keys
        assert "pending_storage" in stored
        assert "pending_tdx-cache" in stored

        # After confirm(storage=true, tdx-cache=false), both passphrases survive
        _simulate_confirm(stored, volumes, {"storage": True, "tdx-cache": False})
        assert fernet.decrypt(stored["storage"].encode()).decode() == volumes["storage"].next
        assert fernet.decrypt(stored["tdx-cache"].encode()).decode() == volumes["tdx-cache"].next


# ---------------------------------------------------------------------------
# sync_server_luks_passphrases (legacy flow, DB mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSyncServerLuksPassphrases:
    """Tests for the legacy POST /luks sync flow."""

    @patch("api.server.util.settings")
    @patch("api.server.util._get_vm_cache_config")
    @patch("api.server.util._create_vm_cache_config")
    async def test_first_boot_creates_passphrases(
        self, mock_create, mock_get, mock_settings
    ):
        mock_settings.fernet_key = Fernet(Fernet.generate_key())
        mock_get.return_value = None
        mock_create.return_value = _make_vm_config()
        db = AsyncMock()

        result = await sync_server_luks_passphrases(
            db, "hotkey1", "vm1", ["storage", "tdx-cache"]
        )

        assert "storage" in result
        assert "tdx-cache" in result
        assert len(result["storage"]) == 128

    @patch("api.server.util.settings")
    @patch("api.server.util._get_vm_cache_config")
    async def test_restart_returns_existing(self, mock_get, mock_settings):
        fernet = Fernet(Fernet.generate_key())
        mock_settings.fernet_key = fernet

        original = generate_cache_passphrase()
        encrypted = fernet.encrypt(original.encode()).decode()
        mock_get.return_value = _make_vm_config(
            volume_passphrases={"storage": encrypted}
        )
        db = AsyncMock()

        result = await sync_server_luks_passphrases(
            db, "hotkey1", "vm1", ["storage"]
        )

        assert result["storage"] == original

    @patch("api.server.util.settings")
    @patch("api.server.util._get_vm_cache_config")
    async def test_restart_wrong_key_raises(self, mock_get, mock_settings):
        """Legacy sync also fails if Fernet key changed."""
        old_fernet = Fernet(Fernet.generate_key())
        new_fernet = Fernet(Fernet.generate_key())

        encrypted = old_fernet.encrypt(b"passphrase").decode()
        mock_settings.fernet_key = new_fernet
        mock_get.return_value = _make_vm_config(
            volume_passphrases={"storage": encrypted}
        )
        db = AsyncMock()

        with pytest.raises(InvalidToken):
            await sync_server_luks_passphrases(
                db, "hotkey1", "vm1", ["storage"]
            )

    @patch("api.server.util.settings")
    @patch("api.server.util._get_vm_cache_config")
    async def test_rekey_generates_new(self, mock_get, mock_settings):
        fernet = Fernet(Fernet.generate_key())
        mock_settings.fernet_key = fernet

        original = generate_cache_passphrase()
        encrypted = fernet.encrypt(original.encode()).decode()
        mock_get.return_value = _make_vm_config(
            volume_passphrases={"storage": encrypted}
        )
        db = AsyncMock()

        result = await sync_server_luks_passphrases(
            db, "hotkey1", "vm1", ["storage"], rekey_volume_names=["storage"]
        )

        assert result["storage"] != original
        assert len(result["storage"]) == 128

    @patch("api.server.util.settings")
    @patch("api.server.util._get_vm_cache_config")
    async def test_prune_removes_unlisted_volumes(self, mock_get, mock_settings):
        fernet = Fernet(Fernet.generate_key())
        mock_settings.fernet_key = fernet

        mock_get.return_value = _make_vm_config(
            volume_passphrases={
                "storage": fernet.encrypt(b"pp1").decode(),
                "old-vol": fernet.encrypt(b"pp2").decode(),
            }
        )
        db = AsyncMock()

        await sync_server_luks_passphrases(
            db, "hotkey1", "vm1", ["storage"]
        )

        stored = mock_get.return_value.volume_passphrases
        assert "storage" in stored
        assert "old-vol" not in stored


# ---------------------------------------------------------------------------
# Full lifecycle: first boot → confirm → restart (the critical regression)
# ---------------------------------------------------------------------------


def _simulate_confirm(stored: dict, volumes: dict, rotated_flags: dict) -> dict:
    """Simulate process_luks_confirm logic on volume_passphrases dict.

    rotated_flags: {vol_name: bool} — mirrors the VM's confirm request.
    Returns the confirmed dict (mutated in place).
    """
    for vol, rotated in rotated_flags.items():
        pending_key = f"pending_{vol}"
        if rotated:
            if pending_key in stored:
                stored[vol] = stored.pop(pending_key)
        else:
            stored.pop(pending_key, None)
    return stored


@pytest.mark.asyncio
class TestFirstBootRestartLifecycle:
    """End-to-end lifecycle: first boot → confirm(rotated=false) → restart.

    Before the fix, the passphrase used to luksFormat was only stored as
    pending_{vol}.  The VM's confirm with rotated=false discarded it,
    leaving volume_passphrases empty.  On restart, rotate returned
    current=None (looks like another first boot) with a completely new
    passphrase that cannot unlock the already-formatted volume.
    """

    @patch("api.server.util.settings")
    @patch("api.server.util._get_vm_cache_config")
    @patch("api.server.util._create_vm_cache_config")
    async def test_passphrase_survives_confirm_false(
        self, mock_create, mock_get, mock_settings
    ):
        """After first boot + confirm(rotated=false), passphrase must persist."""
        fernet = Fernet(Fernet.generate_key())
        mock_settings.fernet_key = fernet

        # ---- Step 1: first boot (no config exists) ----
        mock_get.return_value = None
        config = _make_vm_config()
        mock_create.return_value = config
        db = AsyncMock()

        volumes, vm_config = await rotate_luks_passphrases(
            db, "hk", "vm1", ["storage", "tdx-cache"]
        )
        format_key_storage = volumes["storage"].next
        format_key_cache = volumes["tdx-cache"].next

        # ---- Step 2: VM confirms with rotated=false (no luksAddKey on first boot) ----
        stored = dict(vm_config.volume_passphrases)
        _simulate_confirm(stored, volumes, {"storage": False, "tdx-cache": False})

        # The passphrase must still be recoverable as current
        assert "storage" in stored, "storage passphrase lost after confirm(rotated=false)"
        assert "tdx-cache" in stored, "tdx-cache passphrase lost after confirm(rotated=false)"
        assert fernet.decrypt(stored["storage"].encode()).decode() == format_key_storage
        assert fernet.decrypt(stored["tdx-cache"].encode()).decode() == format_key_cache

    @patch("api.server.util.settings")
    @patch("api.server.util._get_vm_cache_config")
    @patch("api.server.util._create_vm_cache_config")
    async def test_restart_after_first_boot_returns_correct_current(
        self, mock_create, mock_get, mock_settings
    ):
        """Full cycle: first boot → confirm → restart must return original key as current."""
        fernet = Fernet(Fernet.generate_key())
        mock_settings.fernet_key = fernet

        # ---- Step 1: first boot ----
        mock_get.return_value = None
        config = _make_vm_config()
        mock_create.return_value = config
        db = AsyncMock()

        volumes_boot, vm_config = await rotate_luks_passphrases(
            db, "hk", "vm1", ["storage"]
        )
        format_key = volumes_boot["storage"].next

        # ---- Step 2: confirm(rotated=false) ----
        stored = dict(vm_config.volume_passphrases)
        _simulate_confirm(stored, volumes_boot, {"storage": False})

        # ---- Step 3: restart — simulate by calling rotate with the post-confirm state ----
        restart_config = _make_vm_config(volume_passphrases=stored)
        mock_get.return_value = restart_config

        volumes_restart, _ = await rotate_luks_passphrases(
            db, "hk", "vm1", ["storage"]
        )

        # Current must be the same key used to format the volume on first boot
        assert volumes_restart["storage"].current == format_key
        assert volumes_restart["storage"].is_first_boot is False
        # Next must be a new rotation key (different from original)
        assert volumes_restart["storage"].next != format_key

    @patch("api.server.util.settings")
    @patch("api.server.util._get_vm_cache_config")
    @patch("api.server.util._create_vm_cache_config")
    async def test_full_rotation_cycle(
        self, mock_create, mock_get, mock_settings
    ):
        """Full cycle: first boot → confirm → restart → confirm(rotated=true).

        Validates the passphrase is rotated correctly over the full lifecycle.
        """
        fernet = Fernet(Fernet.generate_key())
        mock_settings.fernet_key = fernet
        db = AsyncMock()

        # ---- First boot ----
        mock_get.return_value = None
        config = _make_vm_config()
        mock_create.return_value = config

        vol_boot, vm_config = await rotate_luks_passphrases(
            db, "hk", "vm1", ["storage"]
        )
        original_key = vol_boot["storage"].next
        stored = dict(vm_config.volume_passphrases)

        # ---- Confirm first boot (rotated=false) ----
        _simulate_confirm(stored, vol_boot, {"storage": False})

        # ---- Restart ----
        restart_config = _make_vm_config(volume_passphrases=stored)
        mock_get.return_value = restart_config

        vol_restart, vm_config2 = await rotate_luks_passphrases(
            db, "hk", "vm1", ["storage"]
        )
        assert vol_restart["storage"].current == original_key
        rotation_key = vol_restart["storage"].next
        assert rotation_key != original_key
        stored2 = dict(vm_config2.volume_passphrases)

        # ---- Confirm restart (rotated=true — luksAddKey succeeded) ----
        _simulate_confirm(stored2, vol_restart, {"storage": True})

        # After promotion, current should be the rotation key
        assert fernet.decrypt(stored2["storage"].encode()).decode() == rotation_key
        assert "pending_storage" not in stored2

    @patch("api.server.util.settings")
    @patch("api.server.util._get_vm_cache_config")
    @patch("api.server.util._create_vm_cache_config")
    async def test_confirm_true_on_first_boot_also_safe(
        self, mock_create, mock_get, mock_settings
    ):
        """Even if confirm(rotated=true) is sent on first boot, passphrase survives."""
        fernet = Fernet(Fernet.generate_key())
        mock_settings.fernet_key = fernet
        db = AsyncMock()

        mock_get.return_value = None
        config = _make_vm_config()
        mock_create.return_value = config

        vol_boot, vm_config = await rotate_luks_passphrases(
            db, "hk", "vm1", ["storage"]
        )
        key = vol_boot["storage"].next
        stored = dict(vm_config.volume_passphrases)

        # Hypothetical: confirm with rotated=true (promotes pending → current)
        _simulate_confirm(stored, vol_boot, {"storage": True})

        assert "storage" in stored
        assert fernet.decrypt(stored["storage"].encode()).decode() == key
