-- migrate:up
ALTER TABLE vm_cache_configs ADD COLUMN volume_passphrases JSONB;

UPDATE vm_cache_configs
SET volume_passphrases = jsonb_build_object('storage', encrypted_passphrase)
WHERE encrypted_passphrase IS NOT NULL;

ALTER TABLE vm_cache_configs ALTER COLUMN volume_passphrases SET NOT NULL;
ALTER TABLE vm_cache_configs DROP COLUMN encrypted_passphrase;

-- migrate:down
ALTER TABLE vm_cache_configs ADD COLUMN encrypted_passphrase VARCHAR;

UPDATE vm_cache_configs
SET encrypted_passphrase = volume_passphrases->>'storage'
WHERE volume_passphrases ? 'storage';

ALTER TABLE vm_cache_configs DROP COLUMN volume_passphrases;
