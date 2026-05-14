-- migrate:up
ALTER TABLE vm_cache_configs ADD COLUMN k3s_encryption_key TEXT;

-- migrate:down
ALTER TABLE vm_cache_configs DROP COLUMN k3s_encryption_key;
