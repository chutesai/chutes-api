-- migrate:up
ALTER TABLE boot_attestations ADD COLUMN measurement_version TEXT;
UPDATE boot_attestations SET measurement_version = '0.1.0' WHERE verified_at IS NOT NULL;

-- migrate:down
ALTER TABLE boot_attestations DROP COLUMN IF EXISTS measurement_version;
