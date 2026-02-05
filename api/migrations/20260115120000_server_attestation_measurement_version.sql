-- migrate:up
ALTER TABLE server_attestations ADD COLUMN measurement_version TEXT;
UPDATE server_attestations SET measurement_version = '0.1.0' WHERE verified_at IS NOT NULL;

-- migrate:down
ALTER TABLE server_attestations DROP COLUMN IF EXISTS measurement_version;
