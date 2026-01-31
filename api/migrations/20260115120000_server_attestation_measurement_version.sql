-- migrate:up
ALTER TABLE server_attestations ADD COLUMN measurement_version TEXT;

-- migrate:down
ALTER TABLE server_attestations DROP COLUMN IF EXISTS measurement_version;
