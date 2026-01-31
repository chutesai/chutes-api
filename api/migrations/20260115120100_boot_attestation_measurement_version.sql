-- migrate:up
ALTER TABLE boot_attestations ADD COLUMN measurement_version TEXT;

-- migrate:down
ALTER TABLE boot_attestations DROP COLUMN IF EXISTS measurement_version;
