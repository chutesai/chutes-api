-- migrate:up
ALTER TABLE images ADD COLUMN IF NOT EXISTS patch_version TEXT;

-- migrate:down
ALTER TABLE images DROP COLUMN IF EXISTS patch_version;
