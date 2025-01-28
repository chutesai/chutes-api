-- migrate:up
ALTER TABLE instances ADD COLUMN IF NOT EXISTS symmetric_key TEXT;

-- migrate:down
ALTER TABLE instances DROP COLUMN IF EXISTS symmetric_key;
