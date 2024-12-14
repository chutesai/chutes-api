-- migrate:up
ALTER TABLE instances ADD COLUMN IF NOT EXISTS consecutive_failures NOT NULL DEFAULT 0;

-- migrate:down
ALTER TABLE instances DROP COLUMN IF EXISTS consecutive_failures;
