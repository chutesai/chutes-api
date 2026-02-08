-- migrate:up
ALTER TABLE instances ADD COLUMN IF NOT EXISTS deployment_id TEXT;

-- migrate:down
ALTER TABLE instances DROP COLUMN IF EXISTS deployment_id;
