-- migrate:up
ALTER TABLE users ADD COLUMN IF NOT EXISTS quotas JSONB;

-- migrate:down
ALTER TABLE users DROP COLUMN IF EXISTS quotas;
