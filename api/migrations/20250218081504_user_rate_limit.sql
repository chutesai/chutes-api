-- migrate:up
ALTER TABLE users ADD COLUMN IF NOT EXISTS rate_limit_overrides JSONB DEFAULT NULL;

-- migrate:down
ALTER TABLE users DROP COLUMN IF EXISTS rate_limit_overrides;
