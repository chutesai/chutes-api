-- migrate:up
ALTER TABLE users ADD COLUMN IF NOT EXISTS wallet_secret TEXT NOT NULL DEFAULT '';

-- migrate:down
ALTER TABLE users DROP COLUMN IF EXISTS wallet_secret;
