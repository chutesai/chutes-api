-- migrate:up
ALTER TABLE users ADD COLUMN IF NOT EXISTS validator_hotkey TEXT;

-- migrate:down
ALTER TABLE users DROP COLUMN IF EXISTS validator_hotkey;
