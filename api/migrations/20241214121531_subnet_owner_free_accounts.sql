-- migrate:up
ALTER TABLE users ADD COLUMN IF NOT EXISTS subnet_owner_hotkey TEXT;

-- migrate:down
ALTER TABLE users DROP COLUMN IF EXISTS subnet_owner_hotkey;
