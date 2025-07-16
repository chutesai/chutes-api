-- migrate:up
ALTER TABLE nodes ADD COLUMN IF NOT EXISTS server_id TEXT;

-- migrate:down
ALTER TABLE nodes DROP COLUMN IF EXISTS server_id;
