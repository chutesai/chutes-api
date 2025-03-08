-- migrate:up
ALTER TABLE payments ADD COLUMN IF NOT EXISTS source_address TEXT;

-- migrate:down
ALTER TABLE payments DROP COLUMN IF EXISTS source_address;
