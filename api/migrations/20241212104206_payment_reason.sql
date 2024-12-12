-- migrate:up
ALTER TABLE payments ADD COLUMN IF NOT EXISTS purpose TEXT NOT NULL DEFAULT 'credits';

-- migrate:down
ALTER TABLE payments DROP COLUMN IF EXISTS purpose;
