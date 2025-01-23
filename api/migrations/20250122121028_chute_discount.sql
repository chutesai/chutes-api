-- migrate:up
ALTER TABLE chutes ADD COLUMN IF NOT EXISTS discount FLOAT DEFAULT 0.0;

-- migrate:down
ALTER TABLE chutes DROP COLUMN IF EXISTS discount;
