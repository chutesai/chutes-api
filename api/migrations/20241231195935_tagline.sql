-- migrate:up
ALTER TABLE chutes ADD COLUMN IF NOT EXISTS tagline TEXT NOT NULL DEFAULT '';

-- migrate:down
ALTER TABLE chutes DROP COLUMN IF EXISTS tagline;
