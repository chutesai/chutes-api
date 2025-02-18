-- migrate:up
ALTER TABLE chutes ADD COLUMN IF NOT EXISTS openrouter BOOLEAN NOT NULL DEFAULT false;

-- migrate:down
ALTER TABLE chutes DROP COLUMN IF EXISTS openrouter;
