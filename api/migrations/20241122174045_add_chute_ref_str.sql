-- migrate:up
ALTER TABLE chutes ADD COLUMN IF NOT EXISTS ref_str TEXT NOT NULL DEFAULT 'entrypoint:chute';

-- migrate:down
ALTER TABLE chutes DROP COLUMN IF EXISTS ref_str;
