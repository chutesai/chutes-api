-- migrate:up
ALTER TABLE chutes ADD COLUMN IF NOT EXISTS tool_description TEXT;

-- migrate:down
ALTER TABLE chutes DROP COLUMN IF EXISTS tool_description;
