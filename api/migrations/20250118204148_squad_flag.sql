-- migrate:up
ALTER TABLE users ADD COLUMN IF NOT EXISTS squad_enabled BOOLEAN DEFAULT false;

-- migrate:down
ALTER TABLE users DROP COLUMN IF EXISTS squad_enabled;
