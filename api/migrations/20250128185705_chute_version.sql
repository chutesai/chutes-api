-- migrate:up
ALTER TABLE chutes ADD COLUMN IF NOT EXISTS chutes_version TEXT;
ALTER TABLE images ADD COLUMN IF NOT EXISTS chutes_version TEXT;
ALTER TABLE instances ADD COLUMN IF NOT EXISTS chutes_version TEXT;

-- migrate:down
ALTER TABLE chutes DROP COLUMN IF EXISTS chutes_version;
ALTER TABLE images DROP COLUMN IF EXISTS chutes_version;
ALTER TABLE instances DROP COLUMN IF EXISTS chutes_version;
