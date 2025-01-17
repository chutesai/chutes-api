-- migrate:up
ALTER TABLE instances ADD COLUMN IF NOT EXISTS verification_error TEXT;

-- migrate:down
ALTER TABLE instances DROP COLUMN IF EXISTS verification_error;
