-- migrate:up

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

-- Partial index to keep "live user" lookups fast now that most queries filter deleted_at IS NULL.
CREATE INDEX IF NOT EXISTS idx_users_active
    ON users (user_id)
    WHERE deleted_at IS NULL;

-- migrate:down

DROP INDEX IF EXISTS idx_users_active;

ALTER TABLE users DROP COLUMN IF EXISTS deleted_at;
