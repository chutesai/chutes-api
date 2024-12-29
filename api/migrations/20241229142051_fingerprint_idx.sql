-- migrate:up
CREATE INDEX idx_user_fingerprint_hash ON users (fingerprint_hash);

-- migrate:down
DROP INDEX IF EXISTS idx_user_fingerprint_hash;
