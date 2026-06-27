-- migrate:up

ALTER TABLE servers
    ADD COLUMN IF NOT EXISTS last_health_at TIMESTAMPTZ;

ALTER TABLE servers
    ADD COLUMN IF NOT EXISTS health_status VARCHAR NOT NULL DEFAULT 'unknown';

CREATE INDEX IF NOT EXISTS idx_servers_health_status
    ON servers (health_status);

CREATE INDEX IF NOT EXISTS idx_servers_last_health
    ON servers (last_health_at);

-- migrate:down

DROP INDEX IF EXISTS idx_servers_last_health;
DROP INDEX IF EXISTS idx_servers_health_status;

ALTER TABLE servers DROP COLUMN IF EXISTS health_status;
ALTER TABLE servers DROP COLUMN IF EXISTS last_health_at;
