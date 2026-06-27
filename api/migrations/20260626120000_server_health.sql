-- migrate:up

ALTER TABLE servers
    ADD COLUMN IF NOT EXISTS last_health_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_servers_last_health
    ON servers (last_health_at);

-- migrate:down

DROP INDEX IF EXISTS idx_servers_last_health;

ALTER TABLE servers DROP COLUMN IF EXISTS last_health_at;
