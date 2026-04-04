-- migrate:up

CREATE TABLE IF NOT EXISTS tee_upgrade_windows (
    id VARCHAR PRIMARY KEY,
    upgrade_window_start TIMESTAMPTZ NOT NULL,
    upgrade_window_end TIMESTAMPTZ NOT NULL,
    target_measurement_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_tee_upgrade_target UNIQUE (target_measurement_version),
    CONSTRAINT chk_window_bounds CHECK (upgrade_window_end > upgrade_window_start)
);

CREATE INDEX IF NOT EXISTS idx_tee_upgrade_window_bounds
    ON tee_upgrade_windows (upgrade_window_start, upgrade_window_end);

ALTER TABLE servers
    ADD COLUMN IF NOT EXISTS maintenance_pending_window_id VARCHAR
        REFERENCES tee_upgrade_windows(id) ON DELETE SET NULL;

ALTER TABLE servers
    ADD COLUMN IF NOT EXISTS version TEXT;

CREATE INDEX IF NOT EXISTS idx_servers_maintenance_pending
    ON servers (miner_hotkey)
    WHERE maintenance_pending_window_id IS NOT NULL;

-- migrate:down

DROP INDEX IF EXISTS idx_servers_maintenance_pending;

ALTER TABLE servers DROP COLUMN IF EXISTS version;
ALTER TABLE servers DROP COLUMN IF EXISTS maintenance_pending_window_id;

DROP TABLE IF EXISTS tee_upgrade_windows;
