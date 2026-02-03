-- migrate:up
ALTER TABLE servers ADD COLUMN name VARCHAR;

UPDATE servers SET name = server_id WHERE name IS NULL;

ALTER TABLE servers ALTER COLUMN name SET NOT NULL;
CREATE UNIQUE INDEX idx_servers_miner_name ON servers(miner_hotkey, name);

-- migrate:down
DROP INDEX IF EXISTS idx_servers_miner_name;
ALTER TABLE servers DROP COLUMN name;
