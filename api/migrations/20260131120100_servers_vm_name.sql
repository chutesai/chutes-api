-- migrate:up
ALTER TABLE servers ADD COLUMN vm_name VARCHAR;

UPDATE servers SET vm_name = server_id WHERE vm_name IS NULL;

ALTER TABLE servers ALTER COLUMN vm_name SET NOT NULL;
CREATE UNIQUE INDEX idx_servers_miner_vm_name ON servers(miner_hotkey, vm_name);

-- migrate:down
DROP INDEX IF EXISTS idx_servers_miner_vm_name;
ALTER TABLE servers DROP COLUMN vm_name;
