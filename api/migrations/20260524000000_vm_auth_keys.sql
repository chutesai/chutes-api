-- migrate:up
CREATE TABLE vm_auth_keys (
    miner_hotkey TEXT NOT NULL,
    vm_name TEXT NOT NULL,
    auth_seed TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (miner_hotkey, vm_name)
);
CREATE INDEX idx_vm_auth_keys_miner ON vm_auth_keys (miner_hotkey);

-- migrate:down
DROP TABLE IF EXISTS vm_auth_keys;
