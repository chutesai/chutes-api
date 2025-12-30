-- migrate:up
CREATE TABLE vm_cache_configs (
    miner_hotkey VARCHAR NOT NULL,
    vm_name VARCHAR NOT NULL,
    encrypted_passphrase VARCHAR NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE,
    last_boot_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (miner_hotkey, vm_name)
);

CREATE INDEX idx_vm_cache_miner ON vm_cache_configs(miner_hotkey);
CREATE INDEX idx_vm_cache_last_boot ON vm_cache_configs(last_boot_at);

-- migrate:down
DROP TABLE vm_cache_configs;
