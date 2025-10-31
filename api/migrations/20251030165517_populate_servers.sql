-- migrate:up

-- Backfill servers table from existing nodes
INSERT INTO servers (server_id, ip, miner_hotkey, created_at, updated_at)
SELECT DISTINCT 
    n.server_id,
    n.verification_host AS ip,
    n.miner_hotkey,
    n.created_at,
    n.created_at
FROM nodes n
WHERE n.server_id IS NOT NULL
    AND n.miner_hotkey IS NOT NULL
ON CONFLICT (server_id) DO NOTHING;

-- migrate:down

-- Nothing to do --