-- migrate:up
ALTER TABLE metagraph_nodes ADD COLUMN IF NOT EXISTS tao_stake DOUBLE PRECISION NOT NULL DEFAULT 0.0;
ALTER TABLE metagraph_nodes ADD COLUMN IF NOT EXISTS alpha_stake DOUBLE PRECISION NOT NULL DEFAULT 0.0;

-- migrate:down
ALTER TABLE metagraph_nodes DROP COLUMN IF EXISTS tao_stake;
ALTER TABLE metagraph_nodes DROP COLUMN IF EXISTS alpha_stake;
