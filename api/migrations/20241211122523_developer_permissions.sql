-- migrate:up
ALTER TABLE users ADD COLUMN IF NOT EXISTS permissions BIGINT NOT NULL DEFAULT 0;
ALTER TABLE users ADD COLUMN IF NOT EXISTS developer_payment_address NOT NULL DEFAULT '';
ALTER TABLE users ADD COLUMN IF NOT EXISTS developer_wallet_secret NOT NULL DEFAULT '';

-- migrate:down
ALTER TABLE users DROP COLUMN IF EXISTS permissions;
ALTER TABLE users DROP COLUMN IF EXISTS developer_payment_address;
ALTER TABLE users DROP COLUMN IF EXISTS developer_wallet_secret;
