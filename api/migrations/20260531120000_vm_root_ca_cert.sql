-- migrate:up
ALTER TABLE servers ADD COLUMN vm_root_ca_cert TEXT;

-- migrate:down
ALTER TABLE servers DROP COLUMN vm_root_ca_cert;
