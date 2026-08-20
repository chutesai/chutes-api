-- migrate:up
-- Broaden the boot-attestation table into the pre-server "vm boot record": one row per boot
-- (append, preserving all existing attestation history), each capturing that boot's full initramfs
-- lifecycle -- the boot quote AND, once /provision runs, the runtime quote + the per-boot VM root
-- CA recorded in fully-measured initramfs (before the miner registers the server via POST /servers).
--
-- In place (no data loss): rename the table, rename quote_data -> boot_quote, and add the new
-- columns. The boot vs provision distinction is which quote column is set, so no phase discriminator
-- is needed. Existing indexes (idx_boot_*) carry over with the table.
ALTER TABLE boot_attestations RENAME TO vm_boot_records;
ALTER TABLE vm_boot_records RENAME COLUMN quote_data TO boot_quote;
ALTER TABLE vm_boot_records ADD COLUMN provision_quote TEXT;
ALTER TABLE vm_boot_records ADD COLUMN vm_root_ca_cert TEXT;
-- The luks_quote_nonce minted at /boot/attestation and consumed at /provision -- ties the two
-- calls of one boot to the same row (deterministic, not by timestamp).
ALTER TABLE vm_boot_records ADD COLUMN provision_nonce TEXT;
ALTER TABLE vm_boot_records ADD COLUMN updated_at TIMESTAMPTZ;

-- migrate:down
ALTER TABLE vm_boot_records DROP COLUMN updated_at;
ALTER TABLE vm_boot_records DROP COLUMN provision_nonce;
ALTER TABLE vm_boot_records DROP COLUMN vm_root_ca_cert;
ALTER TABLE vm_boot_records DROP COLUMN provision_quote;
ALTER TABLE vm_boot_records RENAME COLUMN boot_quote TO quote_data;
ALTER TABLE vm_boot_records RENAME TO boot_attestations;
