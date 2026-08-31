-- migrate:up

ALTER TABLE external_backend_accounts
    ADD COLUMN management_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN artifact_relay_invalidated_at TIMESTAMPTZ;

ALTER TABLE external_backend_accounts
    ADD CONSTRAINT ck_external_backend_accounts_management_metadata
    CHECK (jsonb_typeof(management_metadata) = 'object');

-- migrate:down

ALTER TABLE external_backend_accounts
    DROP CONSTRAINT ck_external_backend_accounts_management_metadata,
    DROP COLUMN artifact_relay_invalidated_at,
    DROP COLUMN management_metadata;
