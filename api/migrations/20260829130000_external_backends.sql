-- migrate:up

ALTER TABLE secrets
    ADD COLUMN IF NOT EXISTS kind VARCHAR(32) NOT NULL DEFAULT 'chute';
UPDATE secrets SET kind = 'chute' WHERE kind IS NULL;
ALTER TABLE secrets ALTER COLUMN kind SET DEFAULT 'chute';
ALTER TABLE secrets ALTER COLUMN kind SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'ck_secrets_kind'
          AND conrelid = 'secrets'::regclass
    ) THEN
        ALTER TABLE secrets
            ADD CONSTRAINT ck_secrets_kind
            CHECK (kind IN ('chute', 'external_backend'));
    END IF;
END $$;

ALTER TABLE chutes
    ADD COLUMN IF NOT EXISTS execution_backend VARCHAR(16) NOT NULL DEFAULT 'hosted';

ALTER TABLE chutes ALTER COLUMN image_id DROP NOT NULL;
ALTER TABLE chute_history ALTER COLUMN image_id DROP NOT NULL;
ALTER TABLE chutes ALTER COLUMN code DROP NOT NULL;
ALTER TABLE chutes ALTER COLUMN filename DROP NOT NULL;
ALTER TABLE chutes ALTER COLUMN ref_str DROP NOT NULL;
ALTER TABLE chute_history ALTER COLUMN code DROP NOT NULL;
ALTER TABLE chute_history ALTER COLUMN filename DROP NOT NULL;
ALTER TABLE chute_history ALTER COLUMN ref_str DROP NOT NULL;

-- ``disabled`` pre-dates this migration and historically allowed NULL. Public
-- response models and all execution checks treat it as a concrete boolean, so
-- normalize legacy rows before enforcing that invariant for both backends.
UPDATE chutes SET disabled = FALSE WHERE disabled IS NULL;
ALTER TABLE chutes ALTER COLUMN disabled SET DEFAULT FALSE;
ALTER TABLE chutes ALTER COLUMN disabled SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'ck_chutes_execution_backend'
          AND conrelid = 'chutes'::regclass
    ) THEN
        ALTER TABLE chutes
            ADD CONSTRAINT ck_chutes_execution_backend
            CHECK (execution_backend IN ('hosted', 'external'));
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'ck_chutes_execution_backend_shape'
          AND conrelid = 'chutes'::regclass
    ) THEN
        ALTER TABLE chutes
            ADD CONSTRAINT ck_chutes_execution_backend_shape
            CHECK (
                (
                    execution_backend = 'hosted'
                    AND image_id IS NOT NULL
                    AND code IS NOT NULL
                    AND filename IS NOT NULL
                    AND ref_str IS NOT NULL
                )
                OR
                (
                    execution_backend = 'external'
                    AND image_id IS NULL
                    AND code IS NULL
                    AND filename IS NULL
                    AND ref_str IS NULL
                )
            );
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS external_backend_accounts (
    account_id VARCHAR PRIMARY KEY,
    user_id VARCHAR NOT NULL REFERENCES users(user_id) ON DELETE RESTRICT,
    name VARCHAR(128) NOT NULL,
    adapter VARCHAR(64) NOT NULL,
    base_url VARCHAR(2048) NOT NULL,
    credential_references JSONB NOT NULL,
    auth_header_templates JSONB NOT NULL,
    connection_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_external_backend_accounts_user_name UNIQUE (user_id, name),
    CONSTRAINT ck_external_backend_accounts_adapter
        CHECK (adapter ~ '^[a-z][a-z0-9._-]{0,63}$'),
    CONSTRAINT ck_external_backend_accounts_credential_references
        CHECK (
            jsonb_typeof(credential_references) = 'object'
            AND credential_references <> '{}'::jsonb
        ),
    CONSTRAINT ck_external_backend_accounts_auth_header_templates
        CHECK (
            jsonb_typeof(auth_header_templates) = 'array'
            AND jsonb_array_length(auth_header_templates) > 0
        ),
    CONSTRAINT ck_external_backend_accounts_base_url
        CHECK (
            base_url ~ '^https?://'
            AND base_url !~ '[?#]'
            AND base_url !~ '^https?://[^/]*@'
        ),
    CONSTRAINT ck_external_backend_accounts_connection_config
        CHECK (jsonb_typeof(connection_config) = 'object'),
    CONSTRAINT ck_external_backend_accounts_no_inline_credentials
        CHECK (
            NOT connection_config ?| ARRAY[
                'access_key', 'access_token', 'api_key', 'auth_token',
                'authorization', 'bearer_token', 'client_secret', 'credential',
                'credentials', 'password', 'private_key', 'secret', 'secret_key'
            ]
        )
);

CREATE TABLE IF NOT EXISTS external_chute_bindings (
    binding_id VARCHAR PRIMARY KEY,
    chute_id VARCHAR NOT NULL REFERENCES chutes(chute_id) ON DELETE CASCADE,
    account_id VARCHAR NOT NULL REFERENCES external_backend_accounts(account_id) ON DELETE RESTRICT,
    routes JSONB NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_external_chute_bindings_chute_id UNIQUE (chute_id),
    CONSTRAINT ck_external_chute_bindings_routes
        CHECK (jsonb_typeof(routes) = 'array' AND jsonb_array_length(routes) > 0)
);

CREATE INDEX IF NOT EXISTS idx_external_chute_bindings_account
    ON external_chute_bindings (account_id);

CREATE TABLE IF NOT EXISTS external_operations (
    operation_id VARCHAR PRIMARY KEY,
    user_id VARCHAR REFERENCES users(user_id) ON DELETE SET NULL,
    account_id VARCHAR REFERENCES external_backend_accounts(account_id) ON DELETE SET NULL,
    binding_id VARCHAR REFERENCES external_chute_bindings(binding_id) ON DELETE SET NULL,
    chute_id VARCHAR REFERENCES chutes(chute_id) ON DELETE SET NULL,
    cord_path VARCHAR(255) NOT NULL,
    operation_mode VARCHAR(16) NOT NULL,
    protocol VARCHAR(64) NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    settlement_status VARCHAR(16) NOT NULL DEFAULT 'pending',
    upstream_operation_id VARCHAR(512),
    upstream_status VARCHAR(128),
    idempotency_key VARCHAR(255),
    route_snapshot JSONB NOT NULL,
    request_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    upstream_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    usage JSONB,
    result_descriptor JSONB,
    error JSONB,
    settlement_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    poll_attempts INTEGER NOT NULL DEFAULT 0,
    lease_owner VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    submitted_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    last_polled_at TIMESTAMPTZ,
    next_poll_at TIMESTAMPTZ,
    lease_expires_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    settled_at TIMESTAMPTZ,
    CONSTRAINT ck_external_operations_operation_mode
        CHECK (operation_mode IN ('sync', 'stream', 'task', 'realtime')),
    CONSTRAINT ck_external_operations_protocol
        CHECK (protocol ~ '^[a-z][a-z0-9._-]{0,63}$'),
    CONSTRAINT ck_external_operations_status
        CHECK (
            status IN (
                'pending', 'submitted', 'running', 'succeeded', 'failed',
                'cancelled', 'expired'
            )
        ),
    CONSTRAINT ck_external_operations_settlement_status
        CHECK (settlement_status IN ('pending', 'settled', 'not_billable', 'failed')),
    CONSTRAINT ck_external_operations_poll_attempts CHECK (poll_attempts >= 0),
    CONSTRAINT ck_external_operations_request_metadata
        CHECK (jsonb_typeof(request_metadata) = 'object'),
    CONSTRAINT ck_external_operations_route_snapshot
        CHECK (jsonb_typeof(route_snapshot) = 'object'),
    CONSTRAINT ck_external_operations_upstream_metadata
        CHECK (jsonb_typeof(upstream_metadata) = 'object'),
    CONSTRAINT ck_external_operations_usage
        CHECK (usage IS NULL OR jsonb_typeof(usage) = 'object'),
    CONSTRAINT ck_external_operations_result_descriptor
        CHECK (result_descriptor IS NULL OR jsonb_typeof(result_descriptor) = 'object'),
    CONSTRAINT ck_external_operations_error
        CHECK (error IS NULL OR jsonb_typeof(error) = 'object'),
    CONSTRAINT ck_external_operations_settlement_metadata
        CHECK (jsonb_typeof(settlement_metadata) = 'object')
);

CREATE INDEX IF NOT EXISTS idx_external_operations_user_created
    ON external_operations (user_id, created_at);

CREATE INDEX IF NOT EXISTS idx_external_operations_account_created
    ON external_operations (account_id, created_at);

CREATE INDEX IF NOT EXISTS idx_external_operations_poll
    ON external_operations (status, next_poll_at);

CREATE INDEX IF NOT EXISTS idx_external_operations_account_status
    ON external_operations (account_id, status);

CREATE INDEX IF NOT EXISTS idx_external_operations_settlement_retry
    ON external_operations (settlement_status, next_poll_at)
    WHERE status IN ('succeeded', 'failed', 'cancelled', 'expired')
      AND settlement_status IN ('pending', 'failed');

CREATE INDEX IF NOT EXISTS idx_external_operations_binding
    ON external_operations (binding_id)
    WHERE binding_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_external_operations_chute
    ON external_operations (chute_id)
    WHERE chute_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_external_operations_upstream_id
    ON external_operations (binding_id, upstream_operation_id)
    WHERE upstream_operation_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_external_operations_idempotency_key
    ON external_operations (binding_id, user_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

-- migrate:down

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM chutes WHERE execution_backend = 'external'
    ) THEN
        RAISE EXCEPTION
            'Cannot downgrade while external Chutes exist; delete or migrate them first';
    END IF;
    IF EXISTS (
        SELECT 1 FROM external_backend_accounts
    ) OR EXISTS (
        SELECT 1 FROM secrets WHERE kind = 'external_backend'
    ) THEN
        RAISE EXCEPTION
            'Cannot downgrade while external backend accounts or credentials exist';
    END IF;
    IF EXISTS (
        SELECT 1 FROM external_operations
    ) THEN
        RAISE EXCEPTION
            'Cannot downgrade while external operation history exists';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM chutes
        WHERE image_id IS NULL OR code IS NULL OR filename IS NULL OR ref_str IS NULL
    ) OR EXISTS (
        SELECT 1
        FROM chute_history
        WHERE image_id IS NULL OR code IS NULL OR filename IS NULL OR ref_str IS NULL
    ) THEN
        RAISE EXCEPTION
            'Cannot downgrade while source-less Chute rows exist';
    END IF;
END $$;

DROP TABLE IF EXISTS external_operations;
DROP TABLE IF EXISTS external_chute_bindings;
DROP TABLE IF EXISTS external_backend_accounts;

ALTER TABLE chutes DROP CONSTRAINT IF EXISTS ck_chutes_execution_backend;
ALTER TABLE chutes DROP CONSTRAINT IF EXISTS ck_chutes_execution_backend_shape;
ALTER TABLE chutes DROP COLUMN IF EXISTS execution_backend;

ALTER TABLE chutes ALTER COLUMN image_id SET NOT NULL;
ALTER TABLE chutes ALTER COLUMN code SET NOT NULL;
ALTER TABLE chutes ALTER COLUMN filename SET NOT NULL;
ALTER TABLE chutes ALTER COLUMN ref_str SET NOT NULL;
ALTER TABLE chute_history ALTER COLUMN image_id SET NOT NULL;
ALTER TABLE chute_history ALTER COLUMN code SET NOT NULL;
ALTER TABLE chute_history ALTER COLUMN filename SET NOT NULL;
ALTER TABLE chute_history ALTER COLUMN ref_str SET NOT NULL;

ALTER TABLE chutes ALTER COLUMN disabled DROP NOT NULL;
ALTER TABLE chutes ALTER COLUMN disabled DROP DEFAULT;

ALTER TABLE secrets DROP CONSTRAINT IF EXISTS ck_secrets_kind;
ALTER TABLE secrets DROP COLUMN IF EXISTS kind;
