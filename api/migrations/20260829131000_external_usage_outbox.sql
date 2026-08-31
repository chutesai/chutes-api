-- migrate:up

CREATE TABLE IF NOT EXISTS external_usage_outbox (
    event_id VARCHAR(255) PRIMARY KEY,
    operation_id VARCHAR NOT NULL UNIQUE
        REFERENCES external_operations(operation_id) ON DELETE RESTRICT,
    user_id VARCHAR NOT NULL,
    chute_id VARCHAR NOT NULL,
    app_id VARCHAR,
    amount NUMERIC(30, 12) NOT NULL,
    paygo_amount NUMERIC(30, 12) NOT NULL,
    input_tokens NUMERIC(30, 6) NOT NULL DEFAULT 0,
    output_tokens NUMERIC(30, 6) NOT NULL DEFAULT 0,
    cached_tokens NUMERIC(30, 6) NOT NULL DEFAULT 0,
    compute_time DOUBLE PRECISION NOT NULL DEFAULT 0,
    track_task_completion BOOLEAN NOT NULL DEFAULT FALSE,
    free_invocation BOOLEAN NOT NULL DEFAULT FALSE,
    increment_invocation_quota BOOLEAN NOT NULL DEFAULT FALSE,
    occurred_at TIMESTAMPTZ NOT NULL,
    attempts BIGINT NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ,
    last_error_code VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_external_usage_outbox_amount CHECK (amount >= 0),
    CONSTRAINT ck_external_usage_outbox_paygo_amount CHECK (paygo_amount >= 0),
    CONSTRAINT ck_external_usage_outbox_tokens CHECK (
        input_tokens >= 0 AND output_tokens >= 0 AND cached_tokens >= 0
    ),
    CONSTRAINT ck_external_usage_outbox_compute_time CHECK (compute_time >= 0),
    CONSTRAINT ck_external_usage_outbox_attempts CHECK (attempts >= 0)
);

CREATE INDEX IF NOT EXISTS idx_external_usage_outbox_due
    ON external_usage_outbox (next_attempt_at, created_at);

-- migrate:down

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM external_usage_outbox)
       OR EXISTS (SELECT 1 FROM external_operations) THEN
        RAISE EXCEPTION
            'Cannot downgrade while external operations or usage charges exist';
    END IF;
END $$;

DROP TABLE IF EXISTS external_usage_outbox;
