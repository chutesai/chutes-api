-- migrate:up

ALTER TABLE external_operations
    DROP CONSTRAINT ck_external_operations_settlement_status;

ALTER TABLE external_operations
    ADD CONSTRAINT ck_external_operations_settlement_status
    CHECK (
        settlement_status IN (
            'pending', 'settled', 'not_billable', 'failed', 'quarantined'
        )
    );

-- migrate:down

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM external_operations
        WHERE settlement_status = 'quarantined'
    ) THEN
        RAISE EXCEPTION
            'Cannot downgrade while quarantined external settlements exist';
    END IF;
END $$;

ALTER TABLE external_operations
    DROP CONSTRAINT ck_external_operations_settlement_status;

ALTER TABLE external_operations
    ADD CONSTRAINT ck_external_operations_settlement_status
    CHECK (settlement_status IN ('pending', 'settled', 'not_billable', 'failed'));
