-- migrate:up
create index if not exists idx_billing_ia on instance_audit (activated_at, stop_billing_at, billed_to, deleted_at);

-- trigger to track instance creation
CREATE OR REPLACE FUNCTION fn_instance_audit_insert()
RETURNS TRIGGER AS $$
DECLARE
    version TEXT;
BEGIN
    SELECT INTO version c.version 
      FROM chutes c 
     WHERE c.chute_id = NEW.chute_id;

    INSERT INTO instance_audit (
        instance_id,
        chute_id,
        version,
	miner_uid,
	miner_hotkey,
	region,
	billed_to,
	stop_billing_at,
	compute_multiplier
    ) VALUES (
        NEW.instance_id,
        NEW.chute_id,
        version,
	NEW.miner_uid,
	NEW.miner_hotkey,
	NEW.region,
	NEW.billed_to,
	NEW.stop_billing_at,
	NEW.compute_multiplier
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- trigger to track instance updates
CREATE OR REPLACE FUNCTION fn_instance_audit_update()
RETURNS TRIGGER AS $$
BEGIN
    -- Instance was verified.
    IF NEW.last_verified_at IS NOT NULL AND OLD.last_verified_at IS NULL THEN
        UPDATE instance_audit
           SET verified_at = NEW.last_verified_at
         WHERE instance_id = NEW.instance_id
           AND verified_at IS NULL;
    END IF;

    -- Instance was activated.
    IF NEW.activated_at IS NOT NULL AND OLD.activated_at IS NULL THEN
        UPDATE instance_audit
           SET activated_at = NEW.activated_at
         WHERE instance_id = NEW.instance_id
           AND activated_at IS NULL;
    END IF;

    -- Update billed_to (set when activated).
    IF NEW.billed_to IS DISTINCT FROM OLD.billed_to AND NEW.billed_to IS NOT NULL THEN
        UPDATE instance_audit
           SET billed_to = NEW.billed_to
         WHERE instance_id = NEW.instance_id;
    END IF;

    -- Update stop_billing_at when it changes (updated regularly and upon activation).
    IF NEW.stop_billing_at IS DISTINCT FROM OLD.stop_billing_at THEN
        UPDATE instance_audit
           SET stop_billing_at = NEW.stop_billing_at
         WHERE instance_id = NEW.instance_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- migrate:down
-- no-op here, backwards compatible...
