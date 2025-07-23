-- migrate:up
CREATE OR REPLACE FUNCTION auto_report_short_lived_instances()
RETURNS TRIGGER AS $$
DECLARE
    lifetime_seconds NUMERIC;
    invocation RECORD;
BEGIN
    IF NEW.chute_id = '35cfa8b4-13a2-5382-b19a-e849f73c5d6a' OR NEW.chute_id = '83ce50c4-6d3f-55a6-88a6-c5db187f2c70' THEN
        RETURN NEW;
    END IF;
    IF NEW.deleted_at IS NOT NULL AND OLD.deleted_at IS NULL AND NEW.verified_at IS NOT NULL THEN
        lifetime_seconds := EXTRACT(EPOCH FROM (NEW.deleted_at - NEW.verified_at));
        IF lifetime_seconds < 3600 THEN
            INSERT INTO reports (
                invocation_id,
                user_id,
                timestamp,
                confirmed_at,
                confirmed_by,
                reason
            )
            SELECT 
                i.parent_invocation_id AS invocation_id,
                'dff3e6bb-3a6b-5a2b-9c48-da3abcd5ca5f' AS user_id,
                NOW() AS timestamp,
                NOW() AS confirmed_at,
                'dff3e6bb-3a6b-5a2b-9c48-da3abcd5ca5f' AS confirmed_by,
                format('instance lifetime less than required minimum: (''%s'')', lifetime_seconds) AS reason
            FROM invocations i
            WHERE i.instance_id = NEW.instance_id
	    AND started_at >= now() - INTERVAL '2 hours'
	    ON CONFLICT (invocation_id) DO NOTHING;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_auto_report_short_lived_instances ON instance_audit;
CREATE TRIGGER trigger_auto_report_short_lived_instances
    BEFORE UPDATE ON instance_audit
    FOR EACH ROW
    EXECUTE FUNCTION auto_report_short_lived_instances();

INSERT INTO reports (
    invocation_id,
    user_id,
    timestamp,
    confirmed_at,
    confirmed_by,
    reason
)
SELECT 
    i.parent_invocation_id AS invocation_id,
    'dff3e6bb-3a6b-5a2b-9c48-da3abcd5ca5f' AS user_id,
    NOW() AS timestamp,
    NOW() AS confirmed_at,
    'dff3e6bb-3a6b-5a2b-9c48-da3abcd5ca5f' AS confirmed_by,
    format('instance lifetime less than required minimum: (''%s'')', EXTRACT(EPOCH FROM (ia.deleted_at - ia.verified_at))) AS reason
FROM instance_audit ia
INNER JOIN invocations i ON i.instance_id = ia.instance_id
WHERE ia.deleted_at >= NOW() - INTERVAL '7 days'
  AND ia.deleted_at IS NOT NULL
  AND ia.verified_at IS NOT NULL
  AND EXTRACT(EPOCH FROM (ia.deleted_at - ia.verified_at)) < 3600
  AND i.started_at >= now() - INTERVAL '7 days'
ON CONFLICT (invocation_id) DO NOTHING;

-- migrate:down
DROP TRIGGER IF EXISTS trigger_auto_report_short_lived_instances ON instance_audit;
DROP FUNCTION IF EXISTS auto_report_short_lived_instances();
