-- migrate:up
CREATE OR REPLACE FUNCTION report_missing_short_lived_instances()
RETURNS TABLE(
    instance_id UUID,
    invocations_reported INT
) AS $$
DECLARE
    rec RECORD;
    lifetime_seconds NUMERIC;
    reported_count INT;
BEGIN
    FOR rec IN
        SELECT 
            ia.instance_id,
            ia.deleted_at,
            ia.verified_at,
            ia.chute_id,
            EXTRACT(EPOCH FROM (ia.deleted_at - ia.verified_at)) AS lifetime
        FROM instance_audit ia
        WHERE ia.deleted_at >= NOW() - INTERVAL '1 hour'
            AND ia.deleted_at IS NOT NULL
            AND ia.verified_at IS NOT NULL
            AND EXTRACT(EPOCH FROM (ia.deleted_at - ia.verified_at)) < 3600
            AND ia.chute_id NOT IN (
                '35cfa8b4-13a2-5382-b19a-e849f73c5d6a',
                '83ce50c4-6d3f-55a6-88a6-c5db187f2c70',
                'eb04d6a6-b250-5f44-b91e-079bc938482a',
                'b5326e54-8d9e-590e-bed4-f3db35d9d4cd'
            )
    LOOP
        reported_count := 0;
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
            format('instance lifetime less than required minimum: (''%s'')', rec.lifetime) AS reason
        FROM invocations i
        WHERE i.instance_id = rec.instance_id
            AND i.started_at >= NOW() - INTERVAL '2 hours'
            AND NOT EXISTS (
                SELECT 1 
                FROM reports r 
                WHERE r.invocation_id = i.parent_invocation_id
            )
        ON CONFLICT (invocation_id) DO NOTHING;
        GET DIAGNOSTICS reported_count = ROW_COUNT;
        IF reported_count > 0 THEN
            instance_id := rec.instance_id;
            invocations_reported := reported_count;
            RETURN NEXT;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- migrate:down
DROP FUNCTION report_missing_short_lived_instances();
