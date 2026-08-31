-- migrate:up

CREATE TABLE external_governance_state (
    scope_type VARCHAR(16) NOT NULL,
    scope_id VARCHAR NOT NULL,
    active_tasks BIGINT NOT NULL DEFAULT 0,
    active_sync_requests BIGINT NOT NULL DEFAULT 0,
    active_realtime BIGINT NOT NULL DEFAULT 0,
    active_streams BIGINT NOT NULL DEFAULT 0,
    unresolved_paygo NUMERIC NOT NULL DEFAULT 0,
    unresolved_charge NUMERIC NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (scope_type, scope_id),
    CONSTRAINT ck_external_governance_state_scope
        CHECK (scope_type IN ('user', 'account')),
    CONSTRAINT ck_external_governance_state_counts
        CHECK (
            active_tasks >= 0
            AND active_sync_requests >= 0
            AND active_realtime >= 0
            AND active_streams >= 0
        ),
    CONSTRAINT ck_external_governance_state_amounts
        CHECK (unresolved_paygo >= 0 AND unresolved_charge >= 0)
);

CREATE TABLE external_governance_buckets (
    scope_type VARCHAR(16) NOT NULL,
    scope_id VARCHAR NOT NULL,
    bucket_start TIMESTAMPTZ NOT NULL,
    operation_count BIGINT NOT NULL DEFAULT 0,
    unresolved_paygo NUMERIC NOT NULL DEFAULT 0,
    settled_paygo NUMERIC NOT NULL DEFAULT 0,
    artifact_relay_bytes BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (scope_type, scope_id, bucket_start),
    CONSTRAINT ck_external_governance_buckets_scope
        CHECK (scope_type IN ('user', 'account')),
    CONSTRAINT ck_external_governance_buckets_values
        CHECK (
            operation_count >= 0
            AND unresolved_paygo >= 0
            AND settled_paygo >= 0
            AND artifact_relay_bytes >= 0
        )
);

CREATE INDEX idx_external_governance_buckets_expiration
    ON external_governance_buckets (bucket_start);

CREATE FUNCTION external_governance_money(value TEXT)
RETURNS NUMERIC
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
AS $$
DECLARE
    amount NUMERIC;
BEGIN
    IF value IS NULL
       OR length(value) > 128
       OR value !~ '^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]{1,4})?$' THEN
        RETURN NULL;
    END IF;
    amount := value::numeric;
    IF amount >= 0 THEN
        RETURN amount;
    END IF;
    RETURN NULL;
EXCEPTION
    WHEN numeric_value_out_of_range OR invalid_text_representation THEN
        RETURN NULL;
END;
$$;

CREATE FUNCTION external_governance_paygo(
    operation_status VARCHAR,
    settlement_state VARCHAR,
    metadata JSONB
)
RETURNS NUMERIC
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT CASE
        WHEN settlement_state = 'not_billable' THEN 0::numeric
        WHEN external_governance_money(metadata->'result'->>'paygo_amount') IS NOT NULL
            THEN external_governance_money(metadata->'result'->>'paygo_amount')
        WHEN external_governance_money(metadata->'result'->>'amount') IS NOT NULL
            THEN external_governance_money(metadata->'result'->>'amount')
        WHEN external_governance_money(metadata->>'observed_cost_estimate') IS NOT NULL
            THEN external_governance_money(metadata->>'observed_cost_estimate')
        WHEN operation_status IN ('pending', 'submitted', 'running')
             OR settlement_state IN ('pending', 'failed', 'quarantined')
            THEN COALESCE(
                external_governance_money(metadata->>'admission_cost_estimate'),
                0::numeric
            )
        ELSE 0::numeric
    END
$$;

CREATE FUNCTION external_governance_apply_state(
    requested_scope_type VARCHAR,
    requested_scope_id VARCHAR,
    task_delta BIGINT,
    sync_delta BIGINT,
    realtime_delta BIGINT,
    stream_delta BIGINT,
    paygo_delta NUMERIC,
    charge_delta NUMERIC
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    IF requested_scope_id IS NULL OR (
        task_delta = 0
        AND sync_delta = 0
        AND realtime_delta = 0
        AND stream_delta = 0
        AND paygo_delta = 0
        AND charge_delta = 0
    ) THEN
        RETURN;
    END IF;
    INSERT INTO external_governance_state (scope_type, scope_id)
    VALUES (requested_scope_type, requested_scope_id)
    ON CONFLICT (scope_type, scope_id) DO NOTHING;

    UPDATE external_governance_state
    SET active_tasks = active_tasks + task_delta,
        active_sync_requests = active_sync_requests + sync_delta,
        active_realtime = active_realtime + realtime_delta,
        active_streams = active_streams + stream_delta,
        unresolved_paygo = unresolved_paygo + paygo_delta,
        unresolved_charge = unresolved_charge + charge_delta,
        updated_at = NOW()
    WHERE scope_type = requested_scope_type AND scope_id = requested_scope_id;
END;
$$;

CREATE FUNCTION external_governance_apply_bucket(
    requested_scope_type VARCHAR,
    requested_scope_id VARCHAR,
    occurred_at TIMESTAMPTZ,
    operation_delta BIGINT,
    settled_paygo_delta NUMERIC,
    unresolved_paygo_delta NUMERIC
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    requested_bucket TIMESTAMPTZ;
BEGIN
    IF requested_scope_id IS NULL
       OR occurred_at IS NULL
       OR (
           operation_delta = 0
           AND settled_paygo_delta = 0
           AND unresolved_paygo_delta = 0
       )
       OR date_trunc('minute', occurred_at) < date_trunc(
            'minute', clock_timestamp() - INTERVAL '24 hours'
       ) THEN
        RETURN;
    END IF;
    requested_bucket := date_trunc('minute', occurred_at);

    -- A deletion can race bounded maintenance after the bucket has aged out.
    -- Never recreate a pruned row for a negative-only delta, and clamp an
    -- unexpectedly short existing row so clock skew or drift cannot make an
    -- otherwise valid operation mutation fail its nonnegative CHECK.
    IF operation_delta <= 0
       AND settled_paygo_delta <= 0
       AND unresolved_paygo_delta <= 0 THEN
        UPDATE external_governance_buckets
        SET operation_count = GREATEST(operation_count + operation_delta, 0),
            settled_paygo = GREATEST(
                settled_paygo + settled_paygo_delta, 0
            ),
            unresolved_paygo = GREATEST(
                unresolved_paygo + unresolved_paygo_delta, 0
            ),
            updated_at = NOW()
        WHERE scope_type = requested_scope_type
          AND scope_id = requested_scope_id
          AND bucket_start = requested_bucket;
        RETURN;
    END IF;

    INSERT INTO external_governance_buckets (
        scope_type,
        scope_id,
        bucket_start,
        operation_count,
        settled_paygo,
        unresolved_paygo
    ) VALUES (
        requested_scope_type,
        requested_scope_id,
        requested_bucket,
        GREATEST(operation_delta, 0),
        GREATEST(settled_paygo_delta, 0),
        GREATEST(unresolved_paygo_delta, 0)
    ) ON CONFLICT (scope_type, scope_id, bucket_start) DO UPDATE
    SET operation_count = GREATEST(
            external_governance_buckets.operation_count + operation_delta,
            0
        ),
        settled_paygo = GREATEST(
            external_governance_buckets.settled_paygo + settled_paygo_delta,
            0
        ),
        unresolved_paygo = GREATEST(
            external_governance_buckets.unresolved_paygo + unresolved_paygo_delta,
            0
        ),
        updated_at = NOW();
END;
$$;

-- Keep admission state correct while the one-time backfill takes its snapshot.
LOCK TABLE external_operations IN SHARE ROW EXCLUSIVE MODE;

INSERT INTO external_governance_state (
    scope_type,
    scope_id,
    active_tasks,
    active_sync_requests,
    active_realtime,
    active_streams,
    unresolved_paygo,
    unresolved_charge
)
SELECT
    scope.scope_type,
    scope.scope_id,
    COUNT(*) FILTER (
        WHERE operation_mode = 'task'
          AND status IN ('pending', 'submitted', 'running')
    ),
    COUNT(*) FILTER (
        WHERE operation_mode = 'sync'
          AND status IN ('pending', 'submitted', 'running')
    ),
    COUNT(*) FILTER (
        WHERE operation_mode = 'realtime'
          AND status IN ('pending', 'submitted', 'running')
    ),
    COUNT(*) FILTER (
        WHERE operation_mode = 'stream'
          AND status IN ('pending', 'submitted', 'running')
    ),
    COALESCE(SUM(
        CASE
            WHEN status IN ('pending', 'submitted', 'running')
                 OR settlement_status IN ('pending', 'failed', 'quarantined')
                THEN external_governance_paygo(
                    status, settlement_status, settlement_metadata
                )
            ELSE 0::numeric
        END
    ), 0::numeric),
    COALESCE(SUM(
        CASE
            WHEN (
                status IN ('pending', 'submitted', 'running')
                OR settlement_status IN ('pending', 'failed', 'quarantined')
            ) AND settlement_metadata->'pricing'->>'free_invocation' IS DISTINCT FROM 'true'
              AND settlement_metadata->'pricing'->>'balance_exempt' IS DISTINCT FROM 'true'
                THEN external_governance_paygo(
                    status, settlement_status, settlement_metadata
                )
            ELSE 0::numeric
        END
    ), 0::numeric)
FROM external_operations
CROSS JOIN LATERAL (
    VALUES
        ('user'::varchar, user_id),
        ('account'::varchar, account_id)
) AS scope(scope_type, scope_id)
WHERE scope.scope_id IS NOT NULL
GROUP BY scope.scope_type, scope.scope_id
ON CONFLICT (scope_type, scope_id) DO UPDATE
SET active_tasks = EXCLUDED.active_tasks,
    active_sync_requests = EXCLUDED.active_sync_requests,
    active_realtime = EXCLUDED.active_realtime,
    active_streams = EXCLUDED.active_streams,
    unresolved_paygo = EXCLUDED.unresolved_paygo,
    unresolved_charge = EXCLUDED.unresolved_charge,
    updated_at = NOW();

INSERT INTO external_governance_buckets (
    scope_type,
    scope_id,
    bucket_start,
    operation_count,
    unresolved_paygo,
    settled_paygo
)
SELECT
    scope_type,
    scope_id,
    bucket_start,
    SUM(operation_count),
    SUM(unresolved_paygo),
    SUM(settled_paygo)
FROM (
    SELECT
        scope.scope_type,
        scope.scope_id,
        date_trunc('minute', operation.created_at) AS bucket_start,
        1::bigint AS operation_count,
        CASE
            WHEN operation.status IN ('pending', 'submitted', 'running')
                 OR operation.settlement_status IN (
                    'pending', 'failed', 'quarantined'
                 )
                THEN external_governance_paygo(
                    operation.status,
                    operation.settlement_status,
                    operation.settlement_metadata
                )
            ELSE 0::numeric
        END AS unresolved_paygo,
        0::numeric AS settled_paygo
    FROM external_operations AS operation
    CROSS JOIN LATERAL (
        VALUES
            ('user'::varchar, operation.user_id),
            ('account'::varchar, operation.account_id)
    ) AS scope(scope_type, scope_id)
    WHERE scope.scope_id IS NOT NULL
      AND operation.created_at >= NOW() - INTERVAL '24 hours'

    UNION ALL

    SELECT
        scope.scope_type,
        scope.scope_id,
        date_trunc('minute', operation.settled_at) AS bucket_start,
        0::bigint AS operation_count,
        0::numeric AS unresolved_paygo,
        external_governance_paygo(
            operation.status,
            operation.settlement_status,
            operation.settlement_metadata
        ) AS settled_paygo
    FROM external_operations AS operation
    CROSS JOIN LATERAL (
        VALUES
            ('user'::varchar, operation.user_id),
            ('account'::varchar, operation.account_id)
    ) AS scope(scope_type, scope_id)
    WHERE scope.scope_id IS NOT NULL
      AND operation.settlement_status = 'settled'
      AND operation.settled_at >= NOW() - INTERVAL '24 hours'
) AS contribution
GROUP BY scope_type, scope_id, bucket_start
ON CONFLICT (scope_type, scope_id, bucket_start) DO UPDATE
SET operation_count = EXCLUDED.operation_count,
    unresolved_paygo = EXCLUDED.unresolved_paygo,
    settled_paygo = EXCLUDED.settled_paygo,
    updated_at = NOW();

CREATE FUNCTION external_governance_apply_scope_transition(
    requested_scope_type VARCHAR,
    old_scope_id VARCHAR,
    new_scope_id VARCHAR,
    old_task BIGINT,
    new_task BIGINT,
    old_sync BIGINT,
    new_sync BIGINT,
    old_realtime BIGINT,
    new_realtime BIGINT,
    old_stream BIGINT,
    new_stream BIGINT,
    old_unresolved_paygo NUMERIC,
    new_unresolved_paygo NUMERIC,
    old_unresolved_charge NUMERIC,
    new_unresolved_charge NUMERIC,
    old_created_at TIMESTAMPTZ,
    new_created_at TIMESTAMPTZ,
    old_settled_at TIMESTAMPTZ,
    new_settled_at TIMESTAMPTZ,
    old_settled_paygo NUMERIC,
    new_settled_paygo NUMERIC
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    -- Coalesce an in-place transition into one net state-row update. Scope
    -- changes necessarily touch the old and new rows separately.
    IF old_scope_id IS NOT NULL AND old_scope_id = new_scope_id THEN
        PERFORM external_governance_apply_state(
            requested_scope_type,
            old_scope_id,
            new_task - old_task,
            new_sync - old_sync,
            new_realtime - old_realtime,
            new_stream - old_stream,
            new_unresolved_paygo - old_unresolved_paygo,
            new_unresolved_charge - old_unresolved_charge
        );
    ELSE
        PERFORM external_governance_apply_state(
            requested_scope_type,
            old_scope_id,
            -old_task,
            -old_sync,
            -old_realtime,
            -old_stream,
            -old_unresolved_paygo,
            -old_unresolved_charge
        );
        PERFORM external_governance_apply_state(
            requested_scope_type,
            new_scope_id,
            new_task,
            new_sync,
            new_realtime,
            new_stream,
            new_unresolved_paygo,
            new_unresolved_charge
        );
    END IF;

    -- Creation moves within the same minute have no net bucket effect.
    IF old_scope_id IS NOT NULL
       AND old_scope_id = new_scope_id
       AND date_trunc('minute', old_created_at)
            IS NOT DISTINCT FROM date_trunc('minute', new_created_at) THEN
        PERFORM external_governance_apply_bucket(
            requested_scope_type,
            old_scope_id,
            old_created_at,
            0,
            0,
            new_unresolved_paygo - old_unresolved_paygo
        );
    ELSE
        PERFORM external_governance_apply_bucket(
            requested_scope_type,
            old_scope_id,
            old_created_at,
            -1,
            0,
            -old_unresolved_paygo
        );
        PERFORM external_governance_apply_bucket(
            requested_scope_type,
            new_scope_id,
            new_created_at,
            1,
            0,
            new_unresolved_paygo
        );
    END IF;

    -- Likewise, a settled charge correction in one minute is one net update.
    IF old_scope_id IS NOT NULL
       AND old_scope_id = new_scope_id
       AND old_settled_at IS NOT NULL
       AND new_settled_at IS NOT NULL
       AND date_trunc('minute', old_settled_at)
            = date_trunc('minute', new_settled_at) THEN
        PERFORM external_governance_apply_bucket(
            requested_scope_type,
            old_scope_id,
            old_settled_at,
            0,
            new_settled_paygo - old_settled_paygo,
            0
        );
    ELSE
        PERFORM external_governance_apply_bucket(
            requested_scope_type,
            old_scope_id,
            old_settled_at,
            0,
            -old_settled_paygo,
            0
        );
        PERFORM external_governance_apply_bucket(
            requested_scope_type,
            new_scope_id,
            new_settled_at,
            0,
            new_settled_paygo,
            0
        );
    END IF;

    RETURN;
END;
$$;

CREATE FUNCTION external_governance_track_operation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    old_user_id VARCHAR;
    new_user_id VARCHAR;
    old_account_id VARCHAR;
    new_account_id VARCHAR;
    old_created_at TIMESTAMPTZ;
    new_created_at TIMESTAMPTZ;
    old_settled_at TIMESTAMPTZ;
    new_settled_at TIMESTAMPTZ;
    old_task BIGINT := 0;
    new_task BIGINT := 0;
    old_sync BIGINT := 0;
    new_sync BIGINT := 0;
    old_realtime BIGINT := 0;
    new_realtime BIGINT := 0;
    old_stream BIGINT := 0;
    new_stream BIGINT := 0;
    old_unresolved_paygo NUMERIC := 0;
    new_unresolved_paygo NUMERIC := 0;
    old_unresolved_charge NUMERIC := 0;
    new_unresolved_charge NUMERIC := 0;
    old_settled_paygo NUMERIC := 0;
    new_settled_paygo NUMERIC := 0;
    old_paygo NUMERIC := 0;
    new_paygo NUMERIC := 0;
BEGIN
    -- Lease acquisition/renewal, heartbeat, retry scheduling, and ordinary
    -- presentation writes do not affect governance. Avoid touching either hot
    -- scope row when every contribution-driving column is unchanged.
    IF TG_OP = 'UPDATE'
       AND OLD.user_id IS NOT DISTINCT FROM NEW.user_id
       AND OLD.account_id IS NOT DISTINCT FROM NEW.account_id
       AND OLD.operation_mode IS NOT DISTINCT FROM NEW.operation_mode
       AND OLD.status IS NOT DISTINCT FROM NEW.status
       AND OLD.settlement_status IS NOT DISTINCT FROM NEW.settlement_status
       AND OLD.settlement_metadata IS NOT DISTINCT FROM NEW.settlement_metadata
       AND OLD.created_at IS NOT DISTINCT FROM NEW.created_at
       AND OLD.settled_at IS NOT DISTINCT FROM NEW.settled_at THEN
        RETURN NULL;
    END IF;

    IF TG_OP <> 'INSERT' THEN
        old_user_id := OLD.user_id;
        old_account_id := OLD.account_id;
        old_created_at := OLD.created_at;
        old_paygo := external_governance_paygo(
            OLD.status, OLD.settlement_status, OLD.settlement_metadata
        );
        IF OLD.status IN ('pending', 'submitted', 'running') THEN
            old_task := CASE WHEN OLD.operation_mode = 'task' THEN 1 ELSE 0 END;
            old_sync := CASE WHEN OLD.operation_mode = 'sync' THEN 1 ELSE 0 END;
            old_realtime := CASE WHEN OLD.operation_mode = 'realtime' THEN 1 ELSE 0 END;
            old_stream := CASE WHEN OLD.operation_mode = 'stream' THEN 1 ELSE 0 END;
        END IF;
        IF OLD.status IN ('pending', 'submitted', 'running')
           OR OLD.settlement_status IN ('pending', 'failed', 'quarantined') THEN
            old_unresolved_paygo := old_paygo;
            IF OLD.settlement_metadata->'pricing'->>'free_invocation'
                    IS DISTINCT FROM 'true'
               AND OLD.settlement_metadata->'pricing'->>'balance_exempt'
                    IS DISTINCT FROM 'true' THEN
                old_unresolved_charge := old_paygo;
            END IF;
        END IF;
        IF OLD.settlement_status = 'settled' THEN
            old_settled_at := OLD.settled_at;
            old_settled_paygo := old_paygo;
        END IF;
    END IF;

    IF TG_OP <> 'DELETE' THEN
        new_user_id := NEW.user_id;
        new_account_id := NEW.account_id;
        new_created_at := NEW.created_at;
        new_paygo := external_governance_paygo(
            NEW.status, NEW.settlement_status, NEW.settlement_metadata
        );
        IF NEW.status IN ('pending', 'submitted', 'running') THEN
            new_task := CASE WHEN NEW.operation_mode = 'task' THEN 1 ELSE 0 END;
            new_sync := CASE WHEN NEW.operation_mode = 'sync' THEN 1 ELSE 0 END;
            new_realtime := CASE WHEN NEW.operation_mode = 'realtime' THEN 1 ELSE 0 END;
            new_stream := CASE WHEN NEW.operation_mode = 'stream' THEN 1 ELSE 0 END;
        END IF;
        IF NEW.status IN ('pending', 'submitted', 'running')
           OR NEW.settlement_status IN ('pending', 'failed', 'quarantined') THEN
            new_unresolved_paygo := new_paygo;
            IF NEW.settlement_metadata->'pricing'->>'free_invocation'
                    IS DISTINCT FROM 'true'
               AND NEW.settlement_metadata->'pricing'->>'balance_exempt'
                    IS DISTINCT FROM 'true' THEN
                new_unresolved_charge := new_paygo;
            END IF;
        END IF;
        IF NEW.settlement_status = 'settled' THEN
            new_settled_at := NEW.settled_at;
            new_settled_paygo := new_paygo;
        END IF;
    END IF;

    -- Single-row writers acquire user before account. Multi-row writers pre-lock
    -- every distinct user in sorted order and then the account before flushing.
    PERFORM external_governance_apply_scope_transition(
        'user', old_user_id, new_user_id,
        old_task, new_task, old_sync, new_sync,
        old_realtime, new_realtime, old_stream, new_stream,
        old_unresolved_paygo, new_unresolved_paygo,
        old_unresolved_charge, new_unresolved_charge,
        old_created_at, new_created_at,
        old_settled_at, new_settled_at,
        old_settled_paygo, new_settled_paygo
    );
    PERFORM external_governance_apply_scope_transition(
        'account', old_account_id, new_account_id,
        old_task, new_task, old_sync, new_sync,
        old_realtime, new_realtime, old_stream, new_stream,
        old_unresolved_paygo, new_unresolved_paygo,
        old_unresolved_charge, new_unresolved_charge,
        old_created_at, new_created_at,
        old_settled_at, new_settled_at,
        old_settled_paygo, new_settled_paygo
    );

    RETURN NULL;
END;
$$;

CREATE TRIGGER trg_external_governance_operation
AFTER INSERT OR DELETE ON external_operations
FOR EACH ROW EXECUTE FUNCTION external_governance_track_operation();

CREATE TRIGGER trg_external_governance_operation_update
AFTER UPDATE OF
    user_id,
    account_id,
    operation_mode,
    status,
    settlement_status,
    settlement_metadata,
    created_at,
    settled_at
ON external_operations
FOR EACH ROW EXECUTE FUNCTION external_governance_track_operation();

-- migrate:down

DROP TRIGGER IF EXISTS trg_external_governance_operation ON external_operations;
DROP TRIGGER IF EXISTS trg_external_governance_operation_update ON external_operations;
DROP FUNCTION IF EXISTS external_governance_track_operation();
DROP FUNCTION IF EXISTS external_governance_apply_scope_transition(
    VARCHAR, VARCHAR, VARCHAR,
    BIGINT, BIGINT, BIGINT, BIGINT, BIGINT, BIGINT, BIGINT, BIGINT,
    NUMERIC, NUMERIC, NUMERIC, NUMERIC,
    TIMESTAMPTZ, TIMESTAMPTZ, TIMESTAMPTZ, TIMESTAMPTZ,
    NUMERIC, NUMERIC
);
DROP FUNCTION IF EXISTS external_governance_apply_bucket(
    VARCHAR, VARCHAR, TIMESTAMPTZ, BIGINT, NUMERIC, NUMERIC
);
DROP FUNCTION IF EXISTS external_governance_apply_state(VARCHAR, VARCHAR, BIGINT, BIGINT, BIGINT, BIGINT, NUMERIC, NUMERIC);
DROP FUNCTION IF EXISTS external_governance_paygo(VARCHAR, VARCHAR, JSONB);
DROP FUNCTION IF EXISTS external_governance_money(TEXT);
DROP TABLE IF EXISTS external_governance_buckets;
DROP TABLE IF EXISTS external_governance_state;
