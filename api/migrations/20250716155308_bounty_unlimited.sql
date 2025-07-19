-- migrate:up
CREATE OR REPLACE FUNCTION increase_bounty(p_target_chute_id TEXT)
RETURNS TABLE(bounty INTEGER, last_increased_at TIMESTAMP, was_increased BOOLEAN) AS $$
DECLARE
    v_version TEXT;
    v_current_date_text TEXT;
    v_bounty_id_today TEXT;
    v_bounty_record_exists_in_history BOOLEAN;
    v_bounties_record_exists BOOLEAN;
    v_initial_bounty INTEGER;
    v_initial_last_increased_at TIMESTAMP;
    v_final_bounty INTEGER;
    v_final_last_increased_at TIMESTAMP;
    v_was_bounty_increased_this_call BOOLEAN := FALSE;
    v_should_check_history BOOLEAN := TRUE;
BEGIN
    RAISE NOTICE 'Starting increase_bounty for chute_id: %', p_target_chute_id;

    -- Debug: Check if chute exists
    SELECT c.chutes_version INTO v_version
    FROM chutes c
    WHERE c.chute_id = p_target_chute_id;

    IF NOT FOUND THEN
        RAISE NOTICE 'Chute not found: %', p_target_chute_id;
        RETURN QUERY SELECT NULL::INTEGER, NULL::TIMESTAMP, FALSE;
        RETURN;
    END IF;

    RAISE NOTICE 'Version found: %', v_version;

    -- Check if version is >= 0.3.0 using semver comparison
    -- This handles versions like '0.3.0', '0.3.0.rc1', '0.3.1', etc.
    IF v_version ~ '^([1-9]\d*|0\.([3-9]\d*|[1-9]\d+))\.' OR
       v_version = '0.3.0' OR
       v_version ~ '^0\.3\.[0-9]+' THEN
        v_should_check_history := FALSE;
        RAISE NOTICE 'Version % is >= 0.3.0, setting v_should_check_history to FALSE', v_version;
    ELSE
        RAISE NOTICE 'Version % is < 0.3.0, setting v_should_check_history to TRUE', v_version;
    END IF;

    -- Currently one bounty per chute per day, just temporary...
    v_current_date_text := TO_CHAR(CURRENT_DATE, 'YYYY-MM-DD');
    v_bounty_id_today := encode(digest(p_target_chute_id || v_version || v_current_date_text, 'sha256'), 'hex');
    RAISE NOTICE 'Bounty ID for today: %', v_bounty_id_today;

    -- Only check bounty history if version < 0.3.0
    IF v_should_check_history THEN
        RAISE NOTICE 'Checking bounty history...';
        -- Bounty already created today?
        SELECT EXISTS (
            SELECT 1
            FROM bounty_history bh
            WHERE bh.bounty_id = v_bounty_id_today
        ) INTO v_bounty_record_exists_in_history;
        RAISE NOTICE 'Bounty exists in history: %', v_bounty_record_exists_in_history;
    ELSE
        -- For versions >= 0.3.0, always treat as if no history exists
        v_bounty_record_exists_in_history := FALSE;
        RAISE NOTICE 'Version >= 0.3.0, skipping history check. Setting v_bounty_record_exists_in_history to FALSE';
    END IF;

    -- Current bounty already exists?
    SELECT EXISTS (
        SELECT 1
        FROM bounties b
        WHERE b.chute_id = p_target_chute_id
    ) INTO v_bounties_record_exists;

    RAISE NOTICE 'Bounty exists in bounties table: %', v_bounties_record_exists;

    IF v_bounties_record_exists THEN
        RAISE NOTICE 'Entering UPDATE path - bounty already exists';
        -- When a bounty already exists, we can increase it's value.
        SELECT b.bounty, b.last_increased_at
        INTO v_initial_bounty, v_initial_last_increased_at
        FROM bounties b
        WHERE b.chute_id = p_target_chute_id;

        RAISE NOTICE 'Current bounty: %, last increased: %', v_initial_bounty, v_initial_last_increased_at;
        RAISE NOTICE 'Time since last increase: %', CURRENT_TIMESTAMP - v_initial_last_increased_at;
        RAISE NOTICE 'Proposed new bounty would be: %', v_initial_bounty + LEAST(300, v_initial_bounty);

        UPDATE bounties
        SET
            previous_bounty = bounties.bounty,
            bounty = CASE
                WHEN CURRENT_TIMESTAMP - bounties.last_increased_at >= interval '30 seconds'
                     AND bounties.bounty + LEAST(300, bounties.previous_bounty) <= 86400
                THEN bounties.bounty + LEAST(300, bounties.previous_bounty)
                ELSE bounties.bounty
            END,
            last_increased_at = CASE
                WHEN CURRENT_TIMESTAMP - bounties.last_increased_at >= interval '30 seconds'
                     AND bounties.bounty + LEAST(300, bounties.previous_bounty) <= 86400
                THEN CURRENT_TIMESTAMP
                ELSE bounties.last_increased_at
            END
        WHERE bounties.chute_id = p_target_chute_id;

        -- Get final values after update
        SELECT b.bounty, b.last_increased_at
        INTO v_final_bounty, v_final_last_increased_at
        FROM bounties b
        WHERE b.chute_id = p_target_chute_id;

        RAISE NOTICE 'After update - bounty: %, last increased: %', v_final_bounty, v_final_last_increased_at;

        IF v_final_bounty <> v_initial_bounty THEN
            v_was_bounty_increased_this_call := TRUE;
            RAISE NOTICE 'Bounty was increased from % to %', v_initial_bounty, v_final_bounty;
            -- Track the history only for versions < 0.3.0
            IF v_should_check_history AND NOT v_bounty_record_exists_in_history THEN
                RAISE NOTICE 'Adding to bounty history';
                INSERT INTO bounty_history (bounty_id, chute_id, version, created_at)
                VALUES (v_bounty_id_today, p_target_chute_id, v_version, CURRENT_TIMESTAMP)
                ON CONFLICT (bounty_id) DO NOTHING;
            END IF;
        ELSE
            RAISE NOTICE 'Bounty was NOT increased (still %)', v_final_bounty;
        END IF;

    ELSE
        RAISE NOTICE 'Entering CREATE path - no bounty record exists';
        RAISE NOTICE 'v_should_check_history: %, v_bounty_record_exists_in_history: %', v_should_check_history, v_bounty_record_exists_in_history;

        -- No bounties record exists, create a new one (if none created previously today for versions < 0.3.0).
        IF NOT v_should_check_history OR NOT v_bounty_record_exists_in_history THEN
            RAISE NOTICE 'Conditions met for creating new bounty record';
            RAISE NOTICE 'Attempting INSERT into bounties table...';

            BEGIN
                INSERT INTO bounties (chute_id, bounty, previous_bounty, last_increased_at)
                VALUES (p_target_chute_id, 100, 100, CURRENT_TIMESTAMP);
                RAISE NOTICE 'INSERT successful';
            EXCEPTION WHEN OTHERS THEN
                RAISE NOTICE 'INSERT failed with error: %', SQLERRM;
                RAISE;
            END;

            v_final_bounty := 100;
            v_final_last_increased_at := CURRENT_TIMESTAMP;
            v_was_bounty_increased_this_call := TRUE;

            -- Track for dedupe only for versions < 0.3.0
            IF v_should_check_history THEN
                RAISE NOTICE 'Adding to bounty history (version < 0.3.0)';
                INSERT INTO bounty_history (bounty_id, chute_id, version, created_at)
                VALUES (v_bounty_id_today, p_target_chute_id, v_version, CURRENT_TIMESTAMP)
                ON CONFLICT (bounty_id) DO NOTHING;
            ELSE
                RAISE NOTICE 'NOT adding to bounty history (version >= 0.3.0)';
            END IF;
        ELSE
            -- History record exists but no bounties record, skip (only for versions < 0.3.0).
            RAISE NOTICE 'History record exists for version < 0.3.0, skipping creation';
            v_final_bounty := NULL;
            v_final_last_increased_at := NULL;
            v_was_bounty_increased_this_call := FALSE;
        END IF;
    END IF;

    RAISE NOTICE 'Final values - bounty: %, last_increased: %, was_increased: %', v_final_bounty, v_final_last_increased_at, v_was_bounty_increased_this_call;
    RETURN QUERY SELECT v_final_bounty, v_final_last_increased_at, v_was_bounty_increased_this_call;
END;
$$ LANGUAGE plpgsql;

-- migrate:down
CREATE OR REPLACE FUNCTION increase_bounty(p_target_chute_id TEXT)
RETURNS TABLE(bounty INTEGER, last_increased_at TIMESTAMP, was_increased BOOLEAN) AS $$
DECLARE
    v_version TEXT;
    v_current_date_text TEXT;
    v_bounty_id_today TEXT;
    v_bounty_record_exists_in_history BOOLEAN;
    v_bounties_record_exists BOOLEAN;
    v_initial_bounty INTEGER;
    v_initial_last_increased_at TIMESTAMP;
    v_final_bounty INTEGER;
    v_final_last_increased_at TIMESTAMP;
    v_was_bounty_increased_this_call BOOLEAN := FALSE;
BEGIN
    SELECT c.chutes_version INTO v_version
    FROM chutes c
    WHERE c.chute_id = p_target_chute_id;

    IF NOT FOUND THEN
        RETURN QUERY SELECT NULL::INTEGER, NULL::TIMESTAMP, FALSE;
        RETURN;
    END IF;

    -- Currently one bounty per chute per day, just temporary...
    v_current_date_text := TO_CHAR(CURRENT_DATE, 'YYYY-MM-DD');
    v_bounty_id_today := encode(digest(p_target_chute_id || v_version || v_current_date_text, 'sha256'), 'hex');

    -- Bounty already created today?
    SELECT EXISTS (
        SELECT 1
        FROM bounty_history bh
        WHERE bh.bounty_id = v_bounty_id_today
    ) INTO v_bounty_record_exists_in_history;

    -- Current bounty already exists?
    SELECT EXISTS (
        SELECT 1
        FROM bounties b
        WHERE b.chute_id = p_target_chute_id
    ) INTO v_bounties_record_exists;

    IF v_bounties_record_exists THEN
        -- When a bounty already exists, we can increase it's value.
        SELECT b.bounty, b.last_increased_at
        INTO v_initial_bounty, v_initial_last_increased_at
        FROM bounties b
        WHERE b.chute_id = p_target_chute_id;
        UPDATE bounties
        SET
            previous_bounty = bounties.bounty,
            bounty = CASE
                WHEN CURRENT_TIMESTAMP - bounties.last_increased_at >= interval '20 seconds'
                     AND bounties.bounty + bounties.previous_bounty <= 5000
                THEN bounties.bounty + bounties.previous_bounty
                ELSE bounties.bounty
            END,
            last_increased_at = CASE
                WHEN CURRENT_TIMESTAMP - bounties.last_increased_at >= interval '20 seconds'
                     AND bounties.bounty + bounties.previous_bounty <= 5000
                THEN CURRENT_TIMESTAMP
                ELSE bounties.last_increased_at
            END
        WHERE bounties.chute_id = p_target_chute_id;

        -- Get final values after update
        SELECT b.bounty, b.last_increased_at
        INTO v_final_bounty, v_final_last_increased_at
        FROM bounties b
        WHERE b.chute_id = p_target_chute_id;

        IF v_final_bounty <> v_initial_bounty THEN
            v_was_bounty_increased_this_call := TRUE;
            -- Track the history so we don't produce multiple bounties in one day.
            IF NOT v_bounty_record_exists_in_history THEN
                INSERT INTO bounty_history (bounty_id, chute_id, version, created_at)
                VALUES (v_bounty_id_today, p_target_chute_id, v_version, CURRENT_TIMESTAMP)
                ON CONFLICT (bounty_id) DO NOTHING;
            END IF;
        END IF;

    ELSE
        -- No bounties record exists, create a new one (if none created previously today).
        IF NOT v_bounty_record_exists_in_history THEN
            INSERT INTO bounties (chute_id, bounty, previous_bounty, last_increased_at)
            VALUES (p_target_chute_id, 100, 100, CURRENT_TIMESTAMP);

            v_final_bounty := 100;
            v_final_last_increased_at := CURRENT_TIMESTAMP;
            v_was_bounty_increased_this_call := TRUE;

            -- Track for dedupe.
            INSERT INTO bounty_history (bounty_id, chute_id, version, created_at)
            VALUES (v_bounty_id_today, p_target_chute_id, v_version, CURRENT_TIMESTAMP)
            ON CONFLICT (bounty_id) DO NOTHING;
        ELSE
            -- History record exists but no bounties record, skip.
            v_final_bounty := NULL;
            v_final_last_increased_at := NULL;
            v_was_bounty_increased_this_call := FALSE;
        END IF;
    END IF;

    RETURN QUERY SELECT v_final_bounty, v_final_last_increased_at, v_was_bounty_increased_this_call;
END;
$$ LANGUAGE plpgsql;
