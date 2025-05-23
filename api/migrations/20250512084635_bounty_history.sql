-- migrate:up
CREATE TABLE IF NOT EXISTS bounty_history (
    bounty_id TEXT NOT NULL PRIMARY KEY,
    chute_id TEXT NOT NULL,
    version TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    claimed_by TEXT
);

CREATE OR REPLACE FUNCTION increase_bounty(p_target_chute_id TEXT)
RETURNS TABLE(bounty INTEGER, last_increased_at TIMESTAMP, was_increased BOOLEAN) AS $$
DECLARE
    v_version TEXT;
    v_current_date_text TEXT;
    v_bounty_id_today TEXT;
    v_bounty_record_exists_in_history BOOLEAN;
    v_initial_bounty INTEGER;
    v_initial_last_increased_at TIMESTAMP;
    v_final_bounty INTEGER;
    v_final_last_increased_at TIMESTAMP;
    v_was_bounty_increased_this_call BOOLEAN := FALSE;
BEGIN
    SELECT c.version INTO v_version
    FROM chutes c
    WHERE c.chute_id = p_target_chute_id;
    IF NOT FOUND THEN
        RETURN QUERY SELECT NULL::INTEGER, NULL::TIMESTAMP, FALSE;
        RETURN;
    END IF;

    v_current_date_text := TO_CHAR(CURRENT_DATE, 'YYYY-MM-DD');
    v_bounty_id_today := encode(digest(p_target_chute_id || v_version || v_current_date_text, 'sha256'), 'hex');

    SELECT EXISTS (
        SELECT 1
        FROM bounty_history bh
        WHERE bh.bounty_id = v_bounty_id_today
    ) INTO v_bounty_record_exists_in_history;

    SELECT b.bounty, b.last_increased_at
    INTO v_initial_bounty, v_initial_last_increased_at
    FROM bounties b
    WHERE b.chute_id = p_target_chute_id;

    IF NOT v_bounty_record_exists_in_history THEN
        INSERT INTO bounties (chute_id, bounty, previous_bounty, last_increased_at)
        VALUES (p_target_chute_id, 100, 100, CURRENT_TIMESTAMP)
        ON CONFLICT (chute_id) DO UPDATE
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

        SELECT b.bounty, b.last_increased_at
        INTO v_final_bounty, v_final_last_increased_at
        FROM bounties b
        WHERE b.chute_id = p_target_chute_id;

        IF (v_initial_bounty IS NULL AND v_final_bounty IS NOT NULL) OR
           (v_initial_bounty IS NOT NULL AND v_final_bounty IS NOT NULL AND v_final_bounty <> v_initial_bounty)
        THEN
            v_was_bounty_increased_this_call := TRUE;
            INSERT INTO bounty_history (bounty_id, chute_id, version, created_at)
            VALUES (v_bounty_id_today, p_target_chute_id, v_version, CURRENT_TIMESTAMP)
            ON CONFLICT (bounty_id) DO NOTHING;
        ELSE
            v_was_bounty_increased_this_call := FALSE;
        END IF;

    ELSE
        v_final_bounty := v_initial_bounty;
        v_final_last_increased_at := v_initial_last_increased_at;
        v_was_bounty_increased_this_call := FALSE;
    END IF;

    RETURN QUERY SELECT v_final_bounty, v_final_last_increased_at, v_was_bounty_increased_this_call;
END;
$$ LANGUAGE plpgsql;

-- migrate:down
drop table if exists bounty_history;
