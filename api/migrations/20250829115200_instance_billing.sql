-- migrate:up
CREATE OR REPLACE FUNCTION update_balance_on_instance_delete()
RETURNS TRIGGER AS $$
DECLARE
    v_user_id TEXT;
    v_is_public BOOLEAN;
    v_user_permissions INTEGER;
    v_has_job BOOLEAN;
    v_total_cost DECIMAL(10,2);
    v_hours_used DECIMAL(10,6);
    v_hour_bucket TIMESTAMP;
BEGIN
    -- Get the user_id, public status, and check if instance has a job.
    SELECT c.user_id, c.public, u.permissions_bitmask,
           EXISTS(SELECT 1 FROM jobs j WHERE j.instance_id = OLD.instance_id)
    INTO v_user_id, v_is_public, v_user_permissions, v_has_job
    FROM chutes c
    JOIN users u ON u.user_id = c.user_id
    WHERE c.chute_id = OLD.chute_id;

    -- Update the audit table to adjust the billing stop time.
    UPDATE instance_audit
       SET stop_billing_at = NOW()
     WHERE instance_id = OLD.instance_id
       AND stop_billing_at > NOW();

    -- Skip billing for free users.
    IF (v_user_permissions & 16) = 16 THEN
        RETURN OLD;
    END IF;

    -- Skip billing for non-job instances on public chutes.
    IF NOT v_has_job AND v_is_public = true THEN
        RETURN OLD;
    END IF;

    -- Update the actual user balance table.
    IF OLD.activated_at IS NOT NULL THEN
        v_hours_used := EXTRACT(EPOCH FROM (
            LEAST(COALESCE(OLD.stop_billing_at, NOW()), NOW()) - OLD.activated_at
        )) / 3600.0;
        v_total_cost := v_hours_used * OLD.hourly_rate;
        v_hour_bucket := date_trunc('hour', NOW());

        UPDATE users
        SET balance = balance - v_total_cost
        WHERE user_id = v_user_id;

        -- Track the amount in the usage_data table.
        INSERT INTO usage_data (user_id, bucket, chute_id, amount, count)
        VALUES (v_user_id::varchar, v_hour_bucket, OLD.chute_id, v_total_cost, 1)
        ON CONFLICT (user_id, bucket, chute_id)
        DO UPDATE SET amount = usage_data.amount + EXCLUDED.amount;
    END IF;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

-- Instance deletion trigger to update balances.
CREATE TRIGGER trigger_update_balance_on_delete
BEFORE DELETE ON instances
FOR EACH ROW
EXECUTE FUNCTION update_balance_on_instance_delete();

-- User balance view, which accounts for both the actual balance and the instances (not yet truly billed).
CREATE MATERIALIZED VIEW IF NOT EXISTS user_current_balance AS
SELECT
    u.user_id,
    u.balance as stored_balance,
    COALESCE(SUM(
        CASE
            WHEN i.activated_at IS NOT NULL
                AND i.stop_billing_at > NOW()
                AND (u.permissions_bitmask & 16) != 16
                AND (j.job_id IS NOT NULL OR c.public = false) THEN
                EXTRACT(EPOCH FROM (
                    LEAST(COALESCE(i.stop_billing_at, NOW()), NOW()) - i.activated_at
                )) / 3600.0 * i.hourly_rate
            ELSE 0
        END
    ), 0) as total_instance_costs,
    u.balance - COALESCE(SUM(
        CASE
            WHEN i.activated_at IS NOT NULL
                AND i.stop_billing_at > NOW()
                AND (u.permissions_bitmask & 16) != 16
                AND (j.job_id IS NOT NULL OR c.public = false) THEN
                EXTRACT(EPOCH FROM (
                    LEAST(COALESCE(i.stop_billing_at, NOW()), NOW()) - i.activated_at
                )) / 3600.0 * i.hourly_rate
            ELSE 0
        END
    ), 0) as effective_balance
FROM users u
LEFT JOIN chutes c ON c.user_id = u.user_id
LEFT JOIN instances i ON i.chute_id = c.chute_id
LEFT JOIN jobs j ON j.instance_id = i.instance_id
GROUP BY u.user_id, u.balance, u.permissions_bitmask;

CREATE UNIQUE INDEX IF NOT EXISTS idx_user_current_balance_user_id ON user_current_balance(user_id);

-- migrate:down
DROP TRIGGER IF EXISTS trigger_update_balance_on_delete ON instances;
DROP FUNCTION IF EXISTS update_balance_on_instance_delete;
