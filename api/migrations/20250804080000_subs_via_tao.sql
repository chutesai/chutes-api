-- migrate:up
DROP TRIGGER IF EXISTS refresh_quota_on_payment ON payments;
DROP TRIGGER IF EXISTS refresh_quota_on_admin_balance ON admin_balance_changes;
DROP FUNCTION IF EXISTS trigger_refresh_quota_on_payment();
DROP FUNCTION IF EXISTS trigger_refresh_quota_on_admin_balance();
DROP FUNCTION IF EXISTS refresh_invocation_quota(VARCHAR);
DROP FUNCTION IF EXISTS refresh_invocation_quota(VARCHAR, DOUBLE PRECISION);

CREATE OR REPLACE FUNCTION refresh_invocation_quota(p_user_id VARCHAR, p_payment_amount DOUBLE PRECISION)
RETURNS VOID AS $$
DECLARE
    v_permissions_bitmask INTEGER;
    v_excluded_bits INTEGER := 816;
    v_quota_exists BOOLEAN;
    v_current_quota INTEGER;
    v_payment_refresh_date TIMESTAMP;
    v_new_quota INTEGER;
    v_tier_price DOUBLE PRECISION;
    v_user_balance DOUBLE PRECISION;
BEGIN
    -- Check user permissions
    SELECT permissions_bitmask, balance INTO v_permissions_bitmask
    FROM users
    WHERE user_id = p_user_id;
    IF v_permissions_bitmask IS NOT NULL AND (v_permissions_bitmask & v_excluded_bits) != 0 THEN
        RETURN;
    END IF;

    -- Get current quota info
    SELECT
        EXISTS(SELECT 1 FROM invocation_quotas WHERE user_id = p_user_id AND is_default = true),
        quota,
        payment_refresh_date
    INTO v_quota_exists, v_current_quota, v_payment_refresh_date
    FROM invocation_quotas
    WHERE user_id = p_user_id AND is_default = true;

    -- Determine tier based on payment amount and user balance
    IF p_payment_amount >= 19.5 THEN
        v_new_quota := 5000;
        v_tier_price := 20;
    ELSIF p_payment_amount >= 9.75 THEN
        v_new_quota := 2000;
        v_tier_price := 10;
    ELSIF p_payment_amount >= 2.75 THEN
        v_new_quota := 300;
        v_tier_price := 3;
    ELSE
        RETURN;
    END IF;

    -- Check if we should update the quota.
    IF NOT v_quota_exists OR
       v_payment_refresh_date IS NULL OR
       v_payment_refresh_date < NOW() - INTERVAL '30 days' OR
       (v_payment_refresh_date >= NOW() - INTERVAL '30 days' AND v_new_quota > COALESCE(v_current_quota, 0)) THEN

        INSERT INTO invocation_quotas (user_id, chute_id, quota, is_default, payment_refresh_date, updated_at)
        VALUES (p_user_id, '*', v_new_quota, true, NOW(), NOW())
        ON CONFLICT (user_id, chute_id)
        DO UPDATE SET
            quota = v_new_quota,
            payment_refresh_date = NOW(),
            updated_at = NOW();
        UPDATE users
        SET balance = balance - v_tier_price
        WHERE user_id = p_user_id;
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION trigger_refresh_quota_on_payment()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM refresh_invocation_quota(NEW.user_id, NEW.amount);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER refresh_quota_on_payment
    AFTER INSERT ON payments
    FOR EACH ROW
    EXECUTE FUNCTION trigger_refresh_quota_on_payment();

-- migrate:down
DROP TRIGGER IF EXISTS refresh_quota_on_payment ON payments;
DROP FUNCTION IF EXISTS trigger_refresh_quota_on_payment();
DROP FUNCTION IF EXISTS refresh_invocation_quota(VARCHAR, DOUBLE PRECISION);
