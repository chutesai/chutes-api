-- migrate:up

CREATE OR REPLACE FUNCTION refresh_invocation_quota(p_user_id VARCHAR, p_amount DOUBLE PRECISION)
RETURNS VOID AS $$
DECLARE
    v_permissions_bitmask INTEGER;
    v_excluded_bits INTEGER := 816;
    v_quota_exists BOOLEAN;
    v_quota_is_zero BOOLEAN;
BEGIN
    IF p_amount <= 1.0 THEN
        RETURN;
    END IF;

    SELECT permissions_bitmask INTO v_permissions_bitmask
    FROM users
    WHERE user_id = p_user_id;

    IF v_permissions_bitmask IS NOT NULL AND (v_permissions_bitmask & v_excluded_bits) != 0 THEN
        RETURN;
    END IF;

    SELECT
        EXISTS(SELECT 1 FROM invocation_quotas WHERE user_id = p_user_id AND is_default = true),
        EXISTS(SELECT 1 FROM invocation_quotas WHERE user_id = p_user_id AND is_default = true AND quota = 0)
    INTO v_quota_exists, v_quota_is_zero;

    IF NOT v_quota_exists THEN
        INSERT INTO invocation_quotas (user_id, chute_id, quota, is_default, payment_refresh_date, updated_at)
        VALUES (p_user_id, 'default', 100, true, NOW(), NOW())
        ON CONFLICT (user_id, chute_id) DO NOTHING;
    ELSIF v_quota_is_zero THEN
        UPDATE invocation_quotas
        SET quota = 100,
            payment_refresh_date = NOW(),
            updated_at = NOW()
        WHERE user_id = p_user_id
          AND is_default = true
          AND quota = 0;
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION trigger_refresh_quota_on_payment()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM refresh_invocation_quota(NEW.user_id, NEW.usd_amount);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION trigger_refresh_quota_on_admin_balance()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.reason LIKE 'web-top%' THEN
        PERFORM refresh_invocation_quota(NEW.user_id, NEW.amount);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER refresh_quota_on_payment
    AFTER INSERT ON payments
    FOR EACH ROW
    EXECUTE FUNCTION trigger_refresh_quota_on_payment();

CREATE TRIGGER refresh_quota_on_admin_balance
    AFTER INSERT ON admin_balance_changes
    FOR EACH ROW
    EXECUTE FUNCTION trigger_refresh_quota_on_admin_balance();

-- migrate:down
DROP TRIGGER IF EXISTS refresh_quota_on_payment ON payments;
DROP TRIGGER IF EXISTS refresh_quota_on_admin_balance ON admin_balance_changes;
DROP FUNCTION IF EXISTS trigger_refresh_quota_on_payment();
DROP FUNCTION IF EXISTS trigger_refresh_quota_on_admin_balance();
DROP FUNCTION IF EXISTS refresh_invocation_quota(VARCHAR, DOUBLE PRECISION);
