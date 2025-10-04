-- migrate:up

DROP MATERIALIZED VIEW IF EXISTS daily_instance_revenue;
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_instance_revenue AS
WITH date_series AS (
    SELECT generate_series(
        (SELECT MIN(DATE(activated_at)) FROM instance_audit WHERE billed_to IS NOT NULL),
        CURRENT_DATE,
        '1 day'::interval
    )::date AS date
),
instance_daily_costs AS (
    SELECT
        d.date,
        i.instance_id,
        i.billed_to,
        i.hourly_rate,
        -- Calculate hours for this specific day
        CASE
            -- Instance runs through the entire day
            WHEN DATE(i.activated_at) < d.date 
                AND DATE(COALESCE(i.stop_billing_at, NOW())) > d.date 
            THEN 24.0
            
            -- Instance starts and ends on the same day
            WHEN DATE(i.activated_at) = d.date 
                AND DATE(COALESCE(i.stop_billing_at, NOW())) = d.date
            THEN EXTRACT(EPOCH FROM (
                COALESCE(i.stop_billing_at, NOW()) - i.activated_at
            )) / 3600.0
            
            -- Instance starts on this day but continues
            WHEN DATE(i.activated_at) = d.date 
                AND DATE(COALESCE(i.stop_billing_at, NOW())) > d.date
            THEN EXTRACT(EPOCH FROM (
                DATE_TRUNC('day', i.activated_at) + INTERVAL '1 day' - i.activated_at
            )) / 3600.0
            
            -- Instance ends on this day
            WHEN DATE(i.activated_at) < d.date 
                AND DATE(COALESCE(i.stop_billing_at, NOW())) = d.date
            THEN EXTRACT(EPOCH FROM (
                COALESCE(i.stop_billing_at, NOW()) - DATE_TRUNC('day', COALESCE(i.stop_billing_at, NOW()))
            )) / 3600.0
            
            ELSE 0
        END AS hours_on_day,
        
        -- Calculate the revenue for this day
        CASE
            -- Instance runs through the entire day
            WHEN DATE(i.activated_at) < d.date 
                AND DATE(COALESCE(i.stop_billing_at, NOW())) > d.date 
            THEN 24.0 * i.hourly_rate
            
            -- Instance starts and ends on the same day
            WHEN DATE(i.activated_at) = d.date 
                AND DATE(COALESCE(i.stop_billing_at, NOW())) = d.date
            THEN EXTRACT(EPOCH FROM (
                COALESCE(i.stop_billing_at, NOW()) - i.activated_at
            )) / 3600.0 * i.hourly_rate
            
            -- Instance starts on this day but continues
            WHEN DATE(i.activated_at) = d.date 
                AND DATE(COALESCE(i.stop_billing_at, NOW())) > d.date
            THEN EXTRACT(EPOCH FROM (
                DATE_TRUNC('day', i.activated_at) + INTERVAL '1 day' - i.activated_at
            )) / 3600.0 * i.hourly_rate
            
            -- Instance ends on this day
            WHEN DATE(i.activated_at) < d.date 
                AND DATE(COALESCE(i.stop_billing_at, NOW())) = d.date
            THEN EXTRACT(EPOCH FROM (
                COALESCE(i.stop_billing_at, NOW()) - DATE_TRUNC('day', COALESCE(i.stop_billing_at, NOW()))
            )) / 3600.0 * i.hourly_rate
            
            ELSE 0
        END AS daily_revenue
        
    FROM date_series d
    CROSS JOIN instance_audit i
    WHERE 
        -- Only include instances that are billed and not deleted
        i.billed_to IS NOT NULL
        AND i.deleted_at IS NULL
        AND i.activated_at IS NOT NULL
        -- Only include days where the instance was active
        AND d.date >= DATE(i.activated_at)
        AND d.date <= DATE(COALESCE(i.stop_billing_at, NOW()))
)
SELECT
    date,
    COUNT(DISTINCT instance_id) AS active_instance_count,
    SUM(hours_on_day) AS total_instance_hours,
    SUM(daily_revenue) AS instance_revenue
FROM instance_daily_costs
WHERE daily_revenue > 0
GROUP BY date
ORDER BY date DESC;
CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_instance_revenue_date ON daily_instance_revenue(date);

DROP MATERIALIZED VIEW IF EXISTS daily_revenue_summary CASCADE;
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_revenue_summary AS 
SELECT
    COALESCE(iq.date, ud.date, ir.date) as date,
    COALESCE(iq.new_subscriber_count, 0) as new_subscriber_count,
    COALESCE(iq.new_subscriber_revenue, 0) as new_subscriber_revenue,
    COALESCE(ud.paygo_revenue, 0) as paygo_revenue,
    COALESCE(ir.instance_revenue, 0) as instance_revenue
FROM (
    SELECT
        date(updated_at) as date,
        count(*) as new_subscriber_count,
        sum(case
            when quota = 300 then 3
            when quota = 2000 then 10
            else 20
        end) as new_subscriber_revenue
    FROM invocation_quotas
    WHERE quota > 200
    GROUP BY date
) iq
FULL OUTER JOIN (
    SELECT
        date(bucket) as date,
        sum(amount) as paygo_revenue
    FROM usage_data
    WHERE user_id != '5682c3e0-3635-58f7-b7f5-694962450dfc'
    GROUP BY date
) ud ON iq.date = ud.date
FULL OUTER JOIN (
    SELECT 
        date,
        instance_revenue
    FROM daily_instance_revenue
) ir ON COALESCE(iq.date, ud.date) = ir.date
ORDER BY date DESC;
CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_revenue_summary_date ON daily_revenue_summary(date);

CREATE OR REPLACE FUNCTION update_balance_on_instance_delete()
RETURNS TRIGGER AS $$
DECLARE
    v_billed_to_user_id TEXT;
    v_is_public BOOLEAN;
    v_user_permissions INTEGER;
    v_has_job BOOLEAN;
    v_total_cost DECIMAL(10,2);
    v_hours_used DECIMAL(10,6);
    v_billing_start TIMESTAMP;
    v_billing_end TIMESTAMP;
    v_current_date DATE;
    v_daily_hours DECIMAL(10,6);
    v_daily_cost DECIMAL(10,2);
BEGIN
    -- Skip if the instance doesn't have a billed_to (user_id) value.
    IF OLD.billed_to IS NULL THEN
        RETURN OLD;
    END IF;
    v_billed_to_user_id := OLD.billed_to;

    -- Get the chute's public status, user permissions, and check if instance has a job.
    SELECT c.public, u.permissions_bitmask,
           EXISTS(SELECT 1 FROM jobs j WHERE j.instance_id = OLD.instance_id)
    INTO v_is_public, v_user_permissions, v_has_job
    FROM chutes c
    JOIN users u ON u.user_id = v_billed_to_user_id
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

    -- Update the actual user balance table and distribute usage across days.
    IF OLD.activated_at IS NOT NULL THEN
        v_billing_start := OLD.activated_at;
        v_billing_end := LEAST(COALESCE(OLD.stop_billing_at, NOW()), NOW());

        -- Calculate total cost for balance update
        v_hours_used := EXTRACT(EPOCH FROM (v_billing_end - v_billing_start)) / 3600.0;
        v_total_cost := v_hours_used * OLD.hourly_rate;

        -- Update user balance with total cost
        UPDATE users
        SET balance = balance - v_total_cost
        WHERE user_id = v_billed_to_user_id;

        -- Distribute the usage data across each day the instance was active
        v_current_date := DATE(v_billing_start);

        -- Iterate through all days between billing start date and end date.
        WHILE v_current_date <= DATE(v_billing_end) LOOP
            -- Calculate hours for this specific day
            IF DATE(v_billing_start) = v_current_date AND DATE(v_billing_end) = v_current_date THEN
                -- Instance starts and ends on the same day
                v_daily_hours := EXTRACT(EPOCH FROM (v_billing_end - v_billing_start)) / 3600.0;

            ELSIF DATE(v_billing_start) = v_current_date THEN
                -- First day - from start time to end of day
                v_daily_hours := EXTRACT(EPOCH FROM (
                    DATE_TRUNC('day', v_billing_start) + INTERVAL '1 day' - v_billing_start
                )) / 3600.0;

            ELSIF DATE(v_billing_end) = v_current_date THEN
                -- Last day - from start of day to end time
                v_daily_hours := EXTRACT(EPOCH FROM (
                    v_billing_end - DATE_TRUNC('day', v_billing_end)
                )) / 3600.0;

            ELSE
                -- Full day in between
                v_daily_hours := 24.0;
            END IF;

            -- Calculate cost for this day
            v_daily_cost := v_daily_hours * OLD.hourly_rate;

            -- Insert into usage_data with the appropriate day's bucket
            -- Using the start of each day as the bucket timestamp
            IF v_daily_cost > 0 THEN
                INSERT INTO usage_data (user_id, bucket, chute_id, amount, count)
                VALUES (
                    v_billed_to_user_id,
                    DATE_TRUNC('day', v_current_date)::TIMESTAMP,
                    OLD.chute_id,
                    v_daily_cost,
                    1
                )
                ON CONFLICT (user_id, bucket, chute_id)
                DO UPDATE SET
                    amount = usage_data.amount + EXCLUDED.amount,
                    count = usage_data.count + EXCLUDED.count;
            END IF;

            -- Move to next day
            v_current_date := v_current_date + INTERVAL '1 day';
        END LOOP;
    END IF;

    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

-- migrate:down

-- no-op, have to do this manually for now...
