-- migrate:up
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_revenue_summary AS SELECT
    COALESCE(iq.date, ud.date) as date,
    COALESCE(iq.new_subscriber_count, 0) as new_subscriber_count,
    COALESCE(iq.new_subscriber_revenue, 0) as new_subscriber_revenue,
    COALESCE(ud.paygo_revenue, 0) as paygo_revenue
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
ORDER BY date DESC
LIMIT 30;
CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_revenue_summary_date ON daily_revenue_summary(date);

-- migrate:down
DROP INDEX IF EXISTS idx_daily_revenue_summary_date;
DROP MATERIALIZED VIEW IF EXISTS daily_revenue_summary;
