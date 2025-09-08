-- migrate:up
CREATE TABLE IF NOT EXISTS p_llm_metrics (
    chute_id text NOT NULL,
    name varchar,
    date date NOT NULL,
    total_requests bigint DEFAULT 0,
    total_input_tokens bigint DEFAULT 0,
    total_output_tokens bigint DEFAULT 0,
    average_tps numeric DEFAULT 0,
    average_ttft numeric DEFAULT 0,
    created_at timestamp DEFAULT NOW(),
    PRIMARY KEY (chute_id, date)
) PARTITION BY RANGE (date);

CREATE INDEX IF NOT EXISTS idx_p_llm_metrics_date ON p_llm_metrics (date);

-- Function to create partition for a specific date
CREATE OR REPLACE FUNCTION create_llm_metrics_part(partition_date date)
RETURNS void AS $$
DECLARE
    partition_name text;
    start_date date;
    end_date date;
BEGIN
    partition_name := 'p_llm_metrics_' || to_char(partition_date, 'YYYYMMDD');
    start_date := partition_date;
    end_date := partition_date + interval '1 day';
    IF NOT EXISTS (
        SELECT 1 FROM pg_tables
        WHERE tablename = partition_name
        AND schemaname = current_schema()
    ) THEN
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF p_llm_metrics FOR VALUES FROM (%L) TO (%L)',
            partition_name,
            start_date,
            end_date
        );
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to generate metrics for a specific day
CREATE OR REPLACE FUNCTION gen_llm_metrics_for_day(target_date date)
RETURNS TABLE (
    out_chute_id text,
    out_name varchar,
    out_date date,
    out_total_requests bigint,
    out_total_input_tokens bigint,
    out_total_output_tokens bigint,
    out_average_tps numeric,
    out_average_ttft numeric
) AS $$
DECLARE
    one_hour_before_midnight timestamp := (CURRENT_DATE + interval '23 hours')::timestamp;
    target_date_end timestamp := (target_date + interval '1 day')::timestamp;
BEGIN
    -- If target date ends before 1 hour before midnight today, check if we have cached data
    IF target_date_end < one_hour_before_midnight THEN
        RETURN QUERY
        SELECT
            vm.chute_id,
            vm.name,
            vm.date,
            vm.total_requests,
            vm.total_input_tokens,
            vm.total_output_tokens,
            vm.average_tps,
            vm.average_ttft
        FROM p_llm_metrics vm
        WHERE vm.date = target_date;

        IF NOT FOUND THEN
            PERFORM create_llm_metrics_part(target_date);
            INSERT INTO p_llm_metrics (chute_id, name, date, total_requests, total_input_tokens, total_output_tokens, average_tps, average_ttft)
            WITH metrics_for_day AS (
                SELECT
                    i.chute_id,
                    COUNT(*) AS total_requests,
                    SUM((i.metrics->>'it')::int) AS total_input_tokens,
                    SUM((i.metrics->>'ot')::int) AS total_output_tokens,
                    AVG(
                        CASE
                            WHEN i.metrics->>'otps' IS NOT NULL THEN
                                (i.metrics->>'otps')::float
                            WHEN i.metrics->>'ot' IS NOT NULL
                                AND i.metrics->>'ttft' IS NOT NULL
                                AND extract(epoch from i.completed_at - i.started_at) - (i.metrics->>'ttft')::float > 0 THEN
                                (i.metrics->>'ot')::int / (extract(epoch from i.completed_at - i.started_at) - (i.metrics->>'ttft')::float)
                            ELSE
                                NULL
                        END
                    )::numeric AS average_tps,
                    AVG((i.metrics->>'ttft')::float)::numeric AS average_ttft
                FROM invocations i
                WHERE i.started_at >= target_date
                AND i.started_at < target_date + interval '1 day'
                AND i.metrics->>'it' IS NOT NULL
                AND i.completed_at IS NOT NULL
                AND i.error_message IS NULL
                GROUP BY i.chute_id
            ),
            latest_chute_names AS (
                SELECT DISTINCT ON (ch.chute_id)
                    ch.chute_id,
                    COALESCE(ch.name, '[unknown]') AS name
                FROM chute_history ch
                WHERE ch.created_at <= target_date + interval '1 day'
                ORDER BY ch.chute_id, ch.created_at DESC
            )
            SELECT
                m.chute_id,
                COALESCE(lcn.name, '[unknown]'),
                target_date,
                m.total_requests,
                m.total_input_tokens,
                m.total_output_tokens,
                m.average_tps,
                m.average_ttft
            FROM metrics_for_day m
            LEFT JOIN latest_chute_names lcn ON m.chute_id = lcn.chute_id;
            RETURN QUERY
            SELECT
                vm.chute_id,
                vm.name,
                vm.date,
                vm.total_requests,
                vm.total_input_tokens,
                vm.total_output_tokens,
                vm.average_tps,
                vm.average_ttft
            FROM p_llm_metrics vm
            WHERE vm.date = target_date;
        END IF;
    ELSE
        -- For recent data (within 1 hour of midnight today), always query live data from invocations
        RETURN QUERY
        WITH metrics_for_day AS (
            SELECT
                i.chute_id,
                COUNT(*) AS total_requests,
                SUM((i.metrics->>'it')::int) AS total_input_tokens,
                SUM((i.metrics->>'ot')::int) AS total_output_tokens,
		AVG(
                    CASE
                        WHEN i.metrics->>'otps' IS NOT NULL THEN
                            (i.metrics->>'otps')::float
                        WHEN i.metrics->>'ot' IS NOT NULL
                            AND i.metrics->>'ttft' IS NOT NULL
                            AND extract(epoch from i.completed_at - i.started_at) - (i.metrics->>'ttft')::float > 0 THEN
                            (i.metrics->>'ot')::int / (extract(epoch from i.completed_at - i.started_at) - (i.metrics->>'ttft')::float)
                        ELSE
                            NULL
                    END
                )::numeric AS average_tps,
                AVG((i.metrics->>'ttft')::float)::numeric AS average_ttft
            FROM invocations i
            WHERE i.started_at >= target_date
            AND i.started_at < target_date + interval '1 day'
            AND i.metrics->>'it' IS NOT NULL
            AND i.completed_at IS NOT NULL
            AND i.error_message IS NULL
            GROUP BY i.chute_id
        ),
        latest_chute_names AS (
            SELECT DISTINCT ON (ch.chute_id)
                ch.chute_id,
                COALESCE(ch.name, '[unknown]') AS name
            FROM chute_history ch
            ORDER BY ch.chute_id, ch.created_at DESC
        )
        SELECT
            m.chute_id,
            COALESCE(lcn.name, '[unknown]')::varchar,
            target_date,
            m.total_requests::bigint,
            m.total_input_tokens::bigint,
            m.total_output_tokens::bigint,
            m.average_tps,
            m.average_ttft
        FROM metrics_for_day m
        LEFT JOIN latest_chute_names lcn ON m.chute_id = lcn.chute_id;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to get metrics for a date range
CREATE OR REPLACE FUNCTION get_llm_metrics(
    start_date date,
    end_date date
) RETURNS TABLE (
    chute_id text,
    name varchar,
    date date,
    total_requests bigint,
    total_input_tokens bigint,
    total_output_tokens bigint,
    average_tps numeric,
    average_ttft numeric
) AS $$
DECLARE
    curr_date date;
BEGIN
    curr_date := start_date;
    WHILE curr_date <= end_date LOOP
        RETURN QUERY
        SELECT * FROM gen_llm_metrics_for_day(curr_date);
        curr_date := curr_date + 1;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to populate metrics for a specific day (for backfilling)
CREATE OR REPLACE FUNCTION populate_llm_metrics_for_day(target_date date)
RETURNS integer AS $$
DECLARE
    yesterday date := CURRENT_DATE - interval '1 day';
    row_count integer := 0;
BEGIN
    IF target_date < yesterday THEN
        SELECT COUNT(*) INTO row_count
        FROM p_llm_metrics
        WHERE date = target_date;
        IF row_count = 0 THEN
            PERFORM create_llm_metrics_part(target_date);
            INSERT INTO p_llm_metrics (chute_id, name, date, total_requests, total_input_tokens, total_output_tokens, average_tps, average_ttft)
            WITH metrics_for_day AS (
                SELECT
                    i.chute_id,
                    COUNT(*) AS total_requests,
                    SUM((i.metrics->>'it')::int) AS total_input_tokens,
                    SUM((i.metrics->>'ot')::int) AS total_output_tokens,
		    AVG(
                        CASE
                            WHEN i.metrics->>'otps' IS NOT NULL THEN
                                (i.metrics->>'otps')::float
                            WHEN i.metrics->>'ot' IS NOT NULL
                                AND i.metrics->>'ttft' IS NOT NULL
                                AND extract(epoch from i.completed_at - i.started_at) - (i.metrics->>'ttft')::float > 0 THEN
                                (i.metrics->>'ot')::int / (extract(epoch from i.completed_at - i.started_at) - (i.metrics->>'ttft')::float)
                            ELSE
                                NULL
                        END
                    )::numeric AS average_tps,
                    AVG((i.metrics->>'ttft')::float)::numeric AS average_ttft
                FROM invocations i
                WHERE i.started_at >= target_date
                AND i.started_at < target_date + interval '1 day'
                AND i.metrics->>'it' IS NOT NULL
                AND i.completed_at IS NOT NULL
                AND i.error_message IS NULL
                GROUP BY i.chute_id
            ),
            latest_chute_names AS (
                SELECT DISTINCT ON (ch.chute_id)
                    ch.chute_id,
                    COALESCE(ch.name, '[unknown]') AS name
                FROM chute_history ch
                WHERE ch.created_at <= target_date + interval '1 day'
                ORDER BY ch.chute_id, ch.created_at DESC
            )
            SELECT
                m.chute_id,
                COALESCE(lcn.name, '[unknown]'),
                target_date,
                m.total_requests,
                m.total_input_tokens,
                m.total_output_tokens,
                m.average_tps,
                m.average_ttft
            FROM metrics_for_day m
            LEFT JOIN latest_chute_names lcn ON m.chute_id = lcn.chute_id;
            GET DIAGNOSTICS row_count = ROW_COUNT;
        END IF;
    END IF;

    RETURN row_count;
END;
$$ LANGUAGE plpgsql;

-- Function to backfill historical data (only for days before yesterday)
CREATE OR REPLACE FUNCTION backfill_llm_metrics(
    start_date date,
    end_date date DEFAULT CURRENT_DATE - interval '2 days'
) RETURNS void AS $$
DECLARE
    curr_date date;
    rows_inserted integer;
    total_rows bigint := 0;
    max_allowed_date date := CURRENT_DATE - interval '2 days';
BEGIN
    IF end_date > max_allowed_date THEN
        end_date := max_allowed_date;
        RAISE NOTICE 'Adjusted end_date to % (cannot cache yesterday or today)', end_date;
    END IF;
    curr_date := start_date;
    WHILE curr_date <= end_date LOOP
        rows_inserted := populate_llm_metrics_for_day(curr_date);
        total_rows := total_rows + rows_inserted;
        RAISE NOTICE 'Processed %: % rows', curr_date, rows_inserted;
        curr_date := curr_date + 1;
    END LOOP;
    RAISE NOTICE 'Backfill complete. Total rows: %', total_rows;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION refresh_llm_metrics_for_day(target_date date)
RETURNS void AS $$
DECLARE
    one_hour_before_midnight timestamp := (CURRENT_DATE + interval '23 hours')::timestamp;
    target_date_end timestamp := (target_date + interval '1 day')::timestamp;
BEGIN
    IF target_date_end >= one_hour_before_midnight THEN
        RAISE EXCEPTION 'Cannot cache metrics for dates within 1 hour of midnight. Use gen_llm_metrics_for_day() to query live data.';
    END IF;
    DELETE FROM p_llm_metrics WHERE date = target_date;
    PERFORM * FROM gen_llm_metrics_for_day(target_date);
END;
$$ LANGUAGE plpgsql;

-- migrate:down
DROP FUNCTION IF EXISTS refresh_llm_metrics_for_day(date);
DROP FUNCTION IF EXISTS backfill_llm_metrics(date, date);
DROP FUNCTION IF EXISTS get_llm_metrics(date, date);
DROP FUNCTION IF EXISTS populate_llm_metrics_for_day(date);
DROP FUNCTION IF EXISTS gen_llm_metrics_for_day(date);
DROP FUNCTION IF EXISTS create_llm_metrics_part(date);
DROP INDEX IF EXISTS idx_p_llm_metrics_date;
DROP TABLE IF EXISTS p_llm_metrics CASCADE;
