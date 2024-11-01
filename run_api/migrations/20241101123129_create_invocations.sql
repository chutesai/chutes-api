-- migrate:up
-- store invocations in a partitioned table, where each partition stores one week of data
CREATE TABLE partitioned_invocations (
    invocation_id TEXT NOT NULL,
    chute_id TEXT NOT NULL,
    function_name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    image_id TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    miner_uid INTEGER NOT NULL,
    miner_hotkey TEXT NOT NULL,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    error TEXT,
    request_path TEXT,
    response_PATH TEXT
) PARTITION BY RANGE (started_at);

-- make a view so we can do some fancy insert logic in a function and make queries easier
CREATE VIEW invocations AS
  SELECT * FROM partitioned_invocations;

CREATE OR REPLACE FUNCTION insert_invocation()
RETURNS TRIGGER AS $$
DECLARE
    partition_start TIMESTAMP;
    partition_end TIMESTAMP;
    partition_name TEXT;
BEGIN
    partition_start := date_trunc('week', NEW.started_at);
    partition_end := partition_start + INTERVAL '1 week';

    -- check if this table already exists
    partition_name := 'partitioned_invocations_' || to_char(partition_start, 'IYYY_IW');
    IF NOT EXISTS (
        SELECT FROM pg_catalog.pg_class c
          JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
         WHERE c.relname = partition_name AND n.nspname = 'public'
    ) THEN
        EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF partitioned_invocations FOR VALUES FROM (%L) TO (%L)', partition_name, partition_start, partition_end);
    END IF;

    -- insert directly into the new partition
    INSERT INTO partitioned_invocations VALUES (NEW.*);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- create the insert trigger
CREATE TRIGGER invocations_insert_trigger
INSTEAD OF INSERT ON invocations
FOR EACH ROW EXECUTE FUNCTION insert_invocation();

-- migrate:down
DROP TRIGGER IF EXISTS invocations_insert_trigger ON invocations;
DROP FUNCTION IF EXISTS insert_invocation();
DROP VIEW IF EXISTS invocations;
DO $$
DECLARE
    partition_name TEXT;
BEGIN
    FOR partition_name IN
        SELECT
            inhrelid::regclass::text AS partition_name
        FROM
            pg_inherits
        WHERE
            inhparent = 'partitioned_invocations'::regclass
    LOOP
        EXECUTE format('DROP TABLE IF EXISTS %I;', partition_name);
    END LOOP;
    DROP TABLE IF EXISTS partitioned_invocations;
END;
$$;
