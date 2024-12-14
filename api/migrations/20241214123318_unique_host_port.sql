-- migrate:up
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'unique_host_port'
    ) THEN
        ALTER TABLE instances ADD CONSTRAINT unique_host_port UNIQUE (host, port);
    END IF;
END $$;

-- migrate:down
ALTER TABLE instances DROP CONSTRAINT IF EXISTS unique_host_port;
