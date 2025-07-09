-- migrate:up
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'nodes' 
        AND column_name = 'seed' 
        AND data_type = 'numeric'
    ) THEN
        ALTER TABLE nodes ALTER COLUMN seed TYPE NUMERIC USING seed::NUMERIC;
    END IF;
END $$;

-- migrate:down
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'nodes' 
        AND column_name = 'seed' 
        AND data_type = 'numeric'
    ) THEN
        ALTER TABLE nodes ALTER COLUMN seed TYPE BIGINT USING seed::BIGINT;
    END IF;
END $$;
