-- migrate:up
ALTER TABLE chutes ADD COLUMN IF NOT EXISTS jobs JSONB;
ALTER TABLE chute_history ADD COLUMN IF NOT EXISTS jobs JSONB;
ALTER TABLE instances ADD COLUMN IF NOT EXISTS config_id VARCHAR;
ALTER TABLE instances ADD COLUMN IF NOT EXISTS cacert VARCHAR;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'fk_instances_config_id'
        AND table_name = 'instances'
    ) THEN
        ALTER TABLE instances
        ADD CONSTRAINT fk_instances_config_id
        FOREIGN KEY (config_id)
        REFERENCES launch_configs(config_id)
        ON DELETE SET NULL;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'uq_instances_config_id'
        AND table_name = 'instances'
    ) THEN
        ALTER TABLE instances
        ADD CONSTRAINT uq_instances_config_id
        UNIQUE (config_id);
    END IF;
END $$;

CREATE OR REPLACE FUNCTION fn_reset_job_assignment()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.job_id IS NOT NULL AND (NEW.failed_at IS NOT NULL OR NEW.verification_error IS NOT NULL) THEN
        UPDATE jobs SET miner_hotkey = NULL, miner_coldkey = NULL, instance_id = NULL WHERE job_id = OLD.job_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_reset_jobs_on_fail
    AFTER UPDATE ON launch_configs
    FOR EACH ROW
    EXECUTE FUNCTION fn_reset_job_assignment();

CREATE OR REPLACE FUNCTION fn_chute_history_insert()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO chute_history (
        entry_id,
        chute_id,
        user_id,
        version,
        name,
        tagline,
        readme,
        tool_description,
        image_id,
        logo_id,
        public,
        standard_template,
        cords,
        jobs,
        node_selector,
        slug,
        code,
        filename,
        ref_str,
        chutes_version,
        openrouter,
        discount,
        created_at,
        updated_at
    ) VALUES (
        gen_random_uuid()::text,
        NEW.chute_id,
        NEW.user_id,
        NEW.version,
        NEW.name,
        NEW.tagline,
        NEW.readme,
        NEW.tool_description,
        NEW.image_id,
        NEW.logo_id,
        NEW.public,
        NEW.standard_template,
        NEW.cords,
        NEW.jobs,
        NEW.node_selector,
        NEW.slug,
        NEW.code,
        NEW.filename,
        NEW.ref_str,
        NEW.chutes_version,
        NEW.openrouter,
        NEW.discount,
        NEW.created_at,
        NEW.updated_at
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION fn_chute_history_update()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.updated_at IS DISTINCT FROM NEW.updated_at THEN
        INSERT INTO chute_history (
            entry_id,
            chute_id,
            user_id,
            version,
            name,
            tagline,
            readme,
            tool_description,
            image_id,
            logo_id,
            public,
            standard_template,
            cords,
            jobs,
            node_selector,
            slug,
            code,
            filename,
            ref_str,
            chutes_version,
            openrouter,
            discount,
            created_at,
            updated_at
        ) VALUES (
            gen_random_uuid()::text,
            NEW.chute_id,
            NEW.user_id,
            NEW.version,
            NEW.name,
            NEW.tagline,
            NEW.readme,
            NEW.tool_description,
            NEW.image_id,
            NEW.logo_id,
            NEW.public,
            NEW.standard_template,
            NEW.cords,
            NEW.jobs,
            NEW.node_selector,
            NEW.slug,
            NEW.code,
            NEW.filename,
            NEW.ref_str,
            NEW.chutes_version,
            NEW.openrouter,
            NEW.discount,
            NEW.created_at,
            NEW.updated_at
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- migrate:down
ALTER TABLE chutes DROP COLUMN IF EXISTS jobs;
ALTER TABLE chute_history DROP COLUMN IF EXISTS jobs;
ALTER TABLE instances DROP CONSTRAINT IF EXISTS uq_instances_config_id;
ALTER TABLE instances DROP CONSTRAINT IF EXISTS fk_instances_config_id;
ALTER TABLE instances DROP COLUMN IF EXISTS config_id;
ALTER TABLE instances DROP COLUMN IF EXISTS cacert;
DROP TRIGGER IF EXISTS tr_reset_jobs_on_fail ON launch_configs;
DROP FUNCTION IF EXISTS fn_reset_job_assignment();
