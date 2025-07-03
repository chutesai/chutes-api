-- migrate:up

-- add the jobs column to chutes/history table
ALTER TABLE chutes ADD COLUMN IF NOT EXISTS jobs JSONB;
ALTER TABLE chutes_history ADD COLUMN IF NOT EXISTS jobs JSONB;

-- job_id and config_id columns on instances
ALTER TABLE instances ADD COLUMN job_id VARCHAR, config_id VARCHAR, cacert VARCHAR;
ALTER TABLE instances ADD CONSTRAINT fk_instances_job_id FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE SET NULL;
ALTER TABLE instances ADD CONSTRAINT fk_instances_config_id FOREIGN KEY (config_id) REFERENCES launch_configs(config_id) ON DELETE SET NULL;

-- update the history table functions to track jobs
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

-- migrate:down
ALTER TABLE chutes DROP COLUMN IF EXISTS jobs;
ALTER TABLE chutes_history DROP COLUMN IF EXISTS jobs;
ALTER TABLE instances DROP CONSTRAINT IF EXISTS fk_instances_job_id;
ALTER TABLE instances DROP COLUMN IF EXISTS job_id;
ALTER TABLE instances DROP COLUMN IF EXISTS cacert;
