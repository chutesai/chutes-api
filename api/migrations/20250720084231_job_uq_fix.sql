-- migrate:up
ALTER TABLE launch_configs DROP CONSTRAINT IF EXISTS uq_job_launch_config;
DROP INDEX IF EXISTS launch_configs_job_id_key;

-- migrate:down
ALTER TABLE launch_configs ADD CONSTRAINT uq_job_launch_config UNIQUE (job_id);
