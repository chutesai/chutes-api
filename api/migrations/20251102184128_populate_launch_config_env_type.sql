-- migrate:up
ALTER TABLE launch_configs ADD COLUMN env_type VARCHAR;
UPDATE launch_configs SET env_type = 'graval';
ALTER TABLE launch_configs ALTER COLUMN env_type SET NOT NULL;

-- migrate:down
ALTER TABLE launch_configs DROP COLUMN env_type;