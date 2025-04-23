-- migrate:up
ALTER TABLE instances ADD COLUMN version text NOT NULL DEFAULT '';

UPDATE instances 
SET version = (
  SELECT chutes.version 
  FROM chutes 
  WHERE chutes.chute_id = instances.chute_id
);

ALTER TABLE instances ALTER COLUMN version DROP DEFAULT;

-- migrate:down
ALTER TABLE instances DROP COLUMN IF EXISTS version;
