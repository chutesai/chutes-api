-- migrate:up
ALTER TABLE nodes ADD COLUMN seed BIGINT NOT NULL DEFAULT 42;

-- migrate:down
ALTER TABLE nodes DROP COLUMN seed;
