-- migrate:up
ALTER TABLE instances ADD CONSTRAINT unique_host_port UNIQUE (host, port);

-- migrate:down
ALTER TABLE instances DROP CONSTRAINT IF EXISTS unique_host_port;
