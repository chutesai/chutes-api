-- migrate:up
CREATE INDEX idx_inv_id ON partitioned_invocations (invocation_id);
CREATE INDEX idx_inv_chute_err ON partitioned_invocations (started_at, chute_id, error_message);
CREATE INDEX idx_inv_response ON partitioned_invocations (started_at, response_path);

-- migrate:down
DROP INDEX IF EXISTS idx_inv_id;
DROP INDEX IF EXISTS idx_inv_chute_err;
DROP INDEX IF EXISTS idx_inv_response;
