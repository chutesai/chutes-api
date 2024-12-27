-- migrate:up
CREATE INDEX idx_parent_inv_id ON partitioned_invocations (parent_invocation_id);

ALTER TABLE partitioned_invocations DROP COLUMN IF EXISTS reported_at CASCADE;
ALTER TABLE partitioned_invocations DROP COLUMN IF EXISTS report_reason CASCADE;

DROP VIEW IF EXISTS invocations;
CREATE VIEW invocations AS SELECT * FROM partitioned_invocations;

CREATE TRIGGER invocations_insert_trigger
  INSTEAD OF INSERT ON invocations
  FOR EACH ROW EXECUTE FUNCTION insert_invocation();

-- migrate:down
DROP INDEX IF EXISTS idx_parent_inv_id;

ALTER TABLE partitioned_invocations ADD COLUMN IF NOT EXISTS reported_at TIMESTAMP;
ALTER TABLE partitioned_invocations ADD COLUMN IF NOT EXISTS report_reason TEXT;

DROP VIEW IF EXISTS invocations;
CREATE VIEW invocations AS SELECT * FROM partitioned_invocations;

CREATE TRIGGER invocations_insert_trigger
  INSTEAD OF INSERT ON invocations
  FOR EACH ROW EXECUTE FUNCTION insert_invocation();
