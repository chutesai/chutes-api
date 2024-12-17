-- migrate:up
ALTER TABLE partitioned_invocations ADD COLUMN IF NOT EXISTS parent_invocation_id TEXT;
UPDATE partitioned_invocations SET parent_invocation_id = invocation_id WHERE parent_invocation_id IS NULL;

DROP VIEW IF EXISTS invocations;
CREATE VIEW invocations AS SELECT * FROM partitioned_invocations;

CREATE TRIGGER invocations_insert_trigger
  INSTEAD OF INSERT ON invocations
  FOR EACH ROW EXECUTE FUNCTION insert_invocation();

-- migrate:down
ALTER TABLE partitioned_invocations DROP COLUMN IF EXISTS parent_invocation_id;

DROP VIEW IF EXISTS invocations;
CREATE VIEW invocations AS SELECT * FROM partitioned_invocations;

CREATE TRIGGER invocations_insert_trigger
  INSTEAD OF INSERT ON invocations
  FOR EACH ROW EXECUTE FUNCTION insert_invocation();
