-- migrate:up
CREATE TABLE IF NOT EXISTS node_history (
  entry_id TEXT NOT NULL PRIMARY KEY,
  node_id TEXT NOT NULL,
  miner_hotkey TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL,
  deleted_at TIMESTAMP
);
CREATE INDEX idx_node_history_node_id ON node_history(node_id);

INSERT INTO node_history (
  entry_id,
  node_id,
  miner_hotkey,
  created_at
)
SELECT
  gen_random_uuid()::text,
  uuid,
  miner_hotkey,
  created_at
FROM nodes;

-- Function to track node creation
CREATE OR REPLACE FUNCTION fn_node_history_insert()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO node_history (
	entry_id,
	node_id,
	miner_hotkey,
        created_at
    ) VALUES (
        gen_random_uuid()::text,
        NEW.uuid,
	NEW.miner_hotkey,
	NEW.created_at
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to track node deletion
CREATE OR REPLACE FUNCTION fn_node_history_delete()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE node_history
    SET deleted_at = NOW()
    WHERE node_id = OLD.uuid
    AND miner_hotkey = OLD.miner_hotkey
    AND entry_id = (
        SELECT entry_id
        FROM node_history
        WHERE node_id = OLD.uuid
        AND miner_hotkey = OLD.miner_hotkey
        ORDER BY created_at DESC
        LIMIT 1
    );
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

-- Insert/update/delete triggers.
CREATE TRIGGER tr_node_history_insert
    AFTER INSERT ON nodes
    FOR EACH ROW
    EXECUTE FUNCTION fn_node_history_insert();
CREATE TRIGGER tr_node_history_delete
    BEFORE DELETE ON nodes
    FOR EACH ROW
    EXECUTE FUNCTION fn_node_history_delete();

-- migrate:down
DROP TRIGGER IF EXISTS tr_node_history_delete ON nodes;
DROP TRIGGER IF EXISTS tr_node_history_insert ON nodes;
DROP FUNCTION IF EXISTS fn_node_history_delete;
DROP FUNCTION IF EXISTS fn_node_history_insert;
DROP INDEX IF EXISTS idx_node_history_node_id;
DROP TABLE IF EXISTS node_history;
