-- migrate:up
CREATE TABLE instance_node_history (
  instance_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  miner_hotkey TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  PRIMARY KEY (instance_id, node_id)
);

CREATE INDEX idx_instance_node_history_hk_node ON instance_node_history(miner_hotkey, node_id);
CREATE INDEX idx_instance_node_history_instance_id ON instance_node_history(instance_id);
CREATE INDEX idx_instance_node_history_node_id ON instance_node_history(node_id);
CREATE INDEX idx_instance_node_history_miner_hotkey ON instance_node_history(miner_hotkey);

-- Backfill the history table with existing instance_nodes entries
INSERT INTO instance_node_history (instance_id, node_id, miner_hotkey)
  SELECT ins.instance_id, ins.node_id, i.miner_hotkey
  FROM instance_nodes ins
  JOIN instances i ON ins.instance_id = i.instance_id;

-- Create the trigger function with miner_hotkey
CREATE OR REPLACE FUNCTION track_instance_node_assignment()
RETURNS TRIGGER AS $$
DECLARE
  v_miner_hotkey TEXT;
BEGIN
  SELECT miner_hotkey INTO v_miner_hotkey
  FROM instances
  WHERE instance_id = NEW.instance_id;
  IF v_miner_hotkey IS NOT NULL THEN
    INSERT INTO instance_node_history (instance_id, node_id, miner_hotkey)
    VALUES (NEW.instance_id, NEW.node_id, v_miner_hotkey)
    ON CONFLICT (instance_id, node_id) DO NOTHING;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger on the instance_nodes table
CREATE TRIGGER track_instance_node_assignment_trigger
  AFTER INSERT ON instance_nodes
  FOR EACH ROW
  EXECUTE FUNCTION track_instance_node_assignment();

-- migrate:down
DROP TRIGGER IF EXISTS track_instance_node_assignment_trigger ON instance_nodes;
DROP FUNCTION IF EXISTS track_instance_node_assignment();
DROP TABLE IF EXISTS instance_node_history;
DROP INDEX IF EXISTS idx_instance_node_history_hk_node;
DROP INDEX IF EXISTS idx_instance_node_history_instance_id;
DROP INDEX IF EXISTS idx_instance_node_history_node_id;
DROP INDEX IF EXISTS idx_instance_node_history_miner_hotkey;
