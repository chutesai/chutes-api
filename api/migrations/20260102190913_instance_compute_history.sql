-- migrate:up
CREATE TABLE instance_compute_history (
    id BIGSERIAL PRIMARY KEY,
    instance_id TEXT NOT NULL,
    compute_multiplier DOUBLE PRECISION NOT NULL,
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMP
);
CREATE INDEX idx_ich_instance_time ON instance_compute_history (instance_id, started_at, ended_at);
CREATE INDEX idx_ich_time_range ON instance_compute_history (started_at, ended_at);
CREATE UNIQUE INDEX idx_ich_instance_open ON instance_compute_history (instance_id) WHERE ended_at IS NULL;

-- Only create history records when an instance activates (not on insert).
CREATE OR REPLACE FUNCTION fn_instance_compute_history_update()
RETURNS TRIGGER AS $$
BEGIN
    -- Case 1: Instance just activated (activated_at changed from NULL to non-NULL).
    -- Create startup period record (0.3x rate) and active period record (full rate).
    IF NEW.activated_at IS NOT NULL AND OLD.activated_at IS NULL THEN
        IF NEW.compute_multiplier IS NOT NULL THEN
            -- Insert startup period: created_at to activated_at at 0.3x rate
            INSERT INTO instance_compute_history (instance_id, compute_multiplier, started_at, ended_at)
            VALUES (NEW.instance_id, NEW.compute_multiplier * 0.3, OLD.created_at, NEW.activated_at);

            -- Insert active period: activated_at onwards at full rate
            INSERT INTO instance_compute_history (instance_id, compute_multiplier, started_at)
            VALUES (NEW.instance_id, NEW.compute_multiplier, NEW.activated_at);
        END IF;
    -- Case 2: compute_multiplier changed (e.g., autoscaler adjustment, bounty decay).
    ELSIF NEW.compute_multiplier IS DISTINCT FROM OLD.compute_multiplier
          AND NEW.compute_multiplier IS NOT NULL THEN

        UPDATE instance_compute_history
           SET ended_at = NOW()
         WHERE instance_id = NEW.instance_id
           AND ended_at IS NULL;

        INSERT INTO instance_compute_history (instance_id, compute_multiplier, started_at)
        VALUES (NEW.instance_id, NEW.compute_multiplier, NOW());
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_ich_update
    AFTER UPDATE ON instances
    FOR EACH ROW
    EXECUTE FUNCTION fn_instance_compute_history_update();

CREATE OR REPLACE FUNCTION fn_instance_compute_history_delete()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE instance_compute_history
       SET ended_at = NOW()
     WHERE instance_id = OLD.instance_id
       AND ended_at IS NULL;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_ich_delete
    BEFORE DELETE ON instances
    FOR EACH ROW
    EXECUTE FUNCTION fn_instance_compute_history_delete();

-- Backfill existing instances: startup period (0.3x rate from created_at to activated_at).
INSERT INTO instance_compute_history (instance_id, compute_multiplier, started_at, ended_at)
SELECT
    ia.instance_id,
    COALESCE(i.compute_multiplier, ia.compute_multiplier, 1.0) * 0.3,
    ia.created_at,
    ia.activated_at
FROM instance_audit ia
LEFT JOIN instances i ON i.instance_id = ia.instance_id
WHERE ia.activated_at IS NOT NULL
  AND ia.created_at IS NOT NULL
  AND ia.created_at < ia.activated_at
  AND COALESCE(i.compute_multiplier, ia.compute_multiplier) IS NOT NULL;

-- Backfill existing instances: active period (full rate from activated_at to deleted_at or still open).
INSERT INTO instance_compute_history (instance_id, compute_multiplier, started_at, ended_at)
SELECT
    ia.instance_id,
    COALESCE(i.compute_multiplier, ia.compute_multiplier, 1.0),
    ia.activated_at,
    ia.deleted_at
FROM instance_audit ia
LEFT JOIN instances i ON i.instance_id = ia.instance_id
WHERE ia.activated_at IS NOT NULL
  AND COALESCE(i.compute_multiplier, ia.compute_multiplier) IS NOT NULL;

-- migrate:down
DROP TRIGGER IF EXISTS tr_ich_delete ON instances;
DROP TRIGGER IF EXISTS tr_ich_update ON instances;
DROP FUNCTION IF EXISTS fn_instance_compute_history_delete;
DROP FUNCTION IF EXISTS fn_instance_compute_history_update;
DROP TABLE IF EXISTS instance_compute_history;
