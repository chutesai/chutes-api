-- migrate:up
CREATE TABLE instance_audit (
    instance_id TEXT PRIMARY KEY,
    chute_id TEXT NOT NULL,
    version TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    verified_at TIMESTAMP,
    deleted_at TIMESTAMP,
    deletion_reason TEXT,
    miner_uid TEXT NOT NULL,
    miner_hotkey TEXT NOT NULL,
    region TEXT
);

-- populate the timestamps
UPDATE instances SET last_verified_at = CURRENT_TIMESTAMP WHERE verified = true;

-- add some indices
CREATE INDEX idx_instance_audit_ts ON instance_audit (verified_at, deleted_at);
CREATE INDEX idx_instance_audit_miner ON instance_audit (miner_hotkey, miner_uid);

-- pre-populate existing instances
INSERT INTO instance_audit (
   instance_id,
   chute_id,
   version,
   created_at,
   verified_at,
   miner_uid,
   miner_hotkey,
   region
)
SELECT
   i.instance_id,
   i.chute_id,
   c.version,
   i.created_at,
   i.last_verified_at,
   i.miner_uid,
   i.miner_hotkey,
   i.region
FROM instances i
JOIN chutes c
  ON i.chute_id = c.chute_id;

-- trigger to track instance creation
CREATE OR REPLACE FUNCTION fn_instance_audit_insert()
RETURNS TRIGGER AS $$
DECLARE
    version TEXT;
BEGIN
    SELECT INTO version c.version 
      FROM chutes c 
     WHERE c.chute_id = NEW.chute_id;

    INSERT INTO instance_audit (
        instance_id,
        chute_id,
        version,
	miner_uid,
	miner_hotkey,
	region
    ) VALUES (
        NEW.instance_id,
        NEW.chute_id,
        version,
	NEW.miner_uid,
	NEW.miner_hotkey,
	NEW.region
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- trigger to track instance updates (verification is the only thing we care about here)
CREATE OR REPLACE FUNCTION fn_instance_audit_update()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.last_verified_at IS NOT NULL AND OLD.last_verified_at IS NULL THEN
        UPDATE instance_audit
           SET verified_at = NEW.last_verified_at
         WHERE instance_id = NEW.instance_id
           AND verified_at IS NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- trigger to track instance deletions
CREATE OR REPLACE FUNCTION fn_instance_audit_delete()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE instance_audit
       SET deleted_at = NOW()
     WHERE instance_id = OLD.instance_id
       AND deleted_at IS NULL;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_instance_audit_delete
    BEFORE DELETE ON instances
    FOR EACH ROW
    EXECUTE FUNCTION fn_instance_audit_delete();

CREATE TRIGGER tr_instance_audit_insert
    AFTER INSERT ON instances
    FOR EACH ROW
    EXECUTE FUNCTION fn_instance_audit_insert();

CREATE TRIGGER tr_instance_audit_update
    AFTER UPDATE ON instances
    FOR EACH ROW
    EXECUTE FUNCTION fn_instance_audit_update();

-- migrate:down
DROP TRIGGER IF EXISTS tr_instance_audit_update ON instances;
DROP TRIGGER IF EXISTS tr_instance_audit_delete ON instances;
DROP TRIGGER IF EXISTS tr_instance_audit_insert ON instances;
DROP FUNCTION IF EXISTS fn_instance_audit_update;
DROP FUNCTION IF EXISTS fn_instance_audit_delete;
DROP FUNCTION IF EXISTS fn_instance_audit_insert;
DROP TABLE IF EXISTS instance_audit;
