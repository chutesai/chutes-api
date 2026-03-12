-- migrate:up
CREATE TABLE IF NOT EXISTS sub_history (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    quota BIGINT NOT NULL,
    effective_date TIMESTAMP,
    first_date TIMESTAMP NOT NULL DEFAULT now(),
    UNIQUE (user_id, quota, effective_date)
);

CREATE INDEX idx_sub_history_user ON sub_history (user_id);

CREATE OR REPLACE FUNCTION fn_sub_history_upsert()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.chute_id = '*' THEN
        INSERT INTO sub_history (user_id, quota, effective_date)
        VALUES (NEW.user_id, NEW.quota, NEW.effective_date)
        ON CONFLICT (user_id, quota, effective_date) DO NOTHING;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_sub_history_insert
    AFTER INSERT ON invocation_quotas
    FOR EACH ROW
    EXECUTE FUNCTION fn_sub_history_upsert();

CREATE TRIGGER trg_sub_history_update
    AFTER UPDATE ON invocation_quotas
    FOR EACH ROW
    EXECUTE FUNCTION fn_sub_history_upsert();

-- migrate:down
DROP TRIGGER IF EXISTS trg_sub_history_update ON invocation_quotas;
DROP TRIGGER IF EXISTS trg_sub_history_insert ON invocation_quotas;
DROP FUNCTION IF EXISTS fn_sub_history_upsert();
DROP TABLE IF EXISTS sub_history;
