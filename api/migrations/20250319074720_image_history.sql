-- migrate:up
CREATE INDEX idx_image_history_image_id ON image_history(image_id);

INSERT INTO image_history (
    entry_id,
    image_id,
    user_id,
    name,
    tag,
    readme,
    logo_id,
    public,
    status,
    chutes_version,
    build_started_at,
    build_completed_at,
    created_at
)
SELECT
    gen_random_uuid()::text,
    image_id,
    user_id,
    name,
    tag,
    readme,
    logo_id,
    public,
    status,
    chutes_version,
    build_started_at,
    build_completed_at,
    created_at
FROM images;

CREATE OR REPLACE FUNCTION fn_image_history_insert()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO image_history (
	entry_id,
        image_id,
        user_id,
        name,
        tag,
        readme,
        logo_id,
        public,
        status,
        chutes_version,
        build_started_at,
        build_completed_at,
        created_at
    ) VALUES (
        gen_random_uuid()::text,
        NEW.image_id,
        NEW.user_id,
        NEW.name,
        NEW.tag,
        NEW.readme,
        NEW.logo_id,
        NEW.public,
        NEW.status,
        NEW.chutes_version,
        NEW.build_started_at,
        NEW.build_completed_at,
        NEW.created_at
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to track image updates
CREATE OR REPLACE FUNCTION fn_image_history_update()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO image_history (
	entry_id,
        image_id,
        user_id,
        name,
        tag,
        readme,
        logo_id,
        public,
        status,
        chutes_version,
        build_started_at,
        build_completed_at,
        created_at
    ) VALUES (
        gen_random_uuid()::text,
        NEW.image_id,
        NEW.user_id,
        NEW.name,
        NEW.tag,
        NEW.readme,
        NEW.logo_id,
        NEW.public,
        NEW.status,
        NEW.chutes_version,
        NEW.build_started_at,
        NEW.build_completed_at,
        NEW.created_at
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to track image deletion
CREATE OR REPLACE FUNCTION fn_image_history_delete()
RETURNS TRIGGER AS $$
BEGIN
    -- Mark the most recent history entry for this image as deleted
    UPDATE image_history
    SET deleted_at = NOW()
    WHERE image_id = OLD.image_id
    AND entry_id = (
        SELECT entry_id
        FROM image_history
        WHERE image_id = OLD.image_id
        ORDER BY created_at DESC
        LIMIT 1
    );
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

-- Insert/update/delete triggers.
CREATE TRIGGER tr_image_history_insert
    AFTER INSERT ON images
    FOR EACH ROW
    EXECUTE FUNCTION fn_image_history_insert();
CREATE TRIGGER tr_image_history_update
    AFTER UPDATE ON images
    FOR EACH ROW
    EXECUTE FUNCTION fn_image_history_update();
CREATE TRIGGER tr_image_history_delete
    BEFORE DELETE ON images
    FOR EACH ROW
    EXECUTE FUNCTION fn_image_history_delete();

-- migrate:down
DROP TRIGGER IF EXISTS tr_image_history_update ON images;
DROP TRIGGER IF EXISTS tr_image_history_delete ON images;
DROP TRIGGER IF EXISTS tr_image_history_insert ON images;
DROP FUNCTION IF EXISTS fn_image_history_update;
DROP FUNCTION IF EXISTS fn_image_history_delete;
DROP FUNCTION IF EXISTS fn_image_history_insert;
DROP INDEX IF EXISTS idx_image_history_image_id;
DROP TABLE IF EXISTS image_history;
