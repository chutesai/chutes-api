-- migrate:up
CREATE INDEX idx_chute_history_chute_id ON chute_history(chute_id);

-- Populate chute_history with current chutes
INSERT INTO chute_history (
  entry_id,
  chute_id,
  user_id,
  version,
  name,
  tagline,
  readme,
  tool_description,
  image_id,
  logo_id,
  public,
  standard_template,
  cords,
  node_selector,
  slug,
  code,
  filename,
  ref_str,
  chutes_version,
  openrouter,
  discount,
  created_at,
  updated_at
)
SELECT
  gen_random_uuid()::text,
  chute_id,
  user_id,
  version,
  name,
  tagline,
  readme,
  tool_description,
  image_id,
  logo_id,
  public,
  standard_template,
  cords,
  node_selector,
  slug,
  code,
  filename,
  ref_str,
  chutes_version,
  openrouter,
  discount,
  created_at,
  updated_at
FROM chutes;

-- Function to track chute creation
CREATE OR REPLACE FUNCTION fn_chute_history_insert()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO chute_history (
	entry_id,
        chute_id,
	user_id,
        version,
        name,
        tagline,
        readme,
        tool_description,
        image_id,
        logo_id,
        public,
        standard_template,
        cords,
        node_selector,
        slug,
        code,
        filename,
        ref_str,
        chutes_version,
        openrouter,
        discount,
        created_at,
        updated_at
    ) VALUES (
        gen_random_uuid()::text,
        NEW.chute_id,
	NEW.user_id,
        NEW.version,
        NEW.name,
        NEW.tagline,
        NEW.readme,
        NEW.tool_description,
        NEW.image_id,
        NEW.logo_id,
        NEW.public,
        NEW.standard_template,
        NEW.cords,
        NEW.node_selector,
        NEW.slug,
        NEW.code,
        NEW.filename,
        NEW.ref_str,
        NEW.chutes_version,
        NEW.openrouter,
        NEW.discount,
        NEW.created_at,
        NEW.updated_at
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to track chute updates (only when updated_at changes)
CREATE OR REPLACE FUNCTION fn_chute_history_update()
RETURNS TRIGGER AS $$
BEGIN
    -- Only create a history record if the updated_at timestamp has changed
    IF OLD.updated_at IS DISTINCT FROM NEW.updated_at THEN
        INSERT INTO chute_history (
            entry_id,
            chute_id,
            user_id,
            version,
            name,
            tagline,
            readme,
            tool_description,
            image_id,
            logo_id,
            public,
            standard_template,
            cords,
            node_selector,
            slug,
            code,
            filename,
            ref_str,
            chutes_version,
            openrouter,
            discount,
            created_at,
            updated_at
        ) VALUES (
            gen_random_uuid()::text,
            NEW.chute_id,
            NEW.user_id,
            NEW.version,
            NEW.name,
            NEW.tagline,
            NEW.readme,
            NEW.tool_description,
            NEW.image_id,
            NEW.logo_id,
            NEW.public,
            NEW.standard_template,
            NEW.cords,
            NEW.node_selector,
            NEW.slug,
            NEW.code,
            NEW.filename,
            NEW.ref_str,
            NEW.chutes_version,
            NEW.openrouter,
            NEW.discount,
            NEW.created_at,
            NEW.updated_at
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to track chute deletion
CREATE OR REPLACE FUNCTION fn_chute_history_delete()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE chute_history
    SET deleted_at = NOW()
    WHERE chute_id = OLD.chute_id
    AND version = OLD.version
    AND entry_id = (
        SELECT entry_id
        FROM chute_history
        WHERE chute_id = OLD.chute_id
        AND version = OLD.version
        ORDER BY created_at DESC
        LIMIT 1
    );
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

-- Insert/update/delete triggers.
CREATE TRIGGER tr_chute_history_insert
    AFTER INSERT ON chutes
    FOR EACH ROW
    EXECUTE FUNCTION fn_chute_history_insert();
CREATE TRIGGER tr_chute_history_update
    AFTER UPDATE ON chutes
    FOR EACH ROW
    EXECUTE FUNCTION fn_chute_history_update();
CREATE TRIGGER tr_chute_history_delete
    BEFORE DELETE ON chutes
    FOR EACH ROW
    EXECUTE FUNCTION fn_chute_history_delete();

-- migrate:down
DROP TRIGGER IF EXISTS tr_chute_history_update ON chutes;
DROP TRIGGER IF EXISTS tr_chute_history_delete ON chutes;
DROP TRIGGER IF EXISTS tr_chute_history_insert ON chutes;
DROP FUNCTION IF EXISTS fn_chute_history_update;
DROP FUNCTION IF EXISTS fn_chute_history_delete;
DROP FUNCTION IF EXISTS fn_chute_history_insert;
DROP INDEX IF EXISTS idx_chute_history_chute_id;
DROP TABLE IF EXISTS chute_history;
