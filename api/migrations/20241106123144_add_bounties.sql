-- migrate:up
CREATE TABLE bounties (
  chute_id TEXT PRIMARY KEY REFERENCES chutes(chute_id) ON DELETE CASCADE,
  bounty INTEGER NOT NULL DEFAULT 100,
  previous_bounty INTEGER NOT NULL DEFAULT 100,
  last_increased_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- function to increase bounties (with rate limits and caps)
CREATE OR REPLACE FUNCTION increase_bounty(target_chute_id TEXT)
RETURNS TABLE(bounty INTEGER, last_increased_at TIMESTAMP) AS $$
DECLARE
    current_bounty INTEGER;
    increased_at TIMESTAMP;
BEGIN
    RETURN QUERY
    with updated AS (
        INSERT INTO bounties (chute_id, bounty, previous_bounty, last_increased_at)
        VALUES (target_chute_id, 100, 100, CURRENT_TIMESTAMP)
        ON CONFLICT (chute_id) DO UPDATE
        SET previous_bounty = bounties.bounty,
            bounty = CASE
                WHEN CURRENT_TIMESTAMP - bounties.last_increased_at >= interval '20 seconds'
                     AND bounties.bounty + bounties.previous_bounty <= 5000
                THEN bounties.bounty + bounties.previous_bounty
                ELSE bounties.bounty
            END,
            last_increased_at = CASE
                WHEN CURRENT_TIMESTAMP - bounties.last_increased_at >= interval '20 seconds'
                     AND bounties.bounty + bounties.previous_bounty <= 5000
                THEN CURRENT_TIMESTAMP
                ELSE bounties.last_increased_at
            END
        RETURNING bounties.bounty, bounties.last_increased_at
    )
    SELECT * FROM updated;
END;
$$ LANGUAGE plpgsql;

-- automatically create bounties when chutes are created
CREATE OR REPLACE FUNCTION initialize_bounty()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM increase_bounty(NEW.chute_id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE TRIGGER after_chute_insert
    AFTER INSERT ON chutes
    FOR EACH ROW
    EXECUTE FUNCTION initialize_bounty();

-- migrate:down
DROP TRIGGER IF EXISTS after_chute_insert ON chutes;
DROP FUNCTION IF EXISTS initialize_bounty;
DROP FUNCTION IF EXISTS increase_bounty;
DROP TABLE IF EXISTS bounties;
