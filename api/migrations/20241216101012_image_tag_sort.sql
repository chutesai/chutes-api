-- migrate:up
CREATE OR REPLACE FUNCTION version_numbers(ver text) RETURNS bigint AS $$
DECLARE
    nums int[];
    result bigint;
BEGIN
    nums := (SELECT array_agg(num::int)
            FROM (SELECT (REGEXP_MATCHES(ver, '(\d+)', 'g'))[1] AS num) AS t);
    result := (COALESCE(nums[1], 0) * 1000000) + 
              (COALESCE(nums[2], 0) * 1000) + 
              COALESCE(nums[3], 0);
              
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- migrate:down
DROP FUNCTION IF EXISTS version_numbers(text);
