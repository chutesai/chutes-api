-- migrate:up
ALTER TABLE price_overrides ADD COLUMN IF NOT EXISTS pricing_rules JSONB;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'ck_price_overrides_pricing_rules'
          AND conrelid = 'price_overrides'::regclass
    ) THEN
        ALTER TABLE price_overrides
            ADD CONSTRAINT ck_price_overrides_pricing_rules
            CHECK (
                pricing_rules IS NULL
                OR jsonb_typeof(pricing_rules) = 'array'
            );
    END IF;
END $$;

-- migrate:down
ALTER TABLE price_overrides DROP CONSTRAINT IF EXISTS ck_price_overrides_pricing_rules;
ALTER TABLE price_overrides DROP COLUMN IF EXISTS pricing_rules;
