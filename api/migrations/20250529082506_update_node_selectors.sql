-- migrate:up

-- Make sure b200/mi300x are excluded from all existing chutes from the GPU supported list change.
UPDATE chutes 
SET node_selector = jsonb_set(
    jsonb_set(
        node_selector,
        '{exclude}',
        COALESCE(node_selector->'exclude', '[]'::jsonb) || '["b200", "mi300x"]'::jsonb
    ),
    '{supported_gpus}',
    (
        SELECT jsonb_agg(gpu)
        FROM jsonb_array_elements_text(node_selector->'supported_gpus') AS gpu
        WHERE gpu NOT IN ('b200', 'mi300x')
    )
)
WHERE created_at <= '2025-05-29T00:00:00'::timestamp;

-- migrate:down
