-- migrate:up
CREATE TABLE root_passphrase_defaults (
    image_version VARCHAR PRIMARY KEY,
    encrypted_passphrase TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- migrate:down
DROP TABLE root_passphrase_defaults;
