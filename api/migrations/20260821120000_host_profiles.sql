-- migrate:up
-- Miner-submitted host profiles (POST /servers/tdx/host_profiles), moved out of object storage.
--
-- These began as transient requests -- park the JSON, generate measurements, delete it -- which is
-- what made a bucket the right home. They are now permanently retained (a fingerprint cannot be
-- inverted back to the topology inputs an RTMR0 regen needs), and reviewing them means asking
-- questions like "which GPU types are waiting?". Permanent structured data you query belongs here,
-- not behind a paginated LIST + N GetObjects.
--
-- The pending/measured lifecycle is `measured_at` rather than two key prefixes, so promotion is an
-- atomic UPDATE instead of copy-then-delete with a crash window in between.
CREATE TABLE host_profiles (
    -- HostProfile.fingerprint: the host-class id, and the join key to a measurement's `fingerprint`.
    fingerprint TEXT PRIMARY KEY,
    -- The parsed document, for querying (GIN below).
    profile JSONB NOT NULL,
    -- Who submitted it. Attribution for triage, not proof: the sr25519 signature is admission
    -- control at the endpoint, so a row existing already means it verified. The signed bytes are
    -- deliberately NOT kept -- re-checking a signature we already checked would only defend
    -- against someone who can write to this table, which nothing else here defends against.
    miner_hotkey TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- NULL = pending (awaiting generation); set = measured, and retained permanently.
    measured_at TIMESTAMPTZ,
    -- When the "new host class submitted" alert went out; NULL means it still needs one.
    notified_at TIMESTAMPTZ
);

-- Pending lookups (the reconciler and the notify sweep) and the published set.
CREATE INDEX idx_host_profiles_measured_at ON host_profiles (measured_at);
-- Containment queries over the document, e.g.
--   SELECT fingerprint FROM host_profiles WHERE profile @> '{"gpu": {"pci_device_ids": ["2901"]}}';
CREATE INDEX idx_host_profiles_profile ON host_profiles USING GIN (profile);

-- migrate:down
DROP TABLE host_profiles;
