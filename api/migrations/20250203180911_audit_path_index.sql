-- migrate:up
CREATE INDEX idx_audit_path ON audit_entries (path);
CREATE INDEX idx_audit_date ON audit_entries (start_time);

-- migrate:down
DROP INDEX idx_audit_path;
DROP INDEX idx_audit_date;
