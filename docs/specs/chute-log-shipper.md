# Chute Log Shipper — validator (chutes-api) side

Captures a chute pod's **pre-registration** stdout/stderr so a launch that never comes up can be
debugged. Before a chute activates there is no `Instance` (and no live 8001 log server) to proxy,
so an in-guest shipper streams the pod's CRI logs to a dedicated in-namespace Loki, keyed by the
launch `config_id` — the only stable identifier available in that window.

## Ingest

`POST /instances/launch_config/{config_id}/logs`

- **Transport:** mTLS only, via the CVM proxy (`cvm.chutes.ai`). The guest presents its per-boot
  registry mTLS leaf; the proxy stamps `X-Client-Cert` + its provenance secret.
- **Auth (server-derived, never self-asserted):** the leaf is verified against the CAs registered
  for the launch config's owning miner. A leaf signed by another miner's VM CA cannot verify, so
  cross-miner injection is rejected. Identity (`chute_id`, `user_id`, `miner_hotkey`) is resolved
  from `config_id`; `server_ip` comes from the proxy. Resolution is cached in Redis per
  `(config_id, cert)` so a burst of batches verifies once, fleet-wide.
- **Body:** `{ "deployment_id": str, "logs": [ { "ts": <RFC3339 ns>, "stream": "stdout|stderr",
  "log": str } ] }`. `deployment_id` is Grafana-correlation only; everything security-relevant is
  derived server-side.
- **Response (the cutoff signal):** `204` = stop shipping; any other `2xx` = keep sending;
  `403`/`404` = terminal reject (unauthorized / unknown config). The guest treats `204`/`403`/`404`
  as stop.
- **Dedup:** a Redis high-watermark on `(config_id, max line ts)` — idempotent across guest retries,
  so at-least-once shipping (e.g. after a VM restart resuming from its checkpoint) is safe.

## Cutoff (when to stop capturing)

Binary and lifecycle-driven: keep capturing while the launch is live; stop (`204`) once it reaches a
terminal state — instance **activated**, launch config **failed**, or the config **deleted**. A
per-chute **support debug override** keeps capture going past activation. There is deliberately **no
wall-clock cap**: a config that never terminalizes is an anomaly whose logs are exactly what we want,
and ingest is bounded per-batch and by Loki retention.

## Storage

A dedicated in-namespace Loki (not the ops OpenSearch cluster; app logs must never tee to stdout and
leak there).

- **Retention:** 24h.
- **Not client-reachable:** a default-deny NetworkPolicy admits only the API (`loki-access` pod
  label); any additional reader (e.g. the ops Grafana) is granted per-cluster via
  `loki.extraIngressFrom` in the ops-repo values.
- **Labels vs fields:** labels are low-cardinality only (`app`, `stream`). High-cardinality
  identifiers (`config_id`, `chute_id`, `user_id`, `miner_hotkey`, `server_ip`, `deployment_id`) ride
  inside the JSON line and are filtered with `| json | field="…"`.

## Reads

- **Internal / support:** the ops Grafana queries Loki directly (filter by `config_id` / `chute_id` /
  `miner_hotkey` / `server_ip`). A support-role debug-override endpoint set lives at
  `/logs/debug/{chute_id}`.
- **Owner / miner-CLI:** deferred to a standalone, security-reviewed follow-up. The read primitive
  (`service.read_config_logs`, server-forced to the owner's `user_id`) exists and is tested; the
  follow-up only adds the routes + auth.

## Confidentiality

A private chute's logs are visible only to its owner and to Chutes support — enforced by construction,
not by a stored field:

- Loki is not client-reachable (perimeter above).
- The API resolves authorization from the DB at **read time**, then builds LogQL with a
  **server-forced matcher** on the authenticated principal — the client can only narrow, never widen.
- Public/private is **re-resolved from the DB at read time**, never stamped into the store (a chute
  can flip public→private within the retention window).

## Error handling

The service layer raises pure domain errors (`api/chute_logs/exceptions.py`); the FastAPI dependency
(`api/chute_logs/dependencies.py`) maps them to HTTP. Business logic carries no transport knowledge.
