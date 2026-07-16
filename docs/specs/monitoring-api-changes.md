# Feature Spec: Monitoring API-Side Changes

Use the sections **Goal**, **Constraints**, **Output Format**, and **Failure Conditions** as a **Prompt Contract** for this task (see [AGENT.md](../../AGENT.md) at repo root).

**Date**: 2026-06-01  
**Status**: draft

---

## Context

Companion changes in `chutes-api` required by the monitoring stack deployed in `chutes-ops`. These are the application-side and Helm chart changes that enable Prometheus scraping, structured log collection, and metrics querying via Mimir.

- **Depends on**: Monitoring stack spec (`docs/specs/monitoring-stack.md`)
- **Packages affected**: None (no new dependencies)
- **Key files**:
  - `charts/templates/api-deployment.yaml` -- Prometheus annotations, Datadog removal
  - `charts/templates/usage-tracker-deployment.yaml` -- Datadog removal
  - `charts/templates/redis-np.yaml` -- monitoring namespace ingress
  - `charts/templates/cm-redis-np.yaml` -- monitoring namespace ingress
  - `charts/templates/quota-redis-np.yaml` -- monitoring namespace ingress
  - `charts/templates/_helpers.tpl` -- `PROMETHEUS_URL` env var
  - `charts/values.yaml` -- remove `datadog_enabled`, add `prometheusUrl`
  - `api/log.py` -- `configure_structured_logging`: JSON-to-stdout sink when `LOG_FORMAT=json`
  - Long-running service deployment templates -- `LOG_FORMAT=json` env via `chutes.loggingEnv`

---

## Design Decisions

- **Remove Datadog entirely** rather than keeping it as a disabled option. The `datadog_enabled` flag, all `tags.datadoghq.com/*` labels, `admission.datadoghq.com/*` annotations, and `DD_LOGS_INJECTION` env vars are all removed.
- **Prometheus annotations always present** on API pods (no longer gated behind a flag).
- **Structured JSON to stdout** (not a file): when `LOG_FORMAT=json`, loguru emits one flat JSON object per line to stdout, which the node Fluent Bit collects and enriches. Each line carries a top-level `text` field for client-side `jq -r .text`. Unset `LOG_FORMAT` keeps human-readable stderr (local dev). No changes to existing log call syntax. Supersedes the abandoned emptyDir file-sink (PR #155).
- **`PROMETHEUS_URL`** added to Helm values and `commonEnv` so it's configurable per environment. Defaults to `http://prometheus-server` for backward compatibility, updated to Mimir endpoint when monitoring stack is deployed.

---

## Changes

### 1. Remove all Datadog references

**`charts/values.yaml`** (line 16):
- Remove `datadog_enabled: false`

**`charts/templates/api-deployment.yaml`**:
- Lines 7-11: Remove `tags.datadoghq.com/*` labels from Deployment metadata
- Lines 26-34: Remove entire `{{- if .Values.datadog_enabled }}` block. Replace with unconditional Prometheus annotations (move to `annotations:` not `labels:`)
- Lines 35-38: Remove Datadog admission annotation block
- Lines 72-75: Remove `DD_LOGS_INJECTION` env var block

**`charts/templates/usage-tracker-deployment.yaml`**:
- Lines 40-43: Remove `DD_LOGS_INJECTION` env var block

**`charts/templates/redis-np.yaml`** (lines 22-27):
- Remove Datadog agent ingress rule block

**`charts/templates/cm-redis-np.yaml`** (lines 22-27):
- Remove Datadog agent ingress rule block

**`charts/templates/quota-redis-np.yaml`** (lines 22-27):
- Remove Datadog agent ingress rule block

### 2. Add Prometheus scrape annotations (unconditional)

**`charts/templates/api-deployment.yaml`**:
- Add to pod template `annotations:` (not `labels:` -- Prometheus expects annotations):

```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/path: /_metrics
  prometheus.io/port: "8000"
```

Note: The current chart incorrectly places these as `labels` instead of `annotations`. They must be `annotations` for Prometheus service discovery to detect them.

### 3. Add `PROMETHEUS_URL` to Helm values and commonEnv

**`charts/values.yaml`**:
- Add: `prometheusUrl: "http://prometheus-server.monitoring.svc.cluster.local"` (cross-namespace URL for Prometheus in the monitoring namespace; permanent -- application code always queries Prometheus directly, not Mimir)

**`charts/templates/_helpers.tpl`** in the `chutes.commonEnv` define block:
- Add:

```yaml
- name: PROMETHEUS_URL
  value: {{ .Values.prometheusUrl }}
```

This makes `PROMETHEUS_URL` available to all pods that use `commonEnv`, including the autoscaler cronjob and API deployment, which both query Prometheus/Mimir in their Python code (`chute_autoscaler.py` line 213, `api/invocation/util.py` line 45).

### 4. Add monitoring namespace ingress to Redis NetworkPolicies

All three Redis NetworkPolicies (`redis-np.yaml`, `cm-redis-np.yaml`, `quota-redis-np.yaml`) need an additional ingress rule allowing `redis_exporter` pods in the `monitoring` namespace to reach Redis.

Add to each NetworkPolicy's `ingress:` array:

```yaml
- from:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: monitoring
      podSelector:
        matchLabels:
          app: prometheus-redis-exporter
  ports:
    - protocol: TCP
      port: {{ .Values.<redis-type>.service.port }}
```

This is a scoped rule: only pods with the `prometheus-redis-exporter` label in the `monitoring` namespace can connect, and only on the Redis port.

### 5. Structured JSON logging to stdout

> **Note:** This supersedes the earlier emptyDir/file-sink approach (PR #155). A node
> Fluent Bit DaemonSet cannot read a pod's `emptyDir`, and the file/sidecar/hostPath
> routes all add cost or ops burden. Instead we emit JSON to **stdout**, which the node
> Fluent Bit already collects and fully enriches (namespace, pod, container, labels,
> annotations) with zero extra config. No volumes, annotations, or Downward API are used.

**`api/log.py`** — `configure_structured_logging()` (called from each entrypoint:
`api/main.py`, `api/socket_server.py`, `api/event_socket_server.py`,
`api/payment/usage_tracker.py`, `api/payment/watcher.py`, `api/image/forge.py`):
- When `LOG_FORMAT=json`, remove loguru's default sink and add a single **stdout** sink
  emitting one JSON object per line.
- When `LOG_FORMAT` is unset (local dev), do nothing — keep loguru's default
  human-readable stderr sink.
- Each JSON line is **flat** (so `logger.bind(...)` context lands at the top level) and
  includes a top-level `text` field holding the rendered human-readable line, so
  `kubectl logs ... | jq -r .text` reproduces a clean log.
- The app does **not** stamp kubernetes metadata — the node Fluent Bit adds the full
  `kubernetes.*` object automatically from the stdout path.

**Deployment templates** — the only chart addition is `LOG_FORMAT=json` (and optional
`LOG_LEVEL`) on the long-running service containers, via the `chutes.loggingEnv` helper:

```yaml
env:
  {{- include "chutes.loggingEnv" . | nindent 12 }}
```

`LOG_FORMAT` / `LOG_LEVEL` are driven by the top-level `logFormat` / `logLevel` values
(default `json` / unset → `INFO`). No `structured-logs` volume, volumeMount, or
`chutes.ai/structured-logs` annotation is needed anymore.

**Which containers ship structured logs** (those whose entrypoints call
`configure_structured_logging`):
- `api-deployment.yaml`
- `socket-deployment.yaml`
- `event-socket-deployment.yaml`
- `forge-deployment.yaml`
- `watchtower-deployment.yaml`
- `usage-tracker-deployment.yaml`

**Client-side readability** — the JSON is rendered human-readable on the client:

```bash
kubectl logs -f deploy/api | jq -r '.text // .message'
# convenience alias:
klogs() { kubectl logs "$@" | jq -r '.text // .message'; }
# or tools that auto-detect JSON: humanlog, fblog, stern
```

In OpenSearch (`api-logs-*`) each doc has the flat app fields (including any
`logger.bind(...)` context at top level) plus a `kubernetes` object
(`namespace_name`/`pod_name`/`container_name`/`labels`) added by the node agent.

### 6. Fix nginx registry-proxy access log

**`charts/templates/registry-proxy-deployment.yaml`**:
- Add `emptyDir` volume for nginx logs:

```yaml
volumes:
  - name: nginx-logs
    emptyDir: {}
```

- Add volumeMount:

```yaml
volumeMounts:
  - name: nginx-logs
    mountPath: /var/log/nginx
```

This makes the registry-proxy access log (`/var/log/nginx/access.log`) available for Fluent Bit to tail, and persists it across container restarts within the pod lifecycle.

---

## Goal

Success =

1. `datadog_enabled` flag and all Datadog-specific labels, annotations, and env vars are removed from all Helm templates and `values.yaml`.
2. Prometheus scrape annotations (`prometheus.io/*`) are present unconditionally on API pod template as `annotations` (not `labels`).
3. `PROMETHEUS_URL` is configurable via `values.yaml` and available to all pods via `commonEnv`.
4. Redis NetworkPolicies allow ingress from `monitoring` namespace for `redis_exporter` pods.
5. When `LOG_FORMAT=json`, `kubectl logs <pod>` shows one JSON object per line; `| jq -r .text` renders clean human-readable lines. With `LOG_FORMAT` unset (local dev), logs stay human-readable on stderr.
6. `logger.bind(request_id=...).info("x")` puts `request_id` at the **top level** of the JSON.
7. `LOG_FORMAT=json` env is present (via `chutes.loggingEnv`) on the long-running service containers; no `emptyDir`/`hostPath` volumes, no `structured-logs` annotation, no Downward API for logging.
8. In OpenSearch (`api-logs-*`), docs have the flat app fields plus a `kubernetes` object added by the node agent.
9. All existing functionality is unchanged -- no log call modifications, no behavior changes.

---

## Constraints

- No new Python dependencies (loguru is already a dependency)
- No changes to existing log call syntax (all `logger.info(f"...")` calls remain as-is)
- `PROMETHEUS_URL` default must use the full cross-namespace form (`http://prometheus-server.monitoring.svc.cluster.local`) since Prometheus lives in the `monitoring` namespace
- NetworkPolicy changes must be scoped (namespace + pod label selector) -- do not open blanket cross-namespace access
- `configure_structured_logging()` must be a no-op when `LOG_FORMAT` is unset (local dev keeps human-readable stderr)

---

## Output Format

1. Modified `charts/values.yaml` -- remove `datadog_enabled`, add `prometheusUrl`
2. Modified `charts/templates/api-deployment.yaml` -- remove Datadog, add Prometheus annotations
3. Modified `charts/templates/usage-tracker-deployment.yaml` -- remove Datadog env var
4. Modified `charts/templates/redis-np.yaml` -- add monitoring namespace ingress
5. Modified `charts/templates/cm-redis-np.yaml` -- add monitoring namespace ingress
6. Modified `charts/templates/quota-redis-np.yaml` -- add monitoring namespace ingress
7. Modified `charts/templates/_helpers.tpl` -- add `PROMETHEUS_URL` to `commonEnv`
8. Modified `api/log.py` -- `configure_structured_logging` emits flat JSON to stdout when `LOG_FORMAT=json`
9. Modified `charts/templates/_helpers.tpl` -- add `chutes.loggingEnv` helper; `charts/values.yaml` -- add `logFormat`/`logLevel`
10. Modified long-running service deployment templates -- add `LOG_FORMAT=json` env; removed `structured-logs` volume/mount/annotation

---

## Failure Conditions

- Any Datadog reference remains in the codebase after changes
- Prometheus annotations are missing or placed as `labels` instead of `annotations`
- `LOG_FORMAT=json` produces nested/non-flat JSON (bound fields buried under `extra` instead of top level), or omits the top-level `text` field
- Existing log calls in Python code require modification
- Redis NetworkPolicies allow unrestricted cross-namespace access (must be scoped to monitoring namespace + exporter label)
- `PROMETHEUS_URL` env var is missing from autoscaler or invocation code's runtime environment
- Local dev (no `LOG_FORMAT`) stops logging human-readable to stderr, or the app reintroduces any `emptyDir`/`hostPath`/Downward API for logging

---

## Rollout Notes

- These changes can be deployed independently of the monitoring stack. The Prometheus annotations and `PROMETHEUS_URL` default are backward-compatible with the existing Prometheus deployment.
- `LOG_FORMAT=json` is harmless even if Fluent Bit isn't deployed yet -- the app just writes JSON to stdout, which is collected once the node agent is running.
- `prometheusUrl` is a permanent value pointing at Prometheus. Mimir only receives `remote_write` from Prometheus for long-term storage and is queried by Grafana -- application code always queries Prometheus directly.
