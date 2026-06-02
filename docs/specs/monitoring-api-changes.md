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
  - `api/main.py` -- loguru dual-sink configuration
  - Various deployment templates -- `emptyDir` volume for structured logs

---

## Design Decisions

- **Remove Datadog entirely** rather than keeping it as a disabled option. The `datadog_enabled` flag, all `tags.datadoghq.com/*` labels, `admission.datadoghq.com/*` annotations, and `DD_LOGS_INJECTION` env vars are all removed.
- **Prometheus annotations always present** on API pods (no longer gated behind a flag).
- **Dual-sink loguru**: human-readable to stdout (for `kubectl logs`), JSON to a file on an `emptyDir` volume (for Fluent Bit). No changes to existing log call syntax.
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

### 5. Dual-sink loguru configuration

**`api/main.py`** (at app startup, in or near the `lifespan` function):
- Keep the default loguru stderr sink (human-readable, for `kubectl logs`)
- Add a JSON file sink:

```python
import sys
from loguru import logger

log_path = os.environ.get("STRUCTURED_LOG_PATH", "/var/log/app/structured.log")
if os.path.isdir(os.path.dirname(log_path)):
    logger.add(
        log_path,
        serialize=True,
        rotation="100 MB",
        retention="1 day",
        compression="gz",
    )
```

The sink only activates if the directory exists (i.e., the `emptyDir` volume is mounted). This makes it safe for local dev where the volume isn't present.

**Deployment templates** (all deployments/cronjobs that should emit structured logs):
- Add `emptyDir` volume:

```yaml
volumes:
  - name: structured-logs
    emptyDir: {}
```

- Add volumeMount to the container:

```yaml
volumeMounts:
  - name: structured-logs
    mountPath: /var/log/app
```

**Which templates need this** (at minimum, the high-traffic services):
- `api-deployment.yaml`
- `socket-deployment.yaml`
- `event-socket-deployment.yaml`
- `forge-deployment.yaml`
- `watchtower-deployment.yaml`
- `payment-watcher-deployment.yaml`
- `usage-tracker-deployment.yaml`
- `autoscaler-cronjob.yaml`

Lower-priority (can add later):
- `graval-worker-deployment.yaml`
- `autostaker-deployment.yaml`
- `bt-tx-tracker-deployment.yaml`
- Other cronjobs

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
5. Loguru dual-sink is configured: stdout remains human-readable, JSON written to `/var/log/app/structured.log` when the volume is mounted.
6. `emptyDir` volumes are mounted on all primary deployment templates for structured log output.
7. Registry-proxy nginx access log directory has an `emptyDir` volume mount.
8. All existing functionality is unchanged -- no log call modifications, no behavior changes.

---

## Constraints

- No new Python dependencies (loguru `serialize=True` and `rotation` are built-in)
- No changes to existing log call syntax (all `logger.info(f"...")` calls remain as-is)
- `PROMETHEUS_URL` default must use the full cross-namespace form (`http://prometheus-server.monitoring.svc.cluster.local`) since Prometheus lives in the `monitoring` namespace
- NetworkPolicy changes must be scoped (namespace + pod label selector) -- do not open blanket cross-namespace access
- Dual-sink loguru must fail gracefully if the volume isn't mounted (local dev without Docker volumes)

---

## Output Format

1. Modified `charts/values.yaml` -- remove `datadog_enabled`, add `prometheusUrl`
2. Modified `charts/templates/api-deployment.yaml` -- remove Datadog, add Prometheus annotations
3. Modified `charts/templates/usage-tracker-deployment.yaml` -- remove Datadog env var
4. Modified `charts/templates/redis-np.yaml` -- add monitoring namespace ingress
5. Modified `charts/templates/cm-redis-np.yaml` -- add monitoring namespace ingress
6. Modified `charts/templates/quota-redis-np.yaml` -- add monitoring namespace ingress
7. Modified `charts/templates/_helpers.tpl` -- add `PROMETHEUS_URL` to `commonEnv`
8. Modified `api/main.py` -- add loguru JSON file sink
9. Modified deployment templates (8+) -- add `emptyDir` volume + volumeMount for structured logs
10. Modified `charts/templates/registry-proxy-deployment.yaml` -- add nginx log `emptyDir` volume

---

## Failure Conditions

- Any Datadog reference remains in the codebase after changes
- Prometheus annotations are missing or placed as `labels` instead of `annotations`
- `kubectl logs <api-pod>` output changes from human-readable to JSON
- Existing log calls in Python code require modification
- Redis NetworkPolicies allow unrestricted cross-namespace access (must be scoped to monitoring namespace + exporter label)
- `PROMETHEUS_URL` env var is missing from autoscaler or invocation code's runtime environment
- App crashes on startup when structured log volume is not mounted (local dev)

---

## Rollout Notes

- These changes can be deployed independently of the monitoring stack. The Prometheus annotations and `PROMETHEUS_URL` default are backward-compatible with the existing Prometheus deployment.
- The `emptyDir` volumes are harmless even if Fluent Bit isn't deployed yet -- loguru writes to the file, it just doesn't get collected until Fluent Bit is running.
- `prometheusUrl` is a permanent value pointing at Prometheus. Mimir only receives `remote_write` from Prometheus for long-term storage and is queried by Grafana -- application code always queries Prometheus directly.
