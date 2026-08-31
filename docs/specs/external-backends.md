# External execution backends

## Purpose and design decision

External execution is represented as a Chute with `execution_backend = "external"`.
It therefore uses the existing Chute/Cord discovery, API-key scope, usage,
pricing, and balance surfaces. It does not pretend to be hosted
compute: an external Chute has no source bundle, instances, miner attribution,
bounty, hardware score, autoscaling target, or TEE lifecycle.

The execution-specific data lives beside the Chute:

- An **account** owns one upstream origin, encrypted credentials, header
  templates, network policy, and funded-account governance limits.
- A **binding** connects exactly one external Chute to an account and stores one
  route for every Cord.
- A **route** describes how that Cord maps to an upstream HTTP or WebSocket
  endpoint, how requests and responses are transformed, how usage is extracted,
  and whether it is synchronous, streaming, asynchronous task, or realtime.
- An **operation** is the durable local record for one accepted attempt. It
  snapshots the route, transport version, pricing, identity, usage, result
  descriptors, polling lease, and settlement state needed for crash recovery.

Accounts, bindings, route configuration, and external Chutes are managed under
`/external/*`. Creating an account, binding, or external Chute is restricted to
the canonical `chutes` system user by exact user ID; holding the billing-admin
role alone does not grant provisioning authority. Existing billing-admin gates
remain on settlement remediation and incident controls. Ownership checks still
scope every managed object to its owning user. Public Chute and operation
responses do not expose credential references or upstream identities.

## Addressing and request contract

Cords remain the public API contract. Their input and output JSON Schemas are
descriptive metadata by default, matching existing hosted-Chute behavior. An
external route may explicitly opt into local validation; merely supplying a
schema never enables enforcement or even schema compilation. Opt-in validation
never retrieves remote `$ref` documents. A route must map one-to-one to the
Chute's Cord paths. The public method, path, and streaming flag must be unique.

The supported invocation forms are:

- a Chute slug hostname and the Cord's `public_api_path`;
- a compatible platform model endpoint, where model routing resolves the
  external Chute; or
- `/chutes/{chute_uuid}/{cord_path}` on the API domain.

The canonical API-domain form intentionally accepts a Chute UUID, not a name.
This prevents Chute names from shadowing management paths and allows names that
contain characters unsuitable for a path segment. API keys scoped to a Chute are
checked again after model resolution.

A declared Cord is authoritative for its exact Chute UUID, public path, and
method except that canonical `/hf_info` and `/evidence` remain reserved Chute
management endpoints. Those two exact paths are rejected case-insensitively when
Cords are declared. Hostname routing remains invocation-first, and nested paths
such as `/results/evidence` remain available in both forms.
Noncanonical undeclared paths retain the platform's native 404/405 behavior. A
canonical invocation-shaped request without credentials receives the same
generic 404 for a declared private Cord as for an unknown Cord, before any Chute
lookup. Credential-form requests follow the same authentication path whether the
Cord exists, and an authenticated unknown Cord then receives 404.

The ingress and process body ceilings default to 128 MiB
(`ingress.proxyBodySize: 128m` and `api.maxRequestBodyBytes: 134217728`). A
route's `request_config.max_request_bytes` may only lower the process ceiling.
The process stops reading a chunked request once the ceiling is exceeded; a large
`Content-Length` is rejected before the body is read.

## Invocation modes

### Synchronous

The API parses and transforms the request, performs one bounded upstream request,
scrubs the response, settles usage, and returns the result. JSON Schema validation
is performed only when its external route explicitly enables it.
The client connection is not forwarded upstream and only explicitly allowlisted
request headers are eligible to cross the boundary.

### Streaming

SSE or raw response bytes are relayed while usage checkpoints are persisted.
Every SSE event is independently scrubbed. Client disconnect stops delivery but
does not erase an accepted operation: a shielded finalizer settles the last known
usage according to `response_config.bill_partial_streams`. A raw stream requires
the Cord to declare `output_content_type`; a JSON output schema cannot be enforced
against arbitrary raw bytes.

### Asynchronous task

Task submission is asynchronous end to end. The API does not hold the client
connection open while media or other long-running work is produced.

1. The API parses the request, creates a `pending` local operation, and sends
   the submission.
2. It maps the upstream task identifier into private operation state and returns
   `202 Accepted`, a local operation ID, `Location`, and `status_url`.
3. A lease-based poller claims due operations with `FOR UPDATE SKIP LOCKED`, maps
   upstream state to the canonical status set, checkpoints usage, and schedules
   the next poll using bounded backoff and configured retry headers.
4. The client reads `GET /external/operations/{operation_id}`. Terminal results
   contain local artifact URLs, never the upstream task ID or artifact reference.

`Idempotency-Key` is supported for task submission. Reusing a key with the same
binding, user, and request fingerprint returns the existing operation; reuse for
a different request returns `409`.

The local status states are `pending`, `submitted`, `running`, `succeeded`,
`failed`, `cancelled`, and `expired`. The settlement states are `pending`,
`settled`, `not_billable`, `failed`, and `quarantined`. Quarantine is an
operator-held, unresolved billing state; it is neither retried automatically nor
treated as free spend.

The client can request cancellation at
`POST /external/operations/{operation_id}/cancel` only after submission has
attached the upstream identity. Cancellation of `pending` returns `409`, rather
than pulling dispatch recovery forward and orphaning work that may already have
been accepted. A task cancellation request is durable and the poller calls the
configured cancel endpoint. Running stream and realtime operations use the same
durable flag as a local cancellation signal; their live budget monitor observes
it within the configured interval, closes the upstream connection, and wakes any
attached downstream consumer. Disconnecting a status or artifact request is not
a cancellation request.

### Realtime

Realtime routes use a bounded WebSocket bridge. The bridge authenticates the
local client, constructs a separately authenticated upstream handshake, filters
query parameters, headers, and subprotocols, validates messages when configured,
and enforces idle, heartbeat, message-size, and maximum-session limits. Usage is
checkpointed during the session and observed spend/balance is rechecked while it
runs. Redirects during the WebSocket handshake are rejected.

## Accounts and authentication

An account creation payload contains:

```json
{
  "name": "production-media",
  "adapter": "generic-http",
  "base_url": "https://api.example.invalid",
  "credentials": {
    "primary": "secret-value-supplied-on-write"
  },
  "auth_header_templates": [
    {
      "name": "Authorization",
      "template": "Bearer {token}",
      "references": {"token": "primary"}
    }
  ],
  "connection_config": {
    "network": {
      "allowed_hosts": ["downloads.example.invalid"],
      "allowed_ports": [443]
    },
    "timeouts": {
      "connect": 10,
      "socket_read": 300,
      "total": 600
    }
  }
}
```

Credential values are write-only. Each value is stored as a Secret row with
`kind = "external_backend"`; hosted miner secret delivery structurally selects
only `kind = "chute"`. Header templates may contain simple named placeholders
whose references exactly match credential names. Control delimiters and
credential-like values in non-secret configuration are rejected.

HTTPS is mandatory. Plain HTTP requires both an account opt-in and the
process-wide `EXTERNAL_ALLOW_INSECURE_UPSTREAMS=true` switch; production should
leave it false. The account origin, route overrides, redirect targets, and
artifact origins are compiled into explicit host/port allowlists. DNS is resolved
through the guarded resolver and private, loopback, link-local, embedded IPv4,
NAT64, and rebinding targets are rejected at connection time.

Account transport or credential mutation is drain-only: it returns `409` while
an active operation or unexpired authenticated artifact can still need the old
configuration. Disabling an account prevents new admissions; already accepted
work continues from its snapshot so it can be polled and settled.

Billing admins have two incident controls, both requiring a nonempty audit
reason. `POST /external/accounts/{account_id}/operations/cancel` marks active
tasks only when their snapshotted route has a configured cancel endpoint, and
marks stream/realtime work for local connection teardown. A cancellable task
that is still `pending` receives a deferred flag without moving its dispatch
recovery deadline; once its remote identity is durably attached, polling becomes
immediately due. In-flight synchronous work and tasks without a cancel endpoint
remain on their normal completion/poll path and are returned in the
`not_cancellable` count rather than being turned into billable failures.

`POST /external/accounts/{account_id}/credentials/force-rotate` is the emergency
path for a credential incident. It accepts replacement values only for existing
credential names, disables the account and its bound Chutes before commit,
invalidates relays for every artifact accepted before the database cutoff time,
and applies the same safe cancellation rules to active work. Non-cancellable
tasks keep polling with the replacement credential. Neither endpoint changes
usage, pricing snapshots, immutable outbox events, or settlement decisions.
Actions are retained in bounded account/operation audit metadata with actor,
time, action, and reason; credential values remain write-only and are never
included in logs or audit records.

## Cord and route configuration

An external Cord is the same public OpenAPI-like contract used by a hosted
Chute. In addition to its Chute path, it declares the public selector and schemas:

```json
{
  "method": "POST",
  "path": "/generate",
  "function": "generate",
  "stream": false,
  "public_api_path": "/v1/generate",
  "public_api_method": "POST",
  "input_schema": {
    "type": "object",
    "properties": {
      "prompt": {"type": "string"},
      "duration": {"type": "number"},
      "resolution": {"type": "string"}
    },
    "required": ["prompt"]
  },
  "output_schema": {"type": "object"}
}
```

Cord schemas do not enumerate every parameter accepted by an upstream server and
are not runtime validators by default. This lets an OpenAI-compatible route pass
current and future vLLM/SGLang/provider extensions through without continually
updating a duplicate parameter model. Operators may leave the schemas empty or
use permissive schemas for documentation.

`standard_template` is currently a catalog/mega-routing compatibility tag, not a
preset expander. Creation therefore still supplies explicit one-to-one Cords and
routes. A Cord represents an endpoint and mode (for example chat buffered, chat
SSE, or embeddings), not every body parameter; the complete JSON body continues
upstream unless a configured transform changes it. Server-owned OpenAI-compatible
Cord/route presets are not part of this change.

Validation is external-route-specific and opt-in:

- `request_config.validate_input_schema: true` validates an HTTP request against
  the Cord's nonempty `input_schema` (or `minimal_input_schema`) before dispatch.
- `response_config.validate_output_schema: true` validates a buffered synchronous
  JSON response against the Cord's nonempty `output_schema`.
- `operation_config.submission_contract.enabled: true` independently validates a
  task submission response using its own schema/content type or the Cord fallback.
- `operation_config.realtime.validate_client_messages: true` validates inbound
  client JSON messages using `message_schema` or the Cord input-schema fallback.

All four controls default to `false`. Disabled schemas are retained as bounded
JSON metadata but are not compiled or semantically interpreted as JSON Schema,
traversed for pricing inference, or used at runtime. Eventual task results, SSE
output events,
realtime upstream messages, and raw output bytes do not support JSON Schema
enforcement. Platform-owned checks remain mandatory regardless of these flags:
request/body bounds and JSON framing, secret/header isolation, task ID/status and
artifact mappings, response scrubbing, safe content types, and billing usage
extraction.

Each route has these stable top-level fields:

```json
{
  "cord_path": "/generate",
  "upstream_resource_id": "resource-name",
  "operation_mode": "task",
  "protocol": "generic-json",
  "path_template": "/v1/jobs",
  "method": "POST",
  "request_config": {},
  "response_config": {},
  "operation_config": {},
  "capabilities": {}
}
```

`base_url` can override the account origin for a route, subject to the same
network policy. `path_template` is an absolute path, never a URL. The supported
body modes are JSON, raw, and none where appropriate; response modes are buffered,
SSE, and raw stream. Per-endpoint sizes, timeouts, redirects, static headers, and
allowed client/response headers are bounded and validated before save.

### Request and response mapping

`request_config` can allow selected client query/header fields, inject the
configured resource into a body/path/query location, and apply a payload
transform. A transform has `remove`, `inject`, and `rewrite` operations. A
mutation selects from `request`, `response`, `context`, `payload`, or an artifact
`item`, then assigns a target data path.

The common value-rule grammar is:

```json
{
  "source": "response",
  "path": "usage.output_seconds",
  "required": true,
  "cast": "number",
  "multiply": 1,
  "divide": 1,
  "add": 0
}
```

A rule may use one `path`, multiple `paths`, or a literal `value`; an optional
`default`; aggregation (`only`, `first`, `last`, `list`, `sum`, `count`, `min`,
`max`, `length`, or `join`); casting; a value `map`; and numeric transforms.
Configuration is strict: unknown fields, missing required data, non-finite usage,
and ambiguous `only` matches fail closed.

`response_config.usage` maps observations into a provider-independent usage
shape. Valid roots are:

- `requests`
- `tokens.<bucket>`
- `images.<bucket>`
- `input_media_seconds.<bucket>`
- `output_media_seconds.<bucket>`
- `characters.<bucket>`
- `counts.<bucket>`
- `tools.<bucket>`
- `dimensions.<path>`

For streams, `usage_mode` is `cumulative` (monotonic checkpoints) or `delta`
(additive observations; `default_requests` must be zero). Pricing is considered
complete only when every applicable charge component has an observed metric.

Public response rules can add removal paths/keys, rewrite safe values, and mark
artifact paths for replacement. The mandatory scrub boundary always removes
absolute upstream URLs and provider-identifying/credential-like keys, and runs
again after configured rewrites. Response headers are rebuilt from a narrow
allowlist; upstream cookies, authentication, tracing, server identity, and network
metadata do not reach the client.

### Task lifecycle mapping

A task route maps the submission identity and defines a poll endpoint. A reduced
example is:

```json
{
  "response_config": {
    "task": {
      "task_id": "data.id",
      "status": {
        "path": "data.state",
        "map": {
          "queued": "submitted",
          "processing": "running",
          "complete": "succeeded",
          "error": "failed",
          "cancelled": "cancelled"
        }
      }
    }
  },
  "operation_config": {
    "task_timeout_seconds": 3600,
    "poll": {
      "endpoint": {
        "path_template": "/v1/jobs/{task_id}",
        "method": "GET",
        "response_mode": "buffered"
      },
      "task": {
        "status": {
          "path": "data.state",
          "map": {
            "queued": "submitted",
            "processing": "running",
            "complete": "succeeded",
            "error": "failed",
            "cancelled": "cancelled"
          }
        },
        "artifacts": [{
          "items": "data.outputs[*]",
          "url": "url",
          "kind": {"value": "video"},
          "content_type": "content_type",
          "size_bytes": "size_bytes",
          "expires_at": "expires_at"
        }]
      },
      "usage": {
        "fields": {
          "output_media_seconds.video": "usage.output_seconds",
          "dimensions.resolution": "usage.resolution"
        }
      },
      "interval_seconds": 2,
      "backoff": {"multiplier": 1.5, "maximum_seconds": 30},
      "retry": {
        "statuses": [429, 502, 503, 504],
        "retry_after_headers": ["retry-after"],
        "max_attempts": 20
      },
      "billable_statuses": ["failed", "cancelled"]
    },
    "artifact": {
      "allowed_hosts": ["downloads.example.invalid"],
      "allowed_ports": [443],
      "authenticated": false,
      "max_bytes": 1073741824
    }
  }
}
```

A cancel endpoint uses the same endpoint/request/retry/usage grammar. Task routes
must explicitly declare `billable_statuses` for polling and cancellation terminal
outcomes, either at their shared task policy or on each call. Success is always
billable. This prevents a missing policy from silently deciding whether failed or
cancelled accepted work is charged.

## Pricing and billing

Pricing rules consume normalized usage, not provider-specific billing objects.
Supported metrics are `request`, `token`, `image`, `input_media_second`,
`output_media_second`, `character`, `count`, and `tool`. A rule selects an optional
bucket, divides quantity by `unit_size`, applies `exact`, `ceil`, `floor`, or
`nearest` rounding, clamps to `minimum_units`/`maximum_units`, and multiplies by
`unit_price`.

Rules may be scoped to Cord, public path, and method; effective-dated; and selected
by dimensions with `eq`, `ne`, `in`, `not_in`, `gt`, `gte`, `lt`, `lte`, or
`exists`. Ungrouped matching rules are additive. Mutually exclusive tiers use a
`match_group`:

```json
[
  {
    "id": "high-resolution",
    "metric": "output_media_second",
    "bucket": "video",
    "unit_price": "0.14",
    "unit_size": 1,
    "conditions": {"resolution": {"eq": "high"}},
    "match_group": "resolution-tier",
    "priority": 20
  },
  {
    "id": "standard-resolution",
    "metric": "output_media_second",
    "bucket": "video",
    "unit_price": "0.068",
    "unit_size": 1,
    "conditions": {"resolution": {"eq": "standard"}},
    "match_group": "resolution-tier",
    "priority": 10
  },
  {
    "id": "resolution-fallback",
    "metric": "output_media_second",
    "bucket": "video",
    "unit_price": "0.068",
    "unit_size": 1,
    "conditions": {},
    "match_group": "resolution-tier",
    "priority": 0,
    "fallback": true
  }
]
```

Every match group requires at least one conditional tier and exactly one
unconditional fallback. Tier priorities must be unique. All members must share
the metric, bucket, scope, and effective window. The highest-priority matching
tier wins; the fallback is used only when no tier matches. This prevents
overlapping `gte` tiers from double-billing. Route validation also checks that
pricing conditions and metrics can be supplied by the Cord/request or configured
usage mapping. The same complete rule parser is enforced at the shared
`PriceOverride` ORM write boundary, so hosted/admin writers cannot persist an
invalid match group through a different route.

The route and pricing rules are snapshotted when the operation is admitted.
Settlement uses that snapshot even if operators edit the Chute later. Every safely
observed positive component is additive, including a configured per-request
minimum. If another usage component is missing, the known positive subtotal is
charged once and its incomplete component count is retained in settlement audit
metadata. A complete, applied result whose observed usage is legitimately zero
is delivered through the same durable outbox as a zero charge. Invalid usage, a
wholly unapplied price, a negative amount, or an incomplete result with no safe
subtotal is retried only up to `EXTERNAL_SETTLEMENT_QUARANTINE_ATTEMPTS` (default
8), then held in `quarantined`. Legacy cached-input discounts are capped by the
observed charged input quantity, so an inconsistent provider cache count cannot
erase the request minimum or other components.

Settlement first writes an immutable usage event keyed by the operation ID to a
PostgreSQL outbox in the operation transaction. The operation remains `pending`
until one database transaction updates `usage_data`, optional app usage, and the
user balance, deletes the outbox row, and marks the operation `settled`. The
outbox's `next_attempt_at` is the authoritative delivery queue, independent of a
possibly stale operation presentation status. A separate pre-price sweep retries
terminal `pending` or `failed` rows which do not yet have an immutable event. Both
queues use lock-skipping claims and bounded crash-recovery deadlines. A crash
before commit rolls back the charge and acknowledgement together, so Redis queue
loss cannot turn a settled operation into lost revenue. The same dedicated worker,
reconcile interval, and settlement batch-size controls drive both queues; no second
billing daemon is required.

Billing admins resolve quarantined operations through
`POST /external/operations/{id}/settlement/retry` or
`POST /external/operations/{id}/settlement/write-off`. Retry may supply corrected,
normalized usage. A broken pricing snapshot may also be replaced only while the
operation is quarantined and has no immutable outbox event. That repair must name
the exact prior snapshot hash and an explicit customer-authorized maximum charge;
the accepted request context and billing identity cannot change, and the corrected
snapshot must completely price the persisted usage beneath the ceiling before it
is accepted. Operation responses expose this one-way snapshot hash, but not the
private snapshot context; the server carries the immutable context into a
replacement automatically. Settlement enforces the ceiling again, and a late
completion hook cannot replace operator-reviewed usage or billability. Both retry
forms require a reason, serialize against settlement and outbox delivery, and
append actor/time/action plus before/after hashes to the bounded audit history.

Known request cost is checked against the user's effective balance and per-request
ceiling at admission. Stream and realtime routes also receive a snapshotted
session exposure policy. `operation_config.session_budget` may lower
`max_exposure_usd`, raise `minimum_cost_per_second_usd`, or shorten
`check_interval_seconds` (at most five seconds). If no floor is supplied, one is
synthesized from the hard session duration and exposure ceiling. Admission
reserves the first interval and runtime checks use the greater of observed cost
and elapsed-time exposure, so missing token/usage events cannot turn a live
connection into unbounded zero-cost risk. Free invocations still contribute their
pay-as-you-go equivalent to funded-account spend limits.

Billing behavior is explicitly data-driven:

- `operation_config.billable_http_statuses` lists upstream HTTP errors known to
  represent accepted/billable work. They cannot also be configured for retry.
- `operation_config.bill_ambiguous_transport_errors` decides whether an ambiguous
  connection failure after dispatch is billable; the safe default is false.
- `response_config.bill_partial_streams` decides whether accepted partial stream
  or realtime work is billed; the default is true.
- task `billable_statuses` decide failed/cancelled/expired outcomes; success is
  always billable.

The API ingress does not retry upstream API pods. Ingress retries can replay an
operation after a pod has already caused funded upstream work; application-level
retry/failover owns the decision with idempotency and billability context. A
connection can still fail after the remote service accepted a request, so route
authors must set the ambiguity policy and use an upstream idempotency mechanism
where one exists.

## Results and artifact relay

The platform stores small sanitized descriptors and, only when explicitly
enabled, bounded inline task metadata. It does **not** download or persist result
blobs. An upstream artifact reference is retained privately and replaced for the
client with:

`GET|HEAD /external/operations/{operation_id}/artifacts/{artifact_index}`

On each authenticated request the API revalidates ownership, expiration, scheme,
host, port, DNS resolution, redirects, and optional artifact authentication, then
streams from the upstream cache. `Range` is supported. The relay strips upstream
identity headers and uses `private, no-store`.

Relay lifetime defaults to 24 hours and is configurable per response from 60
seconds through 30 days. An upstream expiration may shorten that lifetime but
cannot extend it. Relay amplification is bounded per user by requests per minute,
concurrent relays, bytes reserved per operation, a compact rolling 24-hour user
ledger, and the route's artifact `max_bytes`. An unknown
`Content-Length` reserves the route's maximum response size before streaming.
Reservations are reconciled to bytes actually delivered when a relay closes, so
an aborted or short download releases its unused allowance. A worker crash keeps
only an expiring conservative reservation until the next relay or rolling-window
expiry.

## Governance and circuit breaker

Process settings are hard ceilings. `connection_config.governance` accepts the
same snake-case fields and may only lower them:

| Environment variable | Account field | Default |
| --- | --- | ---: |
| `EXTERNAL_MAX_ACTIVE_TASKS_PER_USER` | `max_active_tasks_per_user` | 4 |
| `EXTERNAL_MAX_ACTIVE_TASKS_PER_ACCOUNT` | `max_active_tasks_per_account` | 256 |
| `EXTERNAL_MAX_ACTIVE_SYNC_REQUESTS_PER_USER` | `max_active_sync_requests_per_user` | 8 |
| `EXTERNAL_MAX_ACTIVE_SYNC_REQUESTS_PER_ACCOUNT` | `max_active_sync_requests_per_account` | 128 |
| `EXTERNAL_MAX_REALTIME_SESSIONS_PER_USER` | `max_realtime_sessions_per_user` | 2 |
| `EXTERNAL_MAX_REALTIME_SESSIONS_PER_ACCOUNT` | `max_realtime_sessions_per_account` | 64 |
| `EXTERNAL_MAX_STREAMS_PER_USER` | `max_streams_per_user` | 4 |
| `EXTERNAL_MAX_STREAMS_PER_ACCOUNT` | `max_streams_per_account` | 128 |
| `EXTERNAL_MAX_DAILY_OPERATIONS_PER_USER` | `max_daily_operations_per_user` | 1,000 |
| `EXTERNAL_MAX_DAILY_OPERATIONS_PER_ACCOUNT` | `max_daily_operations_per_account` | 100,000 |
| `EXTERNAL_MAX_DAILY_PAYGO_USD_PER_USER` | `max_daily_paygo_usd_per_user` | 25 |
| `EXTERNAL_MAX_DAILY_PAYGO_USD_PER_ACCOUNT` | `max_daily_paygo_usd_per_account` | 1,000 |
| `EXTERNAL_MAX_ESTIMATED_OPERATION_COST_USD` | `max_estimated_operation_cost_usd` | 50 |
| `EXTERNAL_ARTIFACT_REQUESTS_PER_MINUTE` | `artifact_requests_per_minute` | 60 |
| `EXTERNAL_ARTIFACT_MAX_CONCURRENT_PER_USER` | `artifact_max_concurrent_per_user` | 3 |
| `EXTERNAL_ARTIFACT_MAX_BYTES_PER_OPERATION` | `artifact_max_bytes_per_operation` | 10 GiB |
| `EXTERNAL_ARTIFACT_MAX_DAILY_BYTES_PER_USER` | `artifact_max_daily_bytes_per_user` | 50 GiB |

Admission count-and-insert is serialized against trigger-maintained user/account
state rows, locked after the authoritative user balance in the same order as
settlement. Active counts and all-age prepaid outstanding exposure are
constant-size per scope; operation count and both settled and unresolved daily
provider spend use compact minute buckets. Admission therefore reads at most a
bounded 24-hour rollup rather than scanning a day's operation volume, while
replicas still cannot race past task/session/stream or spend limits. Updates that
touch only leases, heartbeats, or presentation fields do not invoke the governance
trigger. Multi-operation writers lock operation rows first, then distinct users
in sorted order, then accounts. Periodic bounded reconciliation re-derives state
and recreates a missing active/unresolved scope row, while database-clock pruning
removes expired buckets. Quarantined settlements remain all-age unresolved funded
exposure until an operator retries or writes them off, but age out of the rolling
daily spend window after 24 hours.

Repeated authentication failures or service/transport failures open a Redis-backed
account admission circuit. Defaults are 3 authentication failures, 10 service
failures, and a 300-second cooldown (`EXTERNAL_CIRCUIT_AUTH_FAILURE_THRESHOLD`,
`EXTERNAL_CIRCUIT_SERVICE_FAILURE_THRESHOLD`, and
`EXTERNAL_CIRCUIT_COOLDOWN_SECONDS`). Healthy responses reset it. The circuit
blocks new admissions but never abandons polling for accepted tasks. Redis
failures leave admission fail-open, but are read through the raw client so
`external_circuit_events_total{reason="backend",action="error"}` records the
degraded protection instead of silently flattening the error.

## Operations, metrics, and retention

The poller can run inside each API process or as a dedicated deployment. Database
leases and token-guarded heartbeats make either topology safe across replicas.
For production, enable `externalOperationWorker.enabled`; the chart automatically
sets `EXTERNAL_POLLER_ENABLED=false` on API pods and true on the worker. The worker
exposes `/health`, `/metrics`, and `/_metrics` on its configured health port.
Health is ready only while the poller task is alive and both PostgreSQL and Redis
respond. The worker receives the existing wallet/PG encryption keys needed to
decrypt retained account credentials; it does not receive the unrelated server
passphrase key. Helm rendering fails if both the dedicated worker and in-API
polling are disabled, preventing a configuration which silently stops polling,
settlement recovery, and retention.

Canonical local task and artifact URLs use `BASE_DOMAIN`, whose application and
chart default is `chutes.ai`. Set the chart's `baseDomain` value for custom
deployments; it configures the wildcard API ingress and injects the identical
value into API and dedicated-worker pods.

Poller tuning variables are:

- `EXTERNAL_POLLER_BATCH_SIZE` (16)
- `EXTERNAL_POLLER_CONCURRENCY` (8)
- `EXTERNAL_POLLER_LEASE_SECONDS` (60)
- `EXTERNAL_POLLER_IDLE_SECONDS` (1)
- `EXTERNAL_POLLER_SHUTDOWN_TIMEOUT_SECONDS` (30)
- `EXTERNAL_SETTLEMENT_RECONCILE_INTERVAL_SECONDS` (5)
- `EXTERNAL_SETTLEMENT_BATCH_SIZE` (64)
- `EXTERNAL_SETTLEMENT_QUARANTINE_ATTEMPTS` (8; valid range 1–100)
- `EXTERNAL_OPERATION_MAINTENANCE_INTERVAL_SECONDS` (60)
- `EXTERNAL_OPERATION_RETENTION_DAYS` (90)
- `EXTERNAL_OPERATION_RETENTION_BATCH_SIZE` (1,000)

Resolved terminal rows are collected only after the retention cutoff and after
their artifact lifetime has also expired. Active operations and unresolved
settlements are never collected. Each maintenance pass locks and removes at most
the configured retention batch, allowing multiple workers to cooperate without a
large delete transaction. Deletes use per-row savepoints so one unexpectedly
referenced historical row cannot halt collection of the rest of the batch.
Idempotency keys disappear with their retained operation row, after the same safe
window.

External-specific Prometheus series are:

- `external_upstream_requests_total` and
  `external_upstream_request_duration_seconds`
- `external_operation_admissions_total` and
  `external_admission_rejections_total`
- `external_operation_queue_depth` and
  `external_operation_oldest_poll_lag_seconds`
- `external_settlement_attempts_total` and `external_settlement_backlog`
- `external_billing_delivery_attempts_total`
- `external_circuit_events_total`
- `external_artifact_relay_requests_total` and
  `external_artifact_relay_bytes_total`
- `external_operation_retention_deletions_total`
- `external_governance_bucket_deletions_total`

At minimum, alert on a nonzero failed-settlement backlog, a sustained pending
settlement backlog, any quarantined settlement, oldest poll lag greater than the
expected poll interval plus lease, authentication-circuit opens, a high upstream
5xx/transport-error ratio, and artifact-limit rejections. Queue depth without
corresponding poll throughput is a worker availability signal.

## Security and trust boundaries

- Client `Authorization`, cookies, forwarded IP, tracing, connection, and
  proxy-authentication headers are never passed through. Authentication is built
  only from operator-owned encrypted credentials and validated templates.
- Response headers and every JSON/SSE value cross a mandatory scrub boundary;
  opaque SSE data is rejected (the protocol's fixed `[DONE]` sentinel is allowed).
  Absolute remote URLs are private unless replaced with a local artifact URL.
- Outbound origins are operator-pinned. Redirects are bounded, revalidated at
  every hop, and strip secrets when crossing an explicitly allowed origin.
- The guarded resolver validates the actual resolved address and does not trust a
  prior hostname check. JSON Schema validation cannot perform network retrieval.
- Request, response, SSE event, WebSocket message/session, inline result, mapping
  depth/node, artifact, retry, timeout, and redirect limits are all bounded.
- Compatibility mega endpoints parse and enforce their JSON request guardrails
  independently of the caller-supplied `Content-Type`; relabeling JSON as text
  cannot bypass file rejection or restricted-field stripping.
- External operations write usage only. They never create miner invocation
  attribution or enter miner sync, scoring, weights, watchtower, bounties, or
  autoscaling.
- Account identifiers, remote task IDs, credential references, upstream URLs,
  and response identity headers remain private even when a route author adds
  response rewrite rules.

Raw HTTP bodies and explicitly enabled opaque WebSocket binary/text frames cannot
be content-scrubbed without changing their protocol. Treat those route flags as a
provider-trust opt-in and use them only for a documented media/wire format; the
default for every opaque realtime direction is deny.

## Database migration and rollout

The API image includes `dbmate`, all SQL migrations, and the dedicated worker
entrypoint. The Helm chart's enabled-by-default `databaseMigrations` Job runs as a
`pre-install,pre-upgrade` hook using the `postgres-secret` `url` key. It converts
the SQLAlchemy `postgresql+asyncpg` scheme for dbmate, runs every pending migration
without writing a schema dump, and blocks the rollout on failure. Failed hook Jobs
are retained for inspection until their configured TTL; successful Jobs are
removed immediately.

Recommended rollout:

1. Render and review the chart with the environment overlay. Confirm the API
   ingress/body limits match, funded-account ceilings are intentionally low, and
   the migration Job can reach the database.
2. Upgrade with no external account enabled. Verify the migration hook, API
   health, and that external tables and Secret `kind` backfill are present.
3. Enable `externalOperationWorker.enabled` and verify API pods report
   `EXTERNAL_POLLER_ENABLED=false`, the worker is ready, and its metrics are
   scraped.
4. Create one disabled account, one private external Chute, its Cords/routes, and
   exhaustive pricing rules. Validate configuration before enabling either the
   account or binding.
5. Exercise sync/stream/task/realtime modes as applicable, including 429,
   ambiguous disconnect, cancellation, partial stream, failed settlement retry,
   Range relay, and expiry cases with a low funded-account cap.
6. Alert on settlement and polling health, then gradually raise account-specific
   limits without exceeding process ceilings and make the Chute public.

Rollback must preserve the database while external Chutes or retained operations
exist. Do not run a down migration merely because application pods are rolled
back; disable new admissions and let accepted operations finish and settle first.
