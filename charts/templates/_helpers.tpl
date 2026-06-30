{{- define "api.labels" -}}
app.kubernetes.io/name: api
{{- end }}

{{- define "socket.labels" -}}
app.kubernetes.io/name: socket
{{- end }}

{{- define "eventSocket.labels" -}}
app.kubernetes.io/name: event-socket
{{- end }}

{{- define "paymentWatcher.labels" -}}
app.kubernetes.io/name: payment-watcher
{{- end }}

{{- define "btTxTracker.labels" -}}
app.kubernetes.io/name: bt-tx-tracker
{{- end }}

{{- define "usageTracker.labels" -}}
app.kubernetes.io/name: usage-tracker
{{- end }}

{{- define "balanceRefresher.labels" -}}
app.kubernetes.io/name: balance-refresher
{{- end }}

{{- define "logProber.labels" -}}
app.kubernetes.io/name: log-prober
{{- end }}

{{- define "serverHealthProber.labels" -}}
app.kubernetes.io/name: server-health-prober
{{- end }}

{{- define "graval.labels" -}}
app.kubernetes.io/name: graval
{{- end }}

{{- define "gravaldb.labels" -}}
app.kubernetes.io/name: gravaldb
{{- end }}

{{- define "gravalWorker.labels" -}}
app.kubernetes.io/name: graval-worker
{{- end }}

{{- define "watchtower.labels" -}}
app.kubernetes.io/name: watchtower
{{- end }}

{{- define "cacher.labels" -}}
app.kubernetes.io/name: cacher
{{- end }}

{{- define "chuteAutoscaler.labels" -}}
app.kubernetes.io/name: chute-autoscaler
{{- end }}

{{- define "chuteAutoscalerDryrun.labels" -}}
app.kubernetes.io/name: chute-autoscaler-dryrun
{{- end }}

{{- define "autostaker.labels" -}}
app.kubernetes.io/name: autostaker
{{- end }}

{{- define "forge.labels" -}}
app.kubernetes.io/name: forge
{{- end }}

{{- define "remoteForge.labels" -}}
app.kubernetes.io/name: remote-forge
redis-access: "true"
db-access: "true"
{{- end }}

{{- define "metasync.labels" -}}
app.kubernetes.io/name: metagraph-syncer
{{- end }}

{{- define "weightsetter.labels" -}}
app.kubernetes.io/name: weightsetter
{{- end }}

{{- define "cmRedis.labels" -}}
app.kubernetes.io/name: cm-redis
{{- end }}

{{- define "quotaRedis.labels" -}}
app.kubernetes.io/name: quota-redis
{{- end }}


{{- define "registry.labels" -}}
app.kubernetes.io/name: registry
{{- end }}

{{- define "registryProxy.labels" -}}
app.kubernetes.io/name: registry-proxy
{{- end }}

{{- define "attestationProxy.labels" -}}
app.kubernetes.io/name: attestation-proxy
{{- end }}

{{- define "claudeProxy.labels" -}}
app.kubernetes.io/name: claude-proxy
{{- end }}

{{- define "pgRouter.labels" -}}
app.kubernetes.io/name: pg-router
{{- end }}

{{- define "responsesProxy.labels" -}}
app.kubernetes.io/name: responses-proxy
{{- end }}

{{- define "connProber.labels" -}}
app.kubernetes.io/name: conn-prober
{{- end }}

{{- define "auditExporter.labels" -}}
app.kubernetes.io/name: audit-exporter
{{- end }}

{{- define "failedChuteCleanup.labels" -}}
app.kubernetes.io/name: failed-chute-cleanup
{{- end }}

{{/*
NetworkPolicy access label helpers.
Include chutes.dbAccess and/or chutes.redisAccess in the spec.selector.matchLabels
and spec.template.metadata.labels of any Deployment/CronJob that needs those policies.
*/}}
{{- define "chutes.dbAccess" -}}
db-access: "true"
{{- end }}

{{- define "chutes.redisAccess" -}}
redis-access: "true"
{{- end }}

{{- define "chutes.sensitiveEnv" -}}
- name: CLLMV_X25519_PRIVATE_KEY
  valueFrom:
    secretKeyRef:
      key: key
      name: cllmv-pkey
- name: PS_OP
  valueFrom:
    secretKeyRef:
      key: key
      name: inspecto
- name: HCAPTCHA_SITEKEY
  valueFrom:
   secretKeyRef:
     key: sitekey
     name: hcaptcha-config
- name: HCAPTCHA_SECRET
  valueFrom:
   secretKeyRef:
     key: secret
     name: hcaptcha-config
- name: CFSV_OP
  valueFrom:
   secretKeyRef:
     key: key
     name: cfsvop
- name: LAUNCH_CONFIG_KEY
  valueFrom:
    secretKeyRef:
      name: launch-config
      key: key
- name: ENVDUMP_UNLOCK
  valueFrom:
    secretKeyRef:
      key: token
      name: envdump
- name: CODECHECK_KEY
  valueFrom:
    secretKeyRef:
      name: codecheck-key
      key: key
- name: IP_CHECK_SALT
  valueFrom:
    secretKeyRef:
      name: ip-check-salt
      key: salt
- name: VALIDATOR_SEED
  valueFrom:
    secretKeyRef:
      name: validator-credentials
      key: seed
- name: WALLET_KEY
  valueFrom:
    secretKeyRef:
      name: wallet-secret
      key: wallet-key
- name: PG_ENCRYPTION_KEY
  valueFrom:
    secretKeyRef:
      name: wallet-secret
      key: pg-key
- name: LUKS_PASSPHRASES
  valueFrom:
    secretKeyRef:
      name: luks-passphrases
      key: passphrases
{{- end }}

{{- define "chutes.commonEnv" -}}
- name: GRAVAL_URL
  value: https://graval.chutes.ai
- name: REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: redis-secret
      key: password
- name: REDIS_HOST
  value: {{ .Values.redis.host | quote }}
- name: REDIS_PORT
  value: {{ .Values.redis.port | quote }}
{{- if .Values.redis.cacertSecret }}
- name: REDIS_CACERT
  value: "/etc/redis-cacert/cacert.pem"
{{- end }}
- name: POSTGRES_PASSWORD
  valueFrom:
    secretKeyRef:
      name: postgres-secret
      key: password
- name: POSTGRESQL
  valueFrom:
    secretKeyRef:
      name: postgres-secret
      key: url
- name: POSTGRESQL_RO
  valueFrom:
    secretKeyRef:
      name: postgres-secret
      key: readonly_url
{{- if .Values.s3ProxyUrl }}
- name: S3_PROXY_URL
  value: {{ .Values.s3ProxyUrl | quote }}
{{- end }}
- name: AWS_ACCESS_KEY_ID
  valueFrom:
    secretKeyRef:
      name: {{ .Values.s3SecretName | default "s3-credentials" }}
      key: access-key-id
- name: AWS_SECRET_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Values.s3SecretName | default "s3-credentials" }}
      key: secret-access-key
- name: AWS_ENDPOINT_URL
  valueFrom:
    secretKeyRef:
      name: {{ .Values.s3SecretName | default "s3-credentials" }}
      key: endpoint-url
- name: AWS_REGION
  valueFrom:
    secretKeyRef:
      name: {{ .Values.s3SecretName | default "s3-credentials" }}
      key: aws-region
- name: STORAGE_BUCKET
  valueFrom:
    secretKeyRef:
      name: {{ .Values.s3SecretName | default "s3-credentials" }}
      key: bucket
- name: REGISTRY_PASSWORD
  valueFrom:
    secretKeyRef:
      name: registry-secret
      key: password
- name: REGISTRY_INSECURE
  value: "true"
{{- end -}}

{{/*
Logging env for long-running services. LOG_FORMAT=json switches loguru to emit one
JSON object per line on stdout (see api/log.py configure_structured_logging); the node
Fluent Bit DaemonSet collects and enriches that stdout. Unset = human-readable stderr.
*/}}
{{- define "chutes.loggingEnv" -}}
- name: LOG_FORMAT
  value: {{ .Values.logFormat | default "json" | quote }}
{{- if .Values.logLevel }}
- name: LOG_LEVEL
  value: {{ .Values.logLevel | quote }}
{{- end }}
{{- end -}}

{{/*
Volume definition for the managed Redis TLS CA certificate.
Only emits when redis.cacertSecret is set.
*/}}
{{- define "chutes.redisCacertVolume" -}}
{{- if .Values.redis.cacertSecret }}
- name: redis-cacert
  secret:
    secretName: {{ .Values.redis.cacertSecret }}
    defaultMode: {{ .Values.redis.cacertMode | default 0444 }}
    items:
    - key: ca
      path: cacert.pem
{{- end }}
{{- end }}

{{/*
VolumeMount for the managed Redis TLS CA certificate.
Only emits when redis.cacertSecret is set.
*/}}
{{- define "chutes.redisCacertMount" -}}
{{- if .Values.redis.cacertSecret }}
- mountPath: /etc/redis-cacert
  name: redis-cacert
  readOnly: true
{{- end }}
{{- end }}

{{/*
Pod-level security context (seccompProfile).
*/}}
{{- define "chutes.podSecurityContext" -}}
seccompProfile:
  type: RuntimeDefault
{{- end }}

{{/*
Container-level security context (drops NET_RAW capability).
*/}}
{{- define "chutes.containerSecurityContext" -}}
capabilities:
  drop:
  - NET_RAW
{{- end }}

{{/*
Build a fully-qualified image reference.
Usage: {{ include "chutes.image" (list . .Values.api.image) | quote }}
Prepends .Values.imageRegistry (with trailing slash stripped) when set.
Public Docker Hub images (redis, etc.) are passed directly
via their own values keys without going through this helper.
*/}}
{{- define "chutes.image" -}}
{{- $root := index . 0 -}}
{{- $img  := index . 1 -}}
{{- $reg  := $root.Values.imageRegistry | default "" | trimSuffix "/" -}}
{{- if $reg -}}
{{- printf "%s/%s" $reg $img -}}
{{- else -}}
{{- $img -}}
{{- end -}}
{{- end }}

{{/*
Default node tolerations for amd64 architecture.
*/}}
{{- define "chutes.defaultTolerations" -}}
- effect: NoSchedule
  key: kubernetes.io/arch
  operator: Equal
  value: amd64
{{- end }}

{{/*
CronJob container resources. CronJobs in prod carry only an ephemeral-storage
limit (cpu/memory are requests-only); render that shape regardless of any
cpu/memory limits present in merged values.
*/}}
{{- define "chutes.cronjobResources" -}}
{{- $lim := .limits | default dict -}}
requests:
  {{- toYaml .requests | nindent 2 }}
limits:
  ephemeral-storage: {{ index $lim "ephemeral-storage" | default "1Gi" | quote }}
{{- end }}

{{/*
Minimal env block used by most CronJobs:
Redis connection + Validator SS58 + Postgres credentials.
*/}}
{{- define "chutes.cronjobEnv" -}}
- name: VALIDATOR_SS58
{{- if .ss58Literal }}
  value: {{ .ss58Literal | quote }}
{{- else }}
  valueFrom:
    secretKeyRef:
      name: validator-credentials
      key: ss58
{{- end }}
- name: REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: redis-secret
      key: password
- name: REDIS_HOST
  value: {{ .Values.redis.host | quote }}
- name: REDIS_PORT
  value: {{ .Values.redis.port | quote }}
{{- if .Values.redis.cacertSecret }}
- name: REDIS_CACERT
  value: "/etc/redis-cacert/cacert.pem"
{{- end }}
- name: POSTGRES_PASSWORD
  valueFrom:
    secretKeyRef:
      name: postgres-secret
      key: password
- name: POSTGRESQL
  valueFrom:
    secretKeyRef:
      name: postgres-secret
      key: url
- name: POSTGRESQL_RO
  valueFrom:
    secretKeyRef:
      name: postgres-secret
      key: readonly_url
{{- end }}

{{/*
CronJob Job/pod retention. Resolution order: per-job override -> global .Values.cronjobDefaults -> hard default.
Call with: (dict "root" . "job" .Values.<jobKey>)
*/}}
{{- define "chutes.successfulJobsHistoryLimit" -}}
{{- $job := .job | default dict -}}
{{- $def := .root.Values.cronjobDefaults | default dict -}}
{{- $job.successfulJobsHistoryLimit | default $def.successfulJobsHistoryLimit | default 3 -}}
{{- end }}

{{- define "chutes.failedJobsHistoryLimit" -}}
{{- $job := .job | default dict -}}
{{- $def := .root.Values.cronjobDefaults | default dict -}}
{{- $job.failedJobsHistoryLimit | default $def.failedJobsHistoryLimit | default 3 -}}
{{- end }}

{{- define "chutes.ttlSecondsAfterFinished" -}}
{{- $job := .job | default dict -}}
{{- $def := .root.Values.cronjobDefaults | default dict -}}
{{- $job.ttlSecondsAfterFinished | default $def.ttlSecondsAfterFinished | default 3600 -}}
{{- end }}
