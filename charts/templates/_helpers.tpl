{{- define "api.labels" -}}
app.kubernetes.io/name: api
redis-access: "true"
db-access: "true"
{{- end }}

{{- define "socket.labels" -}}
app.kubernetes.io/name: socket
redis-access: "true"
db-access: "true"
{{- end }}

{{- define "eventSocket.labels" -}}
app.kubernetes.io/name: event-socket
redis-access: "true"
db-access: "true"
{{- end }}

{{- define "paymentWatcher.labels" -}}
app.kubernetes.io/name: payment-watcher
redis-access: "true"
db-access: "true"
{{- end }}

{{- define "graval.labels" -}}
app.kubernetes.io/name: graval
redis-access: "true"
db-access: "true"
{{- end }}

{{- define "gravaldb.labels" -}}
app.kubernetes.io/name: gravaldb
{{- end }}

{{- define "gravalWorker.labels" -}}
app.kubernetes.io/name: graval-worker
redis-access: "true"
db-access: "true"
{{- end }}

{{- define "autostaker.labels" -}}
app.kubernetes.io/name: autostaker
redis-access: "true"
db-access: "true"
{{- end }}

{{- define "forge.labels" -}}
app.kubernetes.io/name: forge
redis-access: "true"
db-access: "true"
{{- end }}

{{- define "metasync.labels" -}}
app.kubernetes.io/name: metasync
redis-access: "true"
db-access: "true"
{{- end }}

{{- define "weightsetter.labels" -}}
app.kubernetes.io/name: weightsetter
redis-access: "true"
db-access: "true"
{{- end }}

{{- define "redis.labels" -}}
app.kubernetes.io/name: redis
{{- end }}

{{- define "cmRedis.labels" -}}
app.kubernetes.io/name: cm-redis
{{- end }}

{{- define "memcached.labels" -}}
app.kubernetes.io/name: memcached
{{- end }}

{{- define "registry.labels" -}}
app.kubernetes.io/name: registry
{{- end }}

{{- define "pgproxy.labels" -}}
app.kubernetes.io/name: pgproxy
{{- end }}

{{- define "registryProxy.labels" -}}
app.kubernetes.io/name: registry-proxy
{{- end }}

{{- define "chutes.sensitiveEnv" -}}
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
{{- end }}

{{- define "chutes.commonEnv" -}}
- name: CHUTES_VERSION
  value: {{ .Values.chutes_version }}
- name: GRAVAL_URL
  value: http://graval:{{ .Values.graval.service.port }}
- name: VALIDATOR_SS58
  valueFrom:
    secretKeyRef:
      name: validator-credentials
      key: ss58
- name: REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: redis-secret
      key: password
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
- name: REDIS_URL
  valueFrom:
    secretKeyRef:
      name: redis-secret
      key: url
- name: AWS_ACCESS_KEY_ID
  valueFrom:
    secretKeyRef:
      name: s3-credentials
      key: access-key-id
- name: AWS_SECRET_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: s3-credentials
      key: secret-access-key
- name: AWS_ENDPOINT_URL
  valueFrom:
    secretKeyRef:
      name: s3-credentials
      key: endpoint-url
- name: AWS_REGION
  valueFrom:
    secretKeyRef:
      name: s3-credentials
      key: aws-region
- name: STORAGE_BUCKET
  valueFrom:
    secretKeyRef:
      name: s3-credentials
      key: bucket
- name: REGISTRY_PASSWORD
  valueFrom:
    secretKeyRef:
      name: registry-secret
      key: password
- name: REGISTRY_INSECURE
  value: "true"
{{- end -}}
