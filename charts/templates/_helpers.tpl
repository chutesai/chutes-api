{{- define "api.labels" -}}
app.kubernetes.io/name: api
redis-access: "true"
{{- end }}

{{- define "socket.labels" -}}
app.kubernetes.io/name: socket
redis-access: "true"
{{- end }}

{{- define "paymentWatcher.labels" -}}
app.kubernetes.io/name: payment-watcher
redis-access: "true"
{{- end }}

{{- define "graval.labels" -}}
app.kubernetes.io/name: graval
redis-access: "true"
{{- end }}

{{- define "gravalWorker.labels" -}}
app.kubernetes.io/name: graval-worker
redis-access: "true"
{{- end }}

{{- define "forge.labels" -}}
app.kubernetes.io/name: forge
redis-access: "true"
{{- end }}

{{- define "metasync.labels" -}}
app.kubernetes.io/name: metasync
redis-access: "true"
{{- end }}

{{- define "redis.labels" -}}
app.kubernetes.io/name: redis
{{- end }}

{{- define "registry.labels" -}}
app.kubernetes.io/name: registry
{{- end }}

{{- define "registryProxy.labels" -}}
app.kubernetes.io/name: proxy
{{- end }}

{{- define "chutes.sensitiveEnv" -}}
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
