#!/bin/bash
set -euo pipefail

# install repos, make sure they're up to date
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# setup log aggregation, metrics, and observability
helm upgrade --install observability grafana/loki-stack --set grafana.enabled=true,prometheus.enabled=true

# install postgres
helm upgrade --install postgres oci://registry-1.docker.io/bitnamicharts/postgresql --values values_postgres.yaml

# install redis
helm upgrade --install redis oci://registry-1.docker.io/bitnamicharts/redis --values values_redis.yaml

# install minio
helm upgrade --install redis oci://registry-1.docker.io/bitnamicharts/minio --values values_minio.yaml

# install registry
