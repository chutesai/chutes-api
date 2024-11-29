#!/bin/bash
set -euo pipefail

# install repos, make sure they're up to date
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add hashicorp https://helm.releases.hashicorp.com
helm repo update

# if you need it, setup log aggregation, metrics, and observability
helm upgrade --install observability grafana/loki-stack --version 2.10.2 --values values_observability.yaml

# if you need it, install postgres
helm upgrade --install postgres oci://registry-1.docker.io/bitnamicharts/postgresql --version 16.2.3 --values values_postgres.yaml

#if you need it,  install redis
helm upgrade --install redis oci://registry-1.docker.io/bitnamicharts/redis --version 20.3.0 --values values_redis.yaml

#if you need it,  install minio
helm upgrade --install minio oci://registry-1.docker.io/bitnamicharts/minio --version 14.8.5 --values values_minio.yaml

#if you need it,  install vault
helm upgrade --install vault oci://registry-1.docker.io/bitnamicharts/vault --version 1.4.30 --values values_vault.yaml
