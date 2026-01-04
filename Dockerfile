###
# BASE
###
FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies and Python 3.12
RUN apt-get update && apt-get -y upgrade && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    curl \
    wget \
    jq \
    vim \
    git \
    gcc \
    g++ \
    make \
    postgresql-client \
    netcat-traditional \
    dnsutils \
    iputils-ping \
    procps \
    ca-certificates \
    gnupg \
    lsb-release \
    uidmap \
    && rm -rf /var/lib/apt/lists/*

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && mv kubectl /usr/local/bin/ \
    && chmod 755 /usr/local/bin/kubectl

# Set Python 3.12 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# NN verification lib.
ADD data/chutes-nnverify.so /usr/local/lib/chutes-nnverify.so
RUN chmod 755 /usr/local/lib/chutes-nnverify.so

###
# FORGE
###
FROM base AS forge

# Install buildah components.
RUN apt-get update && apt-get install -y \
    buildah \
    podman \
    skopeo \
    fuse-overlayfs \
    slirp4netns \
    && rm -rf /var/lib/apt/lists/*

# Configure buildah for rootless operation
RUN touch /etc/subgid /etc/subuid \
    && chmod g=u /etc/subgid /etc/subuid /etc/passwd \
    && echo root:10000:65536 > /etc/subuid \
    && echo root:10000:65536 > /etc/subgid

# Create containers config directory
RUN mkdir -p /root/.config/containers /etc/containers

# Configure storage driver for better performance
RUN echo '[storage]' > /etc/containers/storage.conf \
    && echo 'driver = "overlay"' >> /etc/containers/storage.conf \
    && echo 'runroot = "/run/containers/storage"' >> /etc/containers/storage.conf \
    && echo 'graphroot = "/var/lib/containers/storage"' >> /etc/containers/storage.conf \
    && echo '[storage.options]' >> /etc/containers/storage.conf \
    && echo 'mount_program = "/usr/bin/fuse-overlayfs"' >> /etc/containers/storage.conf \
    && echo 'mountopt = "nodev,metacopy=on"' >> /etc/containers/storage.conf

# Copy configuration files
ADD data/registries.conf /etc/containers/registries.conf
ADD data/containers.conf /etc/containers/containers.conf

# Create necessary directories
RUN mkdir -p /root/build /forge /var/lib/containers

# Install trivy
RUN curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin v0.68.1

# Install cosign
ENV COSIGN_VERSION=2.5.3
RUN curl -LO "https://github.com/sigstore/cosign/releases/download/v${COSIGN_VERSION}/cosign_${COSIGN_VERSION}_amd64.deb" && \
    dpkg -i cosign_${COSIGN_VERSION}_amd64.deb && \
    rm cosign_${COSIGN_VERSION}_amd64.deb

# Install uv (instead of Poetry) for dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy and install Python dependencies
ADD pyproject.toml /forge/
ADD uv.lock /forge/
ADD README.md /forge/
WORKDIR /forge/
RUN uv sync --no-dev

# Copy scripts
ADD data/buildah_cleanup.sh /usr/local/bin/buildah_cleanup.sh
ADD data/generate_fs_challenge.sh /usr/local/bin/generate_fs_challenge.sh
ADD data/trivy_scan.sh /usr/local/bin/trivy_scan.sh
RUN chmod +x /usr/local/bin/*.sh

# Copy application code
ADD --chown=root api /forge/api
ADD --chown=root metasync /forge/metasync
ADD --chown=root tokenizer /app/tokenizer
ADD --chown=chutes watchtower.py /forge/watchtower.py

# Set environment for buildah
ENV BUILDAH_ISOLATION=chroot
ENV STORAGE_DRIVER=overlay

ENTRYPOINT ["uv", "run", "python", "-m", "api.image.forge"]


###
# API
###
FROM base AS api
RUN curl -fsSL -o /usr/local/bin/dbmate https://github.com/amacneil/dbmate/releases/latest/download/dbmate-linux-amd64 && chmod +x /usr/local/bin/dbmate
RUN useradd chutes -s /bin/bash -d /home/chutes && mkdir -p /home/chutes && chown chutes:chutes /home/chutes
RUN mkdir -p /app && chown chutes:chutes /app
RUN ln -s /usr/bin/python3 /usr/bin/python

USER chutes
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH=$PATH:/home/chutes/.local/bin

USER root
WORKDIR /tmp/nv-attest
COPY --chown=chutes:chutes nv-attest /tmp/nv-attest
RUN poetry build -f wheel \
    && python -m venv /app/nv-attest \
    && /app/nv-attest/bin/pip install --no-cache-dir dist/*.whl
RUN rm -rf /tmp/nv-attest
RUN ln -s /app/nv-attest/bin/chutes-nvattest /usr/bin/chutes-nvattest
USER chutes

# Install uv for main app deps
WORKDIR /app
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/chutes/.local/bin:$PATH"

ADD pyproject.toml /app/
ADD --chown=chutes uv.lock /app/
ADD --chown=chutes README.md /app/
RUN uv sync --extra dev

ADD --chown=chutes api /app/api
ADD --chown=chutes audit_exporter.py /app/audit_exporter.py
ADD --chown=chutes failed_chute_cleanup.py /app/failed_chute_cleanup.py
ADD --chown=chutes metasync /app/metasync
ADD --chown=chutes tokenizer /app/tokenizer
ADD --chown=chutes watchtower.py /app/watchtower.py
ADD --chown=chutes cacher.py /app/cacher.py
ADD --chown=chutes chute_autoscaler.py /app/chute_autoscaler.py
ADD --chown=chutes balance_refresher.py /app/balance_refresher.py
ADD --chown=chutes data/cache_hit_cluster_params.json /app/cache_hit_cluster_params.json
ADD --chown=chutes log_prober.py /app/log_prober.py
ADD --chown=chutes conn_prober.py /app/conn_prober.py
ADD --chown=chutes scripts /app/scripts

ENV PYTHONPATH=/app

ENTRYPOINT ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--loop", "uvloop", "--http", "httptools"]
