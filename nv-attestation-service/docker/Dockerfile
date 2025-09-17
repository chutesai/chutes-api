FROM rancher/k3s:v1.33.1-k3s1 AS base

ENV \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  POETRY_VERSION=1.5.1 \
  POETRY_VIRTUALENVS_IN_PROJECT=true \
  POETRY_NO_INTERACTION=1 \
  POETRY_HOME="/opt/poetry" \
  POETRY="/opt/poetry/bin/poetry"

ENV PATH="$POETRY_HOME/bin:$PATH"
ENV K3S_KUBECONFIG_MODE=644

ARG PROJECT_DIR

USER root

WORKDIR /build
RUN wget http://dl-cdn.alpinelinux.org/alpine/v3.22/main/x86_64/apk-tools-static-2.14.9-r2.apk && \
    tar -xzf apk-tools-static-2.14.9-r2.apk && \
    mkdir /sbin && \
    cp sbin/apk.static /sbin/apk && \
    chmod +x /sbin/apk && \
    mkdir -p /etc/apk && \
    echo "http://dl-cdn.alpinelinux.org/alpine/v3.22/main" > /etc/apk/repositories && \
    echo "http://dl-cdn.alpinelinux.org/alpine/v3.22/community" >> /etc/apk/repositories && \
    mkdir -p /lib/apk/db && \
    mkdir -p /var/cache/apk && \
    mkdir -p /etc/apk/keys && \
    apk --root / --initdb add --no-cache --allow-untrusted && \
    wget http://dl-cdn.alpinelinux.org/alpine/v3.22/main/x86_64/alpine-keys-2.5-r0.apk && \
    tar -xzf alpine-keys-*.apk && \
    cp etc/apk/keys/* /etc/apk/keys/ 2>/dev/null || true

# Install additional development tools
RUN apk add --no-cache \
    curl \
    jq \
    vim \
    htop \
    bash \
    tcpdump \
    bind-tools

# Install OPA
RUN curl -L -o opa https://openpolicyagent.org/downloads/latest/opa_linux_amd64_static && \
    chmod 755 ./opa && \
    mv opa /usr/bin

# Install cosign verify


# Copy k3s configuration
COPY config/k3s-config.yml /etc/rancher/k3s/config.yml

# Install python
RUN apk add --update --no-cache python3 && ln -sf python3 /usr/bin/python

WORKDIR /
RUN rm -r /build

FROM base AS build
# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && chmod a+x /opt/poetry/bin/poetry

WORKDIR /app/${PROJECT_DIR}
COPY ${PROJECT_DIR}/pyproject.toml ${PROJECT_DIR}/poetry.lock ./
RUN poetry install --only main

##################################################
## PRODUCTION ##
##################################################

FROM base AS production
ARG PROJECT_DIR

USER root

ENV PATH="/app/.venv/bin:$PATH"
ENV APP_USER=chutes
ENV APP_GROUP=chutes
RUN addgroup -g 1001 ${APP_GROUP} && \
    adduser -u 1001 -G ${APP_GROUP} -h /home/${APP_USER} -s /bin/sh -D ${APP_USER}

WORKDIR /app
COPY --chown=${APP_USER}:${APP_GROUP} ${PROJECT_DIR}/ ./
COPY --from=build /app .
# USER ${APP_USER}

# ENTRYPOINT ["/app/scripts/entrypoint.sh"]

##################################################
## DEVELOPMENT ##
##################################################

FROM build AS development

ENV PATH="/app/.venv/bin:$PATH"
ENV APP_USER=chutes
ENV APP_GROUP=chutes
ENV ENVIRONMENT=development
ENV K3S_KUBECONFIG_MODE=644
ENV DEBUG=true

RUN addgroup -g 1001 ${APP_GROUP} && \
    adduser -u 1001 -G ${APP_GROUP} -h /home/${APP_USER} -s /bin/sh -D ${APP_USER}

USER root

WORKDIR /app/${PROJECT_DIR}
COPY --chown=${APP_USER}:${APP_GROUP} ./${PROJECT_DIR} ./
RUN poetry install
RUN chown ${APP_USER}:${APP_GROUP} .

# Copy over custom entrypoint for k3d to start adminssion-controller
COPY --chmod=744 scripts/k3d-entrypoint-custom.sh /bin/k3d-entrypoint-custom.sh
# USER ${APP_USER}