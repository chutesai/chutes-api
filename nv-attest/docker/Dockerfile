# Base layer
FROM python:3.12 AS base

ENV \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  POETRY_VERSION=1.5.1 \
  POETRY_VIRTUALENVS_IN_PROJECT=true \
  POETRY_NO_INTERACTION=1 \
  POETRY_HOME="/opt/poetry" \
  POETRY="/opt/poetry/bin/poetry"

ENV PATH="$POETRY_HOME/bin:$PATH"

ARG PROJECT_DIR

USER root

WORKDIR /build
## Setup any common packages or dependencies

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

USER ${APP_USER}

# ENTRYPOINT ["/app/scripts/entrypoint.sh"]

##################################################
## DEVELOPMENT ##
##################################################

FROM build AS development

ENV PATH="/app/.venv/bin:$PATH"
ENV APP_USER=chutes
ENV APP_GROUP=chutes
ENV ENVIRONMENT=development
ENV DEBUG=true

RUN addgroup -g 1001 ${APP_GROUP} && \
    adduser -u 1001 -G ${APP_GROUP} -h /home/${APP_USER} -s /bin/sh -D ${APP_USER}

USER root

WORKDIR /app/${PROJECT_DIR}
COPY --chown=${APP_USER}:${APP_GROUP} ./${PROJECT_DIR} ./
RUN poetry install
RUN chown ${APP_USER}:${APP_GROUP} .

USER ${APP_USER}