FROM quay.io/buildah/stable:v1.37.3 AS base

# Setup the various configurations/requirements for buildah.
RUN dnf install -y iputils procps vim
RUN touch /etc/subgid /etc/subuid \
  && chmod g=u /etc/subgid /etc/subuid /etc/passwd \
  && echo build:10000:65536 > /etc/subuid \
  && echo build:10000:65536 > /etc/subgid
RUN mkdir -p /root/.config/containers \
  && (echo '[storage]';echo 'driver = "vfs"') > /root/.config/containers/storage.conf
ADD data/registries.conf /etc/containers/registries.conf
RUN mkdir -p /root/build /forge

# Kubectl.
RUN dnf update && dnf install curl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && mv kubectl /usr/local/bin/ && chmod 755 /usr/local/bin/kubectl

# Layer for the buildah daemon.
FROM base AS forge
RUN curl -sSL https://install.python-poetry.org | python3 -
ADD pyproject.toml /forge/
ADD poetry.lock /forge/
WORKDIR /forge/
ENV PATH=$PATH:/root/.local/bin
RUN poetry install
ADD . /forge
ENTRYPOINT ["poetry", "run", "taskiq", "worker", "run_api.image.forge:broker", "--workers", "1", "--max-async-tasks", "1"]

# And finally our application code.
FROM base AS api
RUN useradd chutes -s /bin/bash -d /home/chutes && mkdir -p /home/chutes && chown chutes:chutes /home/chutes
USER chutes
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH=$PATH:/home/chutes/.local/bin
ADD pyproject.toml /app/
ADD poetry.lock /app/
WORKDIR /app
RUN poetry install
ADD --chown=chutes . /app
ENTRYPOINT ["poetry", "run", "uvicorn", "run_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
