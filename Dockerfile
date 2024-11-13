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
RUN dnf install -y jq
RUN curl -sSL https://install.python-poetry.org | python3 -
ADD pyproject.toml /forge/
ADD poetry.lock /forge/
WORKDIR /forge/
ENV PATH=$PATH:/root/.local/bin
RUN poetry install
ADD data/buildah_cleanup.sh /usr/local/bin/buildah_cleanup.sh
ADD data/generate_fs_challenge.sh /usr/local/bin/generate_fs_challenge.sh
ADD . /forge
ENTRYPOINT ["poetry", "run", "taskiq", "worker", "api.image.forge:broker", "--workers", "1", "--max-async-tasks", "1"]

# Layer for the metagraph syncer.
FROM base AS metasync
RUN dnf install -y git cmake gcc gcc-c++ python3-devel
RUN useradd chutes -s /bin/bash -d /home/chutes && mkdir -p /home/chutes && chown chutes:chutes /home/chutes
USER chutes
RUN python3 -m venv /home/chutes/venv
ENV PATH=/home/chutes/venv/bin:$PATH
ADD pyproject.toml /tmp/
RUN egrep '^(SQLAlchemy|pydantic-settings|asyncpg) ' /tmp/pyproject.toml | sed 's/ = "^/==/g' | sed 's/"//g' > /tmp/requirements.txt
# TODO: Pin the below versions
RUN pip install git+https://github.com/rayonlabs/fiber.git redis netaddr && pip install -r /tmp/requirements.txt 
ADD --chown=chutes . /app
WORKDIR /app
ENV PYTHONPATH=/app
ENTRYPOINT ["python", "metasync/sync_metagraph.py"]

# Main API.
FROM base AS api
RUN curl -fsSL -o /usr/local/bin/dbmate https://github.com/amacneil/dbmate/releases/latest/download/dbmate-linux-amd64 && chmod +x /usr/local/bin/dbmate
RUN useradd chutes -s /bin/bash -d /home/chutes && mkdir -p /home/chutes && chown chutes:chutes /home/chutes
RUN mkdir -p /app && chown chutes:chutes /app
USER chutes
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH=$PATH:/home/chutes/.local/bin
ADD pyproject.toml /app/
ADD poetry.lock /app/
WORKDIR /app
RUN poetry install
ADD --chown=chutes . /app
ENTRYPOINT ["poetry", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# GraVal
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS graval
RUN apt-get update
RUN apt-get -y install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev libexpat1-dev lzma liblzma-dev libpq-dev
WORKDIR /usr/src
RUN wget https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tgz
RUN tar -xzf Python-3.12.7.tgz
WORKDIR /usr/src/Python-3.12.7
RUN ./configure --enable-optimizations --enable-shared --with-system-expat --with-ensurepip=install --prefix=/opt/python
RUN make -j
RUN make altinstall
RUN ln -s /opt/python/bin/pip3.12 /opt/python/bin/pip
RUN ln -s /opt/python/bin/python3.12 /opt/python/bin/python
RUN echo /opt/python/lib >> /etc/ld.so.conf && ldconfig
RUN rm -rf /usr/src/Python*
RUN apt-get -y install google-perftools git
RUN curl -fsSL -o /usr/local/bin/dbmate https://github.com/amacneil/dbmate/releases/latest/download/dbmate-linux-amd64 && chmod +x /usr/local/bin/dbmate
RUN useradd chutes -s /bin/bash -d /home/chutes && mkdir -p /home/chutes && chown chutes:chutes /home/chutes
RUN mkdir -p /app && chown chutes:chutes /app
USER chutes
ENV PATH=/opt/python/bin:$PATH
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH=$PATH:/home/chutes/.local/bin
ADD pyproject.toml /app/
ADD poetry.lock /app/
WORKDIR /app
RUN poetry install
ADD --chown=chutes . /app
ENTRYPOINT ["poetry", "run", "taskiq", "worker", "api.node.graval:broker", "--workers", "1", "--max-async-tasks", "1"]
