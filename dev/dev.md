## Env setup

Make sure you have a local GPU to run vllm with, for now.

### 1. Mac

##### 1.0 Env
```bash
pyenv local 3.12.4
poetry env use $(pyenv which python)
poetry install --with dev --no-root
poetry shell
```

### 2. Linux

#### 2.0 Bootstrap
```bash
sudo -E bash dev/bootstrap.sh
```

#### 2.1 Install deps
```bash
poetry env use 3.12
poetry install --no-root
poetry shell
```

## Run stuff

### 1. Create the network

```bash
docker network create kind
```

### 2. Start all the things

#### 2.0 Base components

```bash
docker compose up -d
```

#### 2.1 [Optional, requires GPU] run the "miner" (dummy vllm instance)

```bash
docker compose -f docker-compose-gpu.yml up -d vllm
```

#### 2.2 [Optional, requires GPU] run the "graval" node validator

```bash
docker compose -f docker-compose-gpu.yml up -d graval
```

*If you do NOT want to verify GPUs/don't have GPU locally, be sure to set the `SKIP_GPU_VERIFICATION` env variable to `true` in the api service.*
