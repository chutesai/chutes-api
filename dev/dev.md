

## Env setup

Make sure you have a local GPU to run vllm with, for now.

### 1. Mac

##### 1.1 Env
```bash
pyenv local 3.12.4
poetry env use $(pyenv which python)
poetry install --with dev --no-root
poetry shell
```

#### 1.2 Start stuff

```bash
docker network create kind
docker compose up -d 
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
#### 2.2 Run

```bash
docker network create kind
docker compose up -d 
```

#### 2.3 Run the miner
```bash
docker compose -f docker-compose-miner.yml up -d vllm
```
