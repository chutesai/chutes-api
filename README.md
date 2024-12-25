# 🪂 Chutes API

This repository contains all code, dockerfiles, etc. used by the chutes.ai platform's API/validator services.

The miner code is available [here](https://github.com/rayonlabs/chutes-miner), and CLI/SDK code [here](https://github.com/rayonlabs/chutes).

## 👨‍💻 Development

View the dev docs [here](dev/dev.md).  The entire chutes API can be run via docker-compose locally, although some components require GPUs (GraVal, vLLM example, etc.).

## 🛡️ Validators

While you absolutely *can* run a full validator on the chutes subnet, we strongly suggest making use of the child hotkey feature instead, with hotkey `5Dt7HZ7Zpw4DppPxFM7Ke3Cm7sDAWhsZXmM5ZAmE7dSVJbcQ` at this stage in the subnet's lifecycle.

Reasons *__to__* run your own validator:
- Decentralization and robustness of the network.
- Opportunity to launch your own instance of chutes with whatever domain/etc., you'd like, and change pricing/payout structure/etc.
- It's a pretty epic project.

Reasons to *__not__* run a validator:
- The platform/API requires fairly extensive infrastructure, and along with it a fair amount of management/technical expertise (postgres, kubernetes, ansible, etc.), leading to high cost/touch.
- Miners need to allocate servers/GPUs to specific validators currently, along with configuring docker registries, certs, etc., and may not support all validators depending on stake, which could lead to low vtrust.
- The CLI/SDK/etc., point to the `chutes.ai` subdomains by default, so any organic usage would need either a fork of the `chutes` package or override via environment variables.
- The chutes validator hotkey take is set to 0%, so your earnings will likely be better using child hotkey vs. running a validator (or WC).

The high costs of properly operating validators across all 64 (soon more?) subnets often exceed potential validator returns given the goal of maximizing APY/reducing take, making selective subnet participation via child hotkeys a more practical approach in our opinion.

We will be happy to walk through the entire API/infrastructure with any concerned validators, and will do our best to ensure all operations are fully transparent.

You can also verify the weights are being set appropriately by downloading invocation stats for the past 7 days via the `GET /invocations/exports/{year}/{month}/{day}/{hour}.csv` and `GET /invocations/exports/recent` endpoints.

## 🛠️ Running a full deployment

Again, not recommended at all, but if you'd really like to run your own validator/API, you'll need to follow these steps.

### 🤖 Provision and bootstrap nodes

We recommend at least two very large CPU-only (+ high RAM & several TB SSD volumes) instances for running the primary components, e.g. API, socket.io servers, graval workers, redis, forge instances (to build images), etc.

Additionally, you'll need GPUs for performing the actual GraVal encryption/verification operations.  Since miners can operate instances with 8+ GPUs each, and we support GPUs with up to 80GB VRAM, you'll likely want to run *at least* 8x a100s/h100s to handle this efficiently.

#### Ansible install (prerequisite)

##### Mac

If you haven't yet, setup homebrew:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install ansible:
```bash
brew install ansible
```

##### Ubuntu/aptitude based systems

```bash
sudo apt update && apt -y install -y ansible python3-pip
```

##### CentOS/RHEL/Fedora

Install epel repo if you haven't (and it's not fedora)
```bash
sudo dnf install epel-release -y
```

Install ansible:
```bash
sudo dnf install ansible -y
```

#### Install ansible collections

```bash
ansible-galaxy collection install community.general
ansible-galaxy collection install kubernetes.core
```

#### Run the playbooks

Once you've launched these nodes and have ansible + collections installed, head over to the `ansible` directory and update `inventory.yml` with the nodes you've created.

Then, run the playbook:
```bash
ansible-playbook -i inventory.yml site.yml
```

Once the nodes are fully bootstrapped with ansible, you'll want to join all kubernetes nodes to a single primary kubernetes instance.  You can do so by running the `join-cluster.yml` playbook:
```bash
ansible-playbook -i inventory.yml join-cluster.yml
```

And finally, install/configure the extras, e.g. kubernetes nvidia GPU operator:
```bash
ansible-playbook -i inventory.yml extras.yml
```

### 🪣 S3-Compatible Object Store

You'll need to configure an S3 compatible object store, which is used to store a variety of items such as image build contexts, image/chute logos, and docker registry data (lots and lots of data to store blobs).

Our validator uses GCS, since our infrastructure runs primarily in GCP, but AWS S3, CloudFlare R2, Minio on-prem, etc. should all work as well.

Once you have a bucket created, you'll need to create the kubernetes secret manually, from the IAM keys/endpoints/etc. you'll be using (run on primary node):
```bash
microk8s kubectl create secret generic s3-credentials \
  --from-literal="access-key-id=[access key id]" \
  --from-literal="secret-access-key=[secret access key]" \
  --from-literal="bucket=[bucket name]" \
  --from-literal="endpoint-url=[endpoint URL, e.g. https://storage.googleapis.com for GCS]" \
  --from-literal="aws-region=[AWS region, or auto for GCS]" \
  -n chutes
```

### 🐘 Postgres Database

You can run a postgres instance within kubernetes, although we highly recommend using a hosted solution such as AWS Aurora or GCP AlloyDB, which enable automatic backups, high availability, security upgrades, etc.

The postgres database is extraordinarily important, including storing (encrypted) bittensor wallet mnemonics.  If you lose the database, you will lose all access to tao payments, images, chutes, metrics, etc.

Once you have a database endpoint configured, you need to manually configure the secret in kubernetes:
```bash
microk8s kubectl create secret generic postgres-secret \
  --from-literal="username=[username, preferred chutes]" \
  --from-literal="password=[password]" \
  --from-literal="url=postgresql+asyncpg://[username]:[URL safe password]@[hostname/IP]:[port]/chutes" \
  --from-literal="hostname=[hostname/IP]" \
  --from-literal="port=5432" \
  --from-literal="database=chutes" \
  -n chutes
```

### 🔑 Other secrets

##### Validator ss58/seed

This secret contains the validator hotkey ss58Address and secretSeed value (with "0x" prefix removed), which is used to sign requests to miners, set weights, etc.
```bash
microk8s kubectl create secret generic validator-credentials \
  --from-literal="ss58=[nhotkey ss58Address]" \
  --from-literal="seed=[hotkey secretSeed, strip 0x prefix]" \
  -n chutes
```

##### Docker hub credentials

You'll need credentials to docker.io to avoid being rate-limited when pulling and/or building images.  Once you have access credentials (e.g. from registering an account and creating credentials), create the secret in kubernetes:
```bash
microk8s kubectl create secret docker-registry regcred \
  --docker-server=docker.io \
  --docker-username=[username] \
  --docker-password=[password] \
  --docker-email=[email address] \
  -n chutes
```

##### Docker registry secret

This will be the *actual* password to the local docker registry running on the validator. Miners have a custom docker registry auth mechanism, using bittensor hotkey signatures, but this is necessary for the forge process/etc.

Create a random secure string/password, then add it as a secret in kubernetes:
```bash
microk8s kubectl create secret generic registry-secret \
  --from-literal="password=[password] \
  -n chutes
```

Also create another secret in the chutes namespace for the forge process specifically:
```bash
microk8s kubectl create secret generic docker-pull \
  --from-literal="username=[username]" \
  --from-literal="password=[password]" \
  -n chutes
```

##### Bittensor wallet secrets

Every time someone registers a user, they get a unique bittensor coldkey address for payments and another for developer deposits (to enable building images/chutes).  The mnemonics for these wallets are stored doubly encrypted in the postgres database, so you need to create two secure strings to use for these encryption layers.

One must be 64 hex chars, the other 128 hex chars.  For example, you can use python to generate these:
```python
>>> import secrets
>>> secrets.token_bytes(64).hex()
'8122a176b26d5e5b4bdd8a53a51137f2c3e988269a1f606e621f15eadeba049a8872ecabc4fa96502cc91a1350c247bc511ebd587db520aadfdfa85345f4867a'
>>> secrets.token_bytes(32).hex()
'81e161b658f53ee95dbb3457b1cc6205071ba4eadc1455d388a66b0a6b6d026d'
```
*Notice that token_bytes(32) produces 64 hex chars and 64 bytes produces 128*

Then, create the secret in kubernetes:
```bash
microk8s kubectl create secret generic wallet-secret \
  --from-literal="wallet-key=[hex string with 64 chars]" \
  --from-literal="pg-key=[hex string with 128 chars]" \
  -n chutes
```

##### Redis password

Redis is used for pubsub and cache, and while it runs within kubernetes, you'll need to create the password before creating the component, e.g. a uuid4() works fine:
```bash
python3 -c 'import uuid; print(uuid.uuid4())'
```

Then create the secret:
```bash
microk8s kubectl create secret generic redis-secret \
  --from-literal="password=[password]" \
  --from-literal="url=redis://:[password]@redis.chutes.svc.cluster.local:6379/0" \
  -n chutes
```

##### GraVal database

Each GPU server/node will run it's own postgres instance storing GraVal challenges, via a kubernetes daemonset. You'll need to configure the postgres password for this database before launching the component:

UUIDs work fine for passwords here, e.g.:
```
python3 -c 'import uuid; print(uuid.uuid4())'
```

Then create the secret:
```bash
microk8s kubectl create secret generic gravaldb-secret \
  --from-literal="password=[password]" \
  -n chutes
```

### 🔒 Wildcard TLS certificate

When chutes (apps) are created on the platform, each chute receives a unique slug generated by the username and the chute name, which is used as a subdomain for doing simple HTTP invocations.

For example, a vLLM standard chute created by user "Jon" of model "unsloth/Llama-3.2-3B-Instruct" might be accessed via `https://jon-unsloth-llama-3-2-3b-instruct.chutes.ai/v1/chat/completions`

In addition to arbitrary subdomains, there are a handful of subdomains that are fixed, e.g. `api.`, `registry.`, `socket.`, `events.` and `llm.` (more to come likely).

In order to properly accomodate all variations, you should purchase a wildcard TLS certificate for whatever domain you'd like to use for your validator.  You *MUST* run TLS servers for miners to properly communicate with your validator API.

### ☸️ Helm charts/kubernetes

Once you have the kubernetes cluster fully operational (ansible bootstrap, cluster joined, extras installed, secrets created), you can deploy the services.  Head over to the `charts` directory, and make any modifications you'd like to `values.yaml`

The fields you'll be most likely to adjust are the two instances of subtensors if you're running a local subtensor, i.e. search and replace `subtensor: wss://entrypoint-finney.opentensor.ai`

Feel free to modify the remaining bits, but it's probably not necessary - you can change the replica counts down as well if you are not expecting a ton of traffic.

Once you are satisfied with your chart values, you can deploy (from the primary node, inside the charts directory):
```bash
helm template . > prod.yaml
microk8s kubectl apply -f prod.yaml -n chutes
```

This will take some time to fully deploy all components, and you can check the status via:
```bash
microk8s kubectl get po -n chutes -o wide
```

Ideally, all components will be in state `Running` within a few minutes.  If you have crashloops or other init errors, etc., you'll need to debug, check logs, adjust charts, etc.

### 🌐 Load balancers/proxies

Once the kubernetes components are all up and all in a "Running" state with "1/1" READY state (meaning all liveness/readiness probes are good), you'll want to setup a load balancer/proxy with your wildcard TLS cert with the following domains:

##### api.[domain] HTTPS (port 443)

This should point to port 32000 on each of your CPU nodes (e.g. with load balancing/failover, exclude any GPU nodes).

##### socket.[domain] HTTPS

This should point to port 32001 on each of your CPU nodes.

##### graval.[domain] HTTPS

This should point to port 32002 on each of your CPU nodes.

##### registry.[domain] HTTPS

This should point to port 32003 on each of your CPU nodes.

##### events.[domain] HTTPS

This should point to port 32004 on each of your CPU nodes.


*__NOTE: all ports must match the nodePort values in your chart!__*

### ⛏️ Request miner allocation

For your validator to be useful, you'll need to ensure miners pick up your validator and allocate resources to it.

You'll need to communicate to all miners (likely via the discord subnet channel), the following bits of information:
- hotkey SS58 address of your validator
- registry subdomain you've configured, e.g. in our case it's `registry.chutes.ai`
- API subdomain, e.g. `api.chutes.ai`
- socket subdomain, e.g. `ws.chutes.ai`

The miners will then need to update their helm charts and redeploy, then allocate GPU nodes specific to your validator.
