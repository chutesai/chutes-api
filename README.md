# 🪂 Chutes API

This repository contains all code, dockerfiles, etc. used by the chutes.ai platform's API/validator services.

The miner code is available [here](https://github.com/rayonlabs/chutes-miner), and CLI/SDK code [here](https://github.com/rayonlabs/chutes).

## 🛡️ Validators

While you *can* run a full validator on the chutes subnet, we strongly suggest making use of the child hotkey feature instead, with hotkey `5Dt7HZ7Zpw4DppPxFM7Ke3Cm7sDAWhsZXmM5ZAmE7dSVJbcQ`

Reasons for not running your own validator:
- The platform/API requires fairly extensive infrastructure, and along with it a fair amount of management/technical expertise (postgres, kubernetes, ansible, etc.), leading to high cost/touch.
- Miners need to manually allocate servers/GPUs to specific validators, along with manually configuring docker registries, certs, etc., and are unlikely to add additional validators depending on stake, which will cause low vtrust.
- Our validator hotkey take is set to 0%, so your earnings will likely be better using child hotkey vs. running a validator (or WC).

The high costs of properly operating validators across all 64 (soon more?) subnets often exceed potential validator returns given the goal of maximizing APY/reducing take, making selective subnet participation via child hotkeys a more practical approach in our opinion.

We will be happy to walk through the entire API/infrastructure with any concerned validators, and will do our best to ensure all operations are fully transparent.
You can verify the weights are being set appropriately by downloading invocation stats for the past 7 days via the `GET /invocations/exports/{year}/{month}/{day}/{hour}.csv` and `GET /invocations/exports/recent` endpoints.

## 🛠️ Configuring a full environment

Again, not recommended, but if you'd really like to run your own validator/API, you'll need to follow these steps.

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

### 🪣 S3-Compatible Object store

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

## Development

View the dev docs [here](dev/dev.md).  The entire chutes API can be run via docker-compose locally, although some components require GPUs (GraVal, vLLM example, etc.).
