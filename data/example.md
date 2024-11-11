

```bash
# Re-create entire environment.
docker compose down
docker compose -f docker-compose-miner.yml down
docker volume rm chutes-api_kind_data chutes-api_kube_config chutes-api_minio_data chutes-api_postgres_data
docker compose up -d --build
docker compose -f docker-compose-miner.yml up -d vllm

# Register a user.
CHUTES_API_URL=http://127.0.0.1:8000 chutes register --username test

# Build an image.
poetry shell
pip install torch vllm  # Needed for this build example
chutes build data/vllm_example:chute --public --debug --wait
# Don't wait for the build, just ctrl+c and end it.
# XXX mark it built so we don't have to wait.
docker exec -it chutes-api-postgres-1 psql -U user -d chutes -c "update images set status = 'built and pushed'"

# Deploy.
chutes deploy data/vllm_example:chute --public 

# Seed the instance to immitate a real miner.
export CHUTE_ID=$(docker compose exec postgres psql -U user -d chutes -c "select chute_id from chutes order by created_at desc limit 1" -t)
export PYTHONPATH=$(pwd)
python ./run_api/bin/seed_instance --chute-id $CHUTE_ID

# Create an API key.
chutes api_key --name admin --admin

# Do some magic -- PASTE THE KEY from chutes_api_key at the end of the curl command:
curl -s http://test-unsloth-llama-3-2-1b-instruct.lvh.me:8000/v1/chat/completions -d '{"model": "unsloth/Llama-3.2-1B-Instruct", "messages": [{"role": "user", "content": "What is the secret to life, the universe, everything?."}], "stream": true, "max_tokens": 25}' -H 'Authorization: Bearer '
```