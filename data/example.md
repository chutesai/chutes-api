```bash
#!/bin/bash

set -eux

# (re)create the entire environment.
docker compose down
docker compose -f docker-compose-gpu.yml down
docker volume rm parachutes-api_minio_data parachutes-api_postgres_data || true
docker compose up -d --build
docker compose -f docker-compose-gpu.yml up --build vllm -d

# Wait a second...
echo "Waiting for services..."
sleep 10

# Register a user.
CHUTES_API_URL=http://127.0.0.1:8000 poetry run chutes register

# Build the image.
cd data
poetry run pip install vllm==0.6.2
poetry run chutes build vllm_example:chute --public --debug --wait &
sleep 30
kill -9 %1
cd -

# Mark image as ready so we don't have to wait.
docker compose exec -it postgres psql -U user chutes -c "update images set status = 'built and pushed'"

# Deploy.
poetry run chutes deploy vllm_example:chute --public

# Seed the instance.
export CHUTE_ID=$(docker compose exec postgres psql -U user -d chutes -c "select chute_id from chutes order by created_at desc limit 1" -t)
export PYTHONPATH=$(pwd)
poetry run python ./api/bin/seed_instance --chute-id $CHUTE_ID

# Create an API key.
poetrun run chutes keys create --name admin --admin

# Do some magic -- PASTE THE KEY from chutes_api_key at the end of the curl command:
# curl -s http://test-unsloth-llama-3-2-1b-instruct.lvh.me:8000/v1/chat/completions -d '{"model": "unsloth/Llama-3.2-1B-Instruct", "messages": [{"role": "user", "content": "What is the secret to life, the universe, everything?."}], "stream": true, "max_tokens": 25}' -H 'Authorization: Bearer '
```
