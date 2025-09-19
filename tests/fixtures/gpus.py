import base64
import json
import os

import pytest

TEST_GPU_NONCE = "931d8dd0add203ac3d8b4fbde75e115278eefcdceac5b87671a748f32364dfcb"

@pytest.fixture
def sample_gpu_evidence():
    with open('nv-attest/tests/assets/evidence.json') as fh:
        content = fh.read()
        evidence = json.loads(content)
        yield evidence

@pytest.fixture
def sample_gpu_evidence_base64(sample_gpu_evidence):
    yield base64.b64encode(json.dumps(sample_gpu_evidence).encode('utf-8')).decode('utf-8')

@pytest.fixture(autouse=True)
def nv_attest():
    if not os.path.exists("./nv-attest/.venv/bin/chutes-nvattest"):
        pytest.skip("chutes-nvattest CLI is not available.")
    
    original_path = os.environ.get("PATH")
    os.environ["PATH"] = f"{original_path}:{os.path.join(os.getcwd(), "nv-attest/.venv/bin")}"

    yield

    os.environ["PATH"] = original_path