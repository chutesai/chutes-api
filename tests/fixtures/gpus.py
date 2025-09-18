import json

import pytest

TEST_GPU_NONCE = "931d8dd0add203ac3d8b4fbde75e115278eefcdceac5b87671a748f32364dfcb"

@pytest.fixture
def sample_gpu_evidence():
    with open('nv-attest/tests/assets/evidence.json') as fh:
        content = fh.read()
        evidence = json.loads(content)
        yield evidence