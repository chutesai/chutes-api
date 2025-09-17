import json

import pytest

from chutes_nvattest.verifier import NvVerifier

TEST_NONCE = "931d8dd0add203ac3d8b4fbde75e115278eefcdceac5b87671a748f32364dfcb"


@pytest.fixture
def verifier():
    return NvVerifier()


@pytest.fixture()
def evidence():
    with open("tests/assets/evidence.json") as fh:
        content = fh.read()
        yield json.loads(content)


def test_verifier_success(verifier, evidence):
    assert verifier.attest(TEST_NONCE, evidence)


def test_verifier_bad_nonce(verifier, evidence):
    assert not verifier.attest("abcd1234", evidence)
