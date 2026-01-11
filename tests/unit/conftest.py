"""
Unit test conftest - override autouse fixtures that require external dependencies.
"""

import pytest


@pytest.fixture(autouse=True)
def nv_attest():
    """Override the root conftest nv_attest fixture for unit tests.

    Unit tests should not require external CLI tools.
    """
    yield
