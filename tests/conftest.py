import os


def pytest_configure(config):
    """Set up environment variables before any modules are imported."""
    os.environ.setdefault("COSIGN_KEY", os.path.abspath("./tests/integration/keys/cosign.key"))
    os.environ.setdefault("COSIGN_PASSWORD", "testpassword")

    os.environ.setdefault("STORAGE_BUCKET", "chutes")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin123")
    os.environ.setdefault("AWS_ENDPOINT_URL", "http://localhost:9000")

    # Print confirmation for debugging
    print("Environment variables set up for testing!")


pytest_configure(None)
