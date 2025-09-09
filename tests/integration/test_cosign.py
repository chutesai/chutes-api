"""
Integration tests for the image forge pipeline.
Tests the actual forge function with real calls to buildah/cosign, only mocking to force error scenarios.
"""

import asyncio
import os
import tempfile
import time
import uuid
import pytest
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock
from sqlalchemy.future import select

# Import the functions we want to test
from api.image.forge import (
    forge, 
    build_and_push_image, 
    sign_image, 
    get_image_digest,
    extract_cfsv_data_from_verification_image,
    upload_filesystem_verification_data
)
from api.image.schemas import Image
from api.user.schemas import User
from api.database import get_session
from api.config import settings
from api.exceptions import (
    BuildFailure,
    PushFailure,
    BuildTimeout,
    PushTimeout,
    SignFailure,
    SignTimeout
)


# Fixtures for real test data
@pytest.fixture
def test_image_data():
    """Create a real test image record."""
    user_id = str(uuid.uuid4())
    image_id = str(uuid.uuid4())
    
    return {
        "user_id": user_id,
        "image_id": image_id,
        "username": "testuser",
        "name": "test-app",
        "tag": "latest",
        "patch_version": None,
    }


@pytest.fixture
def real_build_context(tmp_path):
    """Create a real build context with working Dockerfile and files."""
    # Create a simple working Python app
    app_content = '''
import sys
print("Hello from test app!")
print(f"Python version: {sys.version}")
'''
    
    dockerfile_content = f'''
FROM python:3.12-slim
WORKDIR /app
COPY app.py .
RUN pip install --no-cache-dir requests
RUN echo "Build completed successfully"
USER nobody
CMD ["python", "app.py"]
'''
    
    # Create build directory
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    
    # Write files
    (build_dir / "app.py").write_text(app_content)
    (build_dir / "Dockerfile").write_text(dockerfile_content)
    
    # Create zip file
    zip_path = tmp_path / "context.zip" 
    import zipfile
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(build_dir / "app.py", "app.py")
    
    return {
        "zip_path": zip_path,
        "dockerfile": dockerfile_content,
        "build_dir": build_dir
    }


# @pytest.fixture
# def real_cosign_keys(tmp_path):
#     """Create real cosign key pair for testing."""
#     import subprocess
    
#     private_key = tmp_path / "cosign.key"
#     public_key = tmp_path / "cosign.pub"
#     password = "test-cosign-password"
    
#     # Generate real cosign keys
#     env = os.environ.copy()
#     env["COSIGN_PASSWORD"] = password
    
#     result = subprocess.run([
#         "cosign", "generate-key-pair",
#         "--output-key-prefix", str(tmp_path / "cosign")
#     ], input=f"{password}\n{password}\n", text=True, capture_output=True, env=env)
    
#     if result.returncode != 0:
#         pytest.skip(f"Could not generate cosign keys: {result.stderr}")
    
#     return {
#         "private_key": private_key,
#         "public_key": public_key, 
#         "password": password
#     }


@pytest.fixture
async def test_database():
    """Setup test database session."""
    async with get_session() as session:
        yield session


# @pytest.fixture
# def mock_s3_only():
#     """Mock only S3 operations since we don't have real S3 in tests."""
#     s3_client = AsyncMock()
    
#     async def mock_s3_context():
#         return s3_client
    
#     with patch.object(settings, 's3_client', return_value=mock_s3_context()):
#         yield s3_client

@pytest.fixture(autouse=True)
def mock_cfsv(tmp_path):
    # Create CFSV binary mock
    mock_cfsv_dir = tmp_path / "mock_bin"
    mock_cfsv_dir.mkdir()
    cfsv_path = mock_cfsv_dir / "cfsv"
    cfsv_path.write_text("""#!/bin/bash
echo "Mock CFSV: $@"
case "$1" in
  index) echo "Mock index created" > "$3" ;;
  collect) echo "Mock data collected" > "$4" ;;
esac
exit 0
""")
    cfsv_path.chmod(0o755)

    with patch('api.image.forge.CFSV_PATH', str(cfsv_path)) as mock:
        yield mock

@pytest.fixture(autouse=True)
def mock_extract_cfsv(mock_cfsv):
    async def mock_extract_cfsv_data(verification_tag: str, build_dir: str) -> str:
        """Mock CFSV data extraction without container mounting."""
        data_file_path = os.path.join(build_dir, "chutesfs.data")
        with open(data_file_path, "w") as f:
            f.write("mock cfsv verification data")
        return data_file_path
    
    with patch('api.image.forge.extract_cfsv_data_from_verification_image', side_effect=mock_extract_cfsv_data) as mock:
        yield mock

@pytest.fixture(autouse=True)
def mock_trivy():
    with patch('api.image.forge.trivy_image_scan') as mock:
        yield mock

@pytest.fixture(autouse=True)
def mock_buildah_storage():
    with patch.dict(os.environ, {
        'STORAGE_DRIVER': 'vfs',
        'STORAGE_OPTS': ''
    }) as mock:
        yield mock

# Integration Tests - Real function calls with minimal mocking
@pytest.mark.asyncio
async def test_get_image_digest_real_image():
    """Test get_image_digest with a real image."""
    # Pull a small real image first
    import subprocess
    
    test_image = "alpine:latest"
    
    # Pull image
    result = subprocess.run(["buildah", "pull", test_image], capture_output=True)
    if result.returncode != 0:
        pytest.skip("Could not pull test image")
    
    try:
        # Test our function
        digest = await get_image_digest(test_image)
        
        # Verify format
        assert digest.startswith("sha256:")
        assert len(digest) == 71  # sha256: + 64 hex chars
        
    finally:
        # Cleanup
        subprocess.run(["buildah", "rmi", "--force", test_image], capture_output=True)


@pytest.mark.asyncio 
async def test_build_and_push_simple_image(real_build_context):
    """Test building and pushing a simple real image."""
    # Create minimal image object
    user = Mock()
    user.username = "testuser"
    
    image = Mock()
    image.image_id = str(uuid.uuid4())
    image.name = "test-simple"
    image.tag = "latest"
    image.patch_version = None
    image.user = user
    
    # Patch settings to use test registry
    original_registry = settings.registry_host
    original_insecure = settings.registry_insecure
    
    try:
        settings.registry_host = "localhost:5000"
        settings.registry_insecure = True
        
        # Change to the build directory before calling the method
        os.chdir(str(real_build_context["build_dir"]))

        result = await build_and_push_image(image, str(real_build_context["build_dir"]))
        
        # Verify result
        assert result == "testuser/test-simple:latest"
        
        # Verify image exists in registry (if registry is running)
        digest = await get_image_digest(f"localhost:5000/testuser/test-simple:latest")
        assert digest.startswith("sha256:")
            
    finally:
        settings.registry_host = original_registry
        settings.registry_insecure = original_insecure
        
        # Cleanup
        import subprocess
        subprocess.run([
            "buildah", "rmi", "--force", 
            f"localhost:5000/testuser/test-simple:latest"
        ], capture_output=True)


@pytest.mark.asyncio
async def test_sign_image(real_build_context):
    """Test signing a real image with real cosign."""
    # Build a simple test image first
    started_at = time.time()
    test_tag = f"localhost:5000/test-sign:{uuid.uuid4().hex[:8]}"
    
    import subprocess
    
    # Build test image
    build_result = subprocess.run([
        "buildah", "build", 
        "--isolation", "chroot",
        "--tag", test_tag,
        "-f", str(real_build_context["build_dir"] / "Dockerfile"),
        str(real_build_context["build_dir"])
    ], capture_output=True)
    
    if build_result.returncode != 0:
        pytest.skip(f"Could not build test image: {build_result.stderr.decode()}")
    
    # Push to registry
    push_result = subprocess.run([
        "buildah", "--tls-verify=false", "push", test_tag
    ], capture_output=True)
    
    if push_result.returncode != 0:
        pytest.skip(f"Could not push test image: {push_result.stderr.decode()}")
    
    try:
        # Create image mock
        image = Mock()
        image.image_id = "test123"

        # Mock capture logs function
        async def mock_capture_logs(stream, name, capture=True):
            if capture and stream:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    print(f"[{name}] {line.decode().strip()}")
        
        # Test signing
        await sign_image(image, test_tag, started_at, mock_capture_logs)
        
        # Verify signature exists (try to verify)
        verify_result = subprocess.run([
            "cosign", "verify",
            "--key", './tests/integration/keys/cosign.pub',
            test_tag
        ], capture_output=True)
        
        assert verify_result.returncode == 0, f"Signature verification failed: {verify_result.stderr.decode()}"
        
    finally:
        # Cleanup
        subprocess.run(["buildah", "rmi", "--force", test_tag], capture_output=True)


# Error scenario tests - Mock only to force errors
@pytest.mark.asyncio
async def test_build_and_push_image_buildah_failure(real_build_context):
    """Test build failure by mocking buildah to fail."""
    user = Mock()
    user.username = "testuser"
    
    image = Mock()
    image.image_id = str(uuid.uuid4())
    image.name = "test-fail"
    image.tag = "latest" 
    image.patch_version = None
    image.user = user
    
    # Mock buildah to fail on first subprocess call
    async def mock_failing_subprocess(*args, **kwargs):
        process = AsyncMock()
        process.returncode = 1  # Failure
        process.stdout = AsyncMock()
        process.stderr = AsyncMock()
        process.wait = AsyncMock()
        
        # Mock readline to return empty (end of stream)
        async def mock_readline():
            return b""
        process.stdout.readline = mock_readline
        process.stderr.readline = mock_readline
        
        return process
    
    with patch('asyncio.create_subprocess_exec', side_effect=mock_failing_subprocess):
            
        with pytest.raises(BuildFailure):
            await build_and_push_image(image, str(real_build_context["build_dir"]))


@pytest.mark.asyncio
async def test_build_and_push_image_timeout(real_build_context):
    """Test build timeout by mocking slow subprocess."""
    user = Mock()
    user.username = "testuser"
    
    image = Mock()
    image.image_id = str(uuid.uuid4())
    image.name = "test-timeout"
    image.tag = "latest"
    image.patch_version = None
    image.user = user
    
    # Temporarily set very short timeout
    original_timeout = settings.build_timeout
    settings.build_timeout = 1
    
    try:
        # Mock subprocess that hangs
        async def mock_hanging_subprocess(*args, **kwargs):
            process = AsyncMock()
            process.returncode = None
            process.stdout = AsyncMock()
            process.stderr = AsyncMock()
            
            # Mock operations that hang longer than timeout
            async def mock_hanging():
                await asyncio.sleep(5)
                return b""
            
            process.stdout.readline = mock_hanging
            process.stderr.readline = mock_hanging
            process.wait = mock_hanging
            
            return process
        
        with patch('asyncio.create_subprocess_exec', side_effect=mock_hanging_subprocess):
                
            with pytest.raises(BuildTimeout):
                await build_and_push_image(image, str(real_build_context["build_dir"]))
                    
    finally:
        settings.build_timeout = original_timeout


@pytest.mark.asyncio
async def test_sign_image_cosign_failure():
    """Test signing failure by corrupting the cosign key."""
    image = Mock()
    image.image_id = "test123"
    
    # Patch settings to use invalid key
    original_key = settings.cosign_key
    settings.cosign_key = Path(os.path.abspath("./tests/integration/keys/invalid.key"))
    
    try:
        async def mock_capture_logs(stream, name, capture=True):
            pass
        
        # This should fail because the key is invalid
        with pytest.raises(SignFailure):
            await sign_image(image, "localhost:5000/test:latest", 1000, mock_capture_logs)
            
    finally:
        settings.cosign_key = original_key


@pytest.mark.asyncio 
async def test_get_image_digest_nonexistent_image():
    """Test digest retrieval failure with nonexistent image."""
    nonexistent_image = f"nonexistent-image:{uuid.uuid4().hex}"
    
    # This should fail naturally since the image doesn't exist
    with pytest.raises(SignFailure, match="Failed to get digest"):
        await get_image_digest(nonexistent_image)