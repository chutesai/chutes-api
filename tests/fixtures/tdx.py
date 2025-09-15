import base64
from pathlib import Path
import pytest


EXPECTED_MRTD="ddc6efcdd2309e10837f8a7f64b71272b7ef003b129460410fe715bdfffec38c7c0c1686dddb2a23d4fd623d145e8455".upper()
EXPECTED_RMTR0="f57dde95ee98c7f7ae1284a5782ff5029d8ff25ad113467c1e8088ee5e3c65b3ce8ff15ba46f329d9b085c55d171d73a".upper()
EXPECTED_RMTR1="d20eb28ca35d29857d00dfe8875063f10900aae31cf23c2cdb0f7091a0e89358c8569edd55ddd596b78c515441422925".upper()
EXPECTED_RMTR2="95521b8702c3e2b43d9a938653762fe4a86a65bcf31b58c2821bc339e56ef806ccd44a26c9d6db3992ef3060beb416b3".upper()
EXPECTED_RMTR3="000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000".upper()
EXPECTED_USER_DATA="61626364313233340000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000".upper()

@pytest.fixture
def valid_quote_bin():
    """Load the valid quote.bin file from test assets."""
    quote_path = Path("tests/assets/quote.bin")
    if not quote_path.exists():
        pytest.skip(f"Valid quote file not found at {quote_path}")
    
    with open(quote_path, "rb") as f:
        quote_bytes = f.read()
    
    return base64.b64encode(quote_bytes).decode('utf-8')


@pytest.fixture
def valid_quote_bytes(valid_quote_bin):
    """Return the decoded bytes of the valid quote."""
    return base64.b64decode(valid_quote_bin)