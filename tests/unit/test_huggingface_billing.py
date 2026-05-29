from api.api_key.schemas import APIKeyScope, Action
from api.payment.router import usd_to_nano_usd


def test_usd_to_nano_usd_is_deterministic():
    assert usd_to_nano_usd("0") == 0
    assert usd_to_nano_usd("0.000218875") == 218875
    assert usd_to_nano_usd("1.234567891") == 1234567891
    assert usd_to_nano_usd("0.0000000005") == 1
    assert usd_to_nano_usd("-1") == 0
    assert usd_to_nano_usd("not-a-number") == 0


def test_api_keys_can_be_scoped_to_billing_read():
    scope = APIKeyScope(object_type="billing", action=Action.READ)
    assert scope.object_type == "billing"


def test_unknown_api_key_scope_is_still_rejected():
    try:
        APIKeyScope(object_type="huggingface", action=Action.READ)
    except ValueError as exc:
        assert "billing" in str(exc)
    else:
        raise AssertionError("unknown API key scope should be rejected")
