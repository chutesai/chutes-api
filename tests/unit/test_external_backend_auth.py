import pytest

from api.api_key.access import credential_has_any_access
from api.external_backend.auth import ExternalAuthScope, external_auth_scope


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "/external/operations",
            ExternalAuthScope("invocations", "__list_or_invalid__"),
        ),
        (
            "/external/operations/operation-id",
            ExternalAuthScope("invocations", "operation-id"),
        ),
        (
            "/external/operations/operation-id/artifacts/0",
            ExternalAuthScope("invocations", "operation-id"),
        ),
        (
            "/external/operations/operation-id/cancel",
            ExternalAuthScope("invocations", "operation-id"),
        ),
        (
            "/external/accounts/account-id",
            ExternalAuthScope("account", "account-id"),
        ),
        (
            "/external/bindings",
            ExternalAuthScope("account", "__list_or_invalid__"),
        ),
        (
            "/external/chutes/chute-id",
            ExternalAuthScope("chutes", "chute-id"),
        ),
    ],
)
def test_external_auth_scope(path, expected):
    assert external_auth_scope(path) == expected


@pytest.mark.parametrize("path", ["/external", "/external/unknown", "/chutes/id"])
def test_external_auth_scope_ignores_unrelated_paths(path):
    assert external_auth_scope(path) is None


def test_operation_scope_accepts_the_server_resolved_chute_invoke_scope():
    class Credential:
        def has_access(self, object_type, object_id, action):
            return (object_type, object_id, action) == (
                "chutes",
                "chute-id",
                "invoke",
            )

    assert credential_has_any_access(
        Credential(),
        ("invocations", "operation-id", "read"),
        (("chutes", "chute-id", "invoke"),),
    )
    assert not credential_has_any_access(
        Credential(),
        ("invocations", "operation-id", "read"),
        (("chutes", "other-chute", "invoke"),),
    )
