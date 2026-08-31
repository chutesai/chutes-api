import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.routing import Match

with patch("ctypes.CDLL", return_value=MagicMock()):
    from api import main


def _request(
    method: str,
    path: str,
    *,
    host: str = "api.chutes.ai",
    headers: list[tuple[bytes, bytes]] | None = None,
    chunks: list[tuple[bytes, bool]] | None = None,
) -> Request:
    raw_headers = [(b"host", host.encode()), *(headers or [])]
    messages = [
        {"type": "http.request", "body": body, "more_body": more_body}
        for body, more_body in (chunks or [(b"", False)])
    ]

    async def receive():
        if messages:
            return messages.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": method,
            "scheme": "https",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": raw_headers,
            "client": ("127.0.0.1", 12345),
            "server": ("api.chutes.ai", 443),
        },
        receive,
    )


@pytest.mark.asyncio
async def test_declared_content_length_is_rejected_before_body_read(monkeypatch):
    monkeypatch.setattr(main.settings, "max_request_body_bytes", 4)
    request = _request(
        "POST",
        "/not-a-route",
        headers=[(b"content-length", b"5")],
        chunks=[(b"would-not-be-read", False)],
    )
    call_next = AsyncMock()

    response = await main.host_router_middleware(request, call_next)

    assert response.status_code == 413
    call_next.assert_not_awaited()


@pytest.mark.asyncio
async def test_chunked_body_stops_at_process_ceiling(monkeypatch):
    monkeypatch.setattr(main.settings, "max_request_body_bytes", 4)
    request = _request(
        "POST",
        "/not-a-route",
        chunks=[(b"123", True), (b"45", False)],
    )
    call_next = AsyncMock()

    response = await main.host_router_middleware(request, call_next)

    assert response.status_code == 413
    call_next.assert_not_awaited()


@pytest.mark.asyncio
async def test_bounded_body_is_preserved_and_hashed_for_downstream(monkeypatch):
    monkeypatch.setattr(main.settings, "max_request_body_bytes", 4)
    request = _request(
        "POST",
        "/not-a-route",
        headers=[(b"content-length", b"4")],
        chunks=[(b"12", True), (b"34", False)],
    )

    async def call_next(inner_request):
        assert await inner_request.body() == b"1234"
        return SimpleNamespace(status_code=404)

    response = await main.host_router_middleware(request, call_next)

    assert response.status_code == 404
    assert request.state.body_sha256 == hashlib.sha256(b"1234").hexdigest()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["hosted", "external"])
async def test_chute_hostname_ping_remains_platform_health_unless_declared(
    monkeypatch, backend
):
    monkeypatch.setattr(main, "chute_id_by_slug", AsyncMock(return_value="chute-id"))
    monkeypatch.setattr(
        main,
        "get_one",
        AsyncMock(
            return_value=SimpleNamespace(
                execution_backend=backend,
                cords=[{"public_api_path": "/generate", "public_api_method": "POST"}],
            )
        ),
    )
    request = _request("GET", "/ping", host="model.chutes.ai")

    async def call_next(inner_request):
        return SimpleNamespace(
            status_code=200,
            invocation_dispatch=inner_request.state.invocation_dispatch,
        )

    response = await main.host_router_middleware(request, call_next)

    assert response.invocation_dispatch is False


@pytest.mark.asyncio
async def test_external_chute_may_explicitly_declare_ping_cord(monkeypatch):
    monkeypatch.setattr(main, "chute_id_by_slug", AsyncMock(return_value="chute-id"))
    monkeypatch.setattr(
        main,
        "get_one",
        AsyncMock(
            return_value=SimpleNamespace(
                execution_backend="external",
                cords=[{"public_api_path": "/ping", "public_api_method": "GET"}],
            )
        ),
    )
    request = _request("GET", "/ping", host="model.chutes.ai")

    async def call_next(inner_request):
        return SimpleNamespace(
            status_code=200,
            chute_id=inner_request.state.chute_id,
            invocation_dispatch=inner_request.state.invocation_dispatch,
        )

    response = await main.host_router_middleware(request, call_next)

    assert response.chute_id == "chute-id"
    assert response.invocation_dispatch is True


@pytest.mark.asyncio
async def test_management_segment_is_never_resolved_as_canonical_chute_name(
    monkeypatch,
):
    get_one = AsyncMock()
    monkeypatch.setattr(main, "get_one", get_one)
    request = _request("POST", "/chutes/code/private")

    async def call_next(inner_request):
        return SimpleNamespace(
            status_code=404,
            invocation_dispatch=inner_request.state.invocation_dispatch,
        )

    response = await main.host_router_middleware(request, call_next)

    assert response.invocation_dispatch is False
    get_one.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_canonical_chute_path_is_not_classified_as_invocation(
    monkeypatch,
):
    chute_id = "4acfd027-8f14-4a58-a60c-d54c59bc26ca"
    monkeypatch.setattr(
        main,
        "get_one",
        AsyncMock(
            return_value=SimpleNamespace(
                execution_backend="hosted",
                cords=[
                    {
                        "public_api_path": "/generate",
                        "public_api_method": "POST",
                    }
                ],
            )
        ),
    )
    request = _request(
        "POST",
        f"/chutes/{chute_id}/not-a-cord",
        headers=[(b"authorization", b"Bearer test-credential")],
    )

    async def call_next(inner_request):
        return SimpleNamespace(
            status_code=404,
            auth_method=inner_request.state.auth_method,
            declared=inner_request.state.canonical_cord_declared,
            invocation_dispatch=inner_request.state.invocation_dispatch,
        )

    response = await main.host_router_middleware(request, call_next)

    assert response.status_code == 404
    assert response.auth_method == "invoke"
    assert response.declared is False
    assert response.invocation_dispatch is True


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["hosted", "external"])
async def test_declared_canonical_chute_path_is_classified_as_invocation(
    monkeypatch, backend
):
    chute_id = "4acfd027-8f14-4a58-a60c-d54c59bc26ca"
    monkeypatch.setattr(
        main,
        "get_one",
        AsyncMock(
            return_value=SimpleNamespace(
                execution_backend=backend,
                cords=[
                    {
                        "public_api_path": "/generate",
                        "public_api_method": "POST",
                    }
                ],
            )
        ),
    )
    request = _request(
        "POST",
        f"/chutes/{chute_id}/generate",
        headers=[(b"authorization", b"Bearer test-credential")],
    )

    async def call_next(inner_request):
        return SimpleNamespace(
            status_code=401,
            auth_method=inner_request.state.auth_method,
            public_path=inner_request.state.invocation_public_path,
            invocation_dispatch=inner_request.state.invocation_dispatch,
        )

    response = await main.host_router_middleware(request, call_next)

    assert response.auth_method == "invoke"
    assert response.public_path == "/generate"
    assert response.invocation_dispatch is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reserved_path", ["/hf_info", "/HF_INFO", "/evidence", "/EVIDENCE"]
)
async def test_canonical_management_paths_cannot_be_shadowed_by_a_cord(
    monkeypatch, reserved_path
):
    chute_id = "4acfd027-8f14-4a58-a60c-d54c59bc26ca"
    monkeypatch.setattr(
        main,
        "get_one",
        AsyncMock(
            return_value=SimpleNamespace(
                execution_backend="external",
                cords=[
                    {
                        "public_api_path": reserved_path,
                        "public_api_method": "GET",
                    }
                ],
            )
        ),
    )
    request = _request("GET", f"/chutes/{chute_id}{reserved_path}")

    async def call_next(inner_request):
        return SimpleNamespace(
            status_code=404,
            auth_method=inner_request.state.auth_method,
            invocation_dispatch=inner_request.state.invocation_dispatch,
        )

    response = await main.host_router_middleware(request, call_next)

    assert response.auth_method == "read"
    assert response.invocation_dispatch is False


def test_unmatched_api_path_has_no_unguarded_http_catchall():
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "https",
        "path": "/definitely-not-an-api-route",
        "root_path": "",
        "query_string": b"",
        "headers": [],
        "state": {"invocation_dispatch": False},
    }

    assert all(
        route.matches(scope)[0] != Match.FULL for route in main.app.router.routes
    )


def test_known_api_path_with_wrong_method_retains_method_not_allowed_match():
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/ping",
        "root_path": "",
        "query_string": b"",
        "headers": [],
        "state": {"invocation_dispatch": False},
    }
    matches = [route.matches(scope)[0] for route in main.app.router.routes]

    assert Match.FULL not in matches
    assert Match.PARTIAL in matches


def test_api_domain_unmatched_and_wrong_method_status_codes():
    client = TestClient(main.app, raise_server_exceptions=False)

    assert client.get("/definitely-not-an-api-route").status_code == 404
    assert client.post("/ping").status_code == 405


def test_unauthenticated_declared_and_unknown_canonical_cords_share_404_shape(
    monkeypatch,
):
    chute_id = "4acfd027-8f14-4a58-a60c-d54c59bc26ca"
    monkeypatch.setattr(
        main,
        "get_one",
        AsyncMock(
            return_value=SimpleNamespace(
                execution_backend="external",
                cords=[
                    {
                        "public_api_path": "/generate",
                        "public_api_method": "POST",
                    }
                ],
            )
        ),
    )
    client = TestClient(main.app, raise_server_exceptions=False)

    declared = client.post(f"/chutes/{chute_id}/generate", json={})
    unknown = client.post(f"/chutes/{chute_id}/unknown", json={})

    assert declared.status_code == unknown.status_code == 404
    assert declared.json() == unknown.json() == {"detail": "Not Found"}


def test_invalid_credentials_cannot_probe_declared_canonical_cords(monkeypatch):
    chute_id = "4acfd027-8f14-4a58-a60c-d54c59bc26ca"
    monkeypatch.setattr(
        main,
        "get_one",
        AsyncMock(
            return_value=SimpleNamespace(
                execution_backend="external",
                cords=[
                    {
                        "public_api_path": "/generate",
                        "public_api_method": "POST",
                    }
                ],
            )
        ),
    )
    client = TestClient(main.app, raise_server_exceptions=False)
    headers = {"Authorization": "Bearer definitely-invalid"}

    declared = client.post(f"/chutes/{chute_id}/generate", json={}, headers=headers)
    unknown = client.post(f"/chutes/{chute_id}/unknown", json={}, headers=headers)

    assert declared.status_code == unknown.status_code == 401
    assert declared.json() == unknown.json()


@pytest.mark.parametrize("backend", ["hosted", "external"])
def test_authenticated_user_cannot_probe_private_chute_cord_paths(monkeypatch, backend):
    from api.invocation import router as invocation_router
    from api.user import service as user_service

    chute_id = "4acfd027-8f14-4a58-a60c-d54c59bc26ca"
    chute = SimpleNamespace(
        chute_id=chute_id,
        user_id="private-owner",
        public=False,
        execution_backend=backend,
        cords=[
            {
                "public_api_path": "/generate",
                "public_api_method": "POST",
            }
        ],
    )
    authenticated_user = SimpleNamespace(user_id="unrelated-user")
    monkeypatch.setattr(main, "get_one", AsyncMock(return_value=chute))
    monkeypatch.setattr(invocation_router, "get_one", AsyncMock(return_value=chute))
    monkeypatch.setattr(invocation_router, "is_shared", AsyncMock(return_value=False))
    monkeypatch.setattr(
        invocation_router, "subnet_role_accessible", lambda *_args: False
    )
    monkeypatch.setattr(
        user_service,
        "get_user_from_token",
        AsyncMock(return_value=authenticated_user),
    )
    client = TestClient(main.app, raise_server_exceptions=False)
    headers = {"Authorization": "Bearer registered-user-token"}

    declared = client.post(f"/chutes/{chute_id}/generate", json={}, headers=headers)
    unknown = client.post(f"/chutes/{chute_id}/unknown", json={}, headers=headers)

    assert declared.status_code == unknown.status_code == 404
    assert declared.json() == unknown.json() == {"detail": "No matching chute found!"}
