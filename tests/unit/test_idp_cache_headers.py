"""
Cache-header regression guard for the IDP (OAuth) router.

Responses that carry session-dependent data -- the login / consent / error HTML pages and the
token endpoint -- must set `Cache-Control: no-store` so browsers and proxies don't reuse a stale,
authenticated view (e.g. a logged-out login page restored from the back/forward cache after the
user signs in).

This module is fail-closed: every route registered on the IDP router must be explicitly
classified as either NO_STORE (must emit the header) or EXEMPT (allowed not to). A newly added
endpoint that is in neither set fails `test_every_idp_route_is_classified`, forcing the author to
decide rather than silently shipping an unclassified endpoint.
"""

import pytest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.database import get_db_session
from api.idp.router import router as idp_router

_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}

# (method, path) -> kwargs for a minimal request that reaches a response the handler itself
# returns (an early error branch), so the Cache-Control header is observable. The patched helpers
# in the test below make these handlers short out before touching the DB / redis.
NO_STORE_REQUESTS = {
    ("POST", "/token"): {"data": {"grant_type": "unsupported"}},
    ("GET", "/authorize"): {
        "params": {"response_type": "nope", "client_id": "x", "redirect_uri": "x"}
    },
    ("GET", "/cli_login"): {"params": {"hotkey": "x", "signature": "x", "nonce": "x"}},
    ("POST", "/login"): {"data": {"client_id": "x", "redirect_uri": "x", "auth_method": "x"}},
    ("GET", "/authorize/consent"): {"params": {"session_id": "x"}},
    ("POST", "/authorize/consent"): {"params": {"session_id": "x"}, "data": {"action": "deny"}},
}

# Endpoints that intentionally do NOT set no-store (JSON CRUD / public metadata). Adding an entry
# here is an explicit assertion that the endpoint's responses are safe to cache.
EXEMPT_ENDPOINTS = {
    ("GET", "/scopes"),
    ("GET", "/cli_login/nonce"),
    ("GET", "/apps"),
    ("POST", "/apps"),
    ("GET", "/apps/{app_id}"),
    ("PATCH", "/apps/{app_id}"),
    ("DELETE", "/apps/{app_id}"),
    ("POST", "/apps/{app_id}/regenerate-secret"),
    ("POST", "/apps/{app_id}/share"),
    ("DELETE", "/apps/{app_id}/share/{user_id}"),
    ("GET", "/apps/{app_id}/shares"),
    ("GET", "/authorizations"),
    ("DELETE", "/authorizations/{app_id}"),
    ("POST", "/token/revoke"),
    ("GET", "/userinfo"),
    ("POST", "/token/introspect"),
}

NO_STORE_ENDPOINTS = set(NO_STORE_REQUESTS)


def _registered_endpoints():
    """All (method, path) pairs registered on the IDP router."""
    endpoints = set()
    for route in idp_router.routes:
        for method in getattr(route, "methods", set()) & _HTTP_METHODS:
            endpoints.add((method, route.path))
    return endpoints


def test_every_idp_route_is_classified():
    """Fail-closed: a new IDP endpoint must be added to NO_STORE_REQUESTS or EXEMPT_ENDPOINTS."""
    registered = _registered_endpoints()
    classified = NO_STORE_ENDPOINTS | EXEMPT_ENDPOINTS

    unclassified = registered - classified
    assert not unclassified, (
        f"New IDP endpoint(s) not classified for cache headers: {sorted(unclassified)}. "
        "Either return NoStore* responses and add to NO_STORE_REQUESTS, or add to "
        "EXEMPT_ENDPOINTS if the responses are safe to cache."
    )

    stale = classified - registered
    assert not stale, (
        f"Classified endpoints no longer exist on the IDP router: {sorted(stale)}. "
        "Remove them from NO_STORE_REQUESTS / EXEMPT_ENDPOINTS."
    )


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(idp_router)
    # The error branches under test don't use the DB; provide a no-op so the dependency resolves.
    app.dependency_overrides[get_db_session] = lambda: None
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(("method", "path"), sorted(NO_STORE_ENDPOINTS))
def test_no_store_endpoints_set_cache_control(client, method, path):
    """Every session-dependent endpoint must emit Cache-Control: no-store on what it returns."""
    request_kwargs = NO_STORE_REQUESTS[(method, path)]
    # Force each handler down an early error branch so we observe the header without real I/O.
    with (
        patch("api.idp.router.verify_and_consume_login_nonce", AsyncMock(return_value=False)),
        patch("api.idp.router.get_app_by_client_id", AsyncMock(return_value=None)),
        patch("api.idp.router._get_session_data", AsyncMock(return_value=None)),
    ):
        resp = client.request(method, path, **request_kwargs)

    assert resp.headers.get("cache-control") == "no-store", (
        f"{method} {path} returned status {resp.status_code} with "
        f"cache-control={resp.headers.get('cache-control')!r}; expected 'no-store'."
    )
