"""Pure header-name policy shared by configuration and transport validation."""

from __future__ import annotations


def requires_secret_backing(name: str) -> bool:
    """Identify credential-like header names that require secret templates."""

    normalized = str(name).lower().replace("_", "-")
    compact = normalized.replace("-", "")
    parts = frozenset(part for part in normalized.split("-") if part)
    if any(
        marker in compact
        for marker in (
            "accesskey",
            "accesstoken",
            "apikey",
            "authtoken",
            "bearertoken",
            "clientsecret",
            "password",
            "passphrase",
            "privatekey",
            "secretkey",
            "sessiontoken",
            "signature",
            "subscriptionkey",
        )
    ):
        return True
    return bool(
        parts
        & {
            "auth",
            "authentication",
            "authorization",
            "cookie",
            "credential",
            "credentials",
            "key",
            "password",
            "passphrase",
            "secret",
            "signature",
            "token",
        }
    )


__all__ = ["requires_secret_backing"]
