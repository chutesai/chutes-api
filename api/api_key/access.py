"""Dependency-light helpers for checking server-resolved scope alternatives."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


Scope = tuple[str, str, str]


class ScopedCredential(Protocol):
    def has_access(self, object_type: str, object_id: str, action: str) -> bool: ...


def credential_has_any_access(
    credential: ScopedCredential,
    primary: Scope,
    alternatives: Iterable[object] = (),
) -> bool:
    """Check a middleware scope plus trusted, server-resolved alternatives."""

    if credential.has_access(*primary):
        return True
    for candidate in alternatives:
        if (
            isinstance(candidate, (list, tuple))
            and len(candidate) == 3
            and all(isinstance(value, str) for value in candidate)
            and credential.has_access(*candidate)
        ):
            return True
    return False


__all__ = ["Scope", "ScopedCredential", "credential_has_any_access"]
