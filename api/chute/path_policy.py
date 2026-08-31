"""Shared policy for canonical Chute management paths."""

RESERVED_CANONICAL_CHUTE_PATHS = frozenset({"/evidence", "/hf_info"})


def is_reserved_canonical_chute_path(path: object) -> bool:
    """Return whether *path* exactly names a reserved canonical endpoint."""
    return isinstance(path, str) and path.casefold() in RESERVED_CANONICAL_CHUTE_PATHS
