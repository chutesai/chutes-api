"""Standard Chute template identifiers shared by execution backends."""

STANDARD_TEMPLATES = frozenset({"vllm", "diffusion", "tei", "embedding"})


def standard_template_matches(
    candidate: str | None,
    requested: str,
    *,
    execution_backend: str = "hosted",
) -> bool:
    """Match a Chute template without widening legacy hosted routing.

    The external catalog accepts ``tei`` as an embedding-compatible declaration,
    while hosted mega routing retains its historical exact ``embedding`` match.
    """

    if requested == "embedding":
        return candidate == "embedding" or (
            candidate == "tei" and execution_backend == "external"
        )
    return candidate == requested


__all__ = ["STANDARD_TEMPLATES", "standard_template_matches"]
