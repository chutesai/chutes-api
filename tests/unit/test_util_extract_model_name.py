import textwrap

from api.util import extract_hf_model_name


def test_extract_model_name_sglang() -> None:
    code = """
    from chutes.chute.template.sglang import build_sglang_chute

    chute = build_sglang_chute(
        username="exampleuser",
        model_name="foo/affine-test",
        image="chutes/sglang:nightly-2025121000",
    )
    """
    assert extract_hf_model_name("chute-1", textwrap.dedent(code)) == "foo/affine-test"


def test_extract_model_name_vllm() -> None:
    code = """
    from chutes.chute.template.vllm import build_vllm_chute

    chute = build_vllm_chute(
        username="exampleuser",
        model_name="foo/affine-test",
        image="chutes/vllm:nightly-2026010900",
    )
    """
    assert extract_hf_model_name("chute-2", textwrap.dedent(code)) == "foo/affine-test"


def test_extract_model_name_fallback_call() -> None:
    code = """
    from chutes.chute.template.sglang import build_sglang_chute

    something_else = build_sglang_chute(
        username="exampleuser",
        model_name="foo/affine-test",
        image="chutes/sglang:nightly-2025121000",
    )
    """
    assert extract_hf_model_name("chute-3", textwrap.dedent(code)) == "foo/affine-test"


def test_extract_model_name_missing_returns_empty() -> None:
    code = """
    from chutes.chute.template.vllm import build_vllm_chute

    chute = build_vllm_chute(
        username="exampleuser",
        image="chutes/vllm:nightly-2026010900",
    )
    """
    assert extract_hf_model_name("chute-4", textwrap.dedent(code)) == ""


def test_extract_model_name_non_string_returns_empty() -> None:
    code = """
    from chutes.chute.template.vllm import build_vllm_chute

    chute = build_vllm_chute(
        username="exampleuser",
        model_name={"name": "foo/affine-test"},
        image="chutes/vllm:nightly-2026010900",
    )
    """
    assert extract_hf_model_name("chute-5", textwrap.dedent(code)) == ""


def test_extract_model_name_invalid_syntax_returns_empty() -> None:
    code = "def nope(:"
    assert extract_hf_model_name("chute-6", code) == ""


def test_extract_model_name_cache_key_includes_chute_id() -> None:
    code = """
    from chutes.chute.template.vllm import build_vllm_chute

    chute = build_vllm_chute(
        username="exampleuser",
        model_name="foo/affine-test",
        image="chutes/vllm:nightly-2026010900",
    )
    """
    code = textwrap.dedent(code)
    assert extract_hf_model_name("chute-7", code) == "foo/affine-test"
    assert extract_hf_model_name("chute-8", code) == "foo/affine-test"
