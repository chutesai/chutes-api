import textwrap
from api.affine import check_affine_code


def assert_valid(code: str) -> None:
    valid, message = check_affine_code(textwrap.dedent(code))
    assert valid, message


def assert_invalid(code: str, expected_substring: str) -> None:
    valid, message = check_affine_code(textwrap.dedent(code))
    assert not valid, "expected invalid code"
    assert expected_substring in message


def test_valid_sglang_chute_with_engine_args() -> None:
    assert_valid(
        """
        from chutes.chute import NodeSelector
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            readme="foo/affine-test",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            concurrency=40,
            revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            node_selector=NodeSelector(
                gpu_count=1,
                min_vram_gb_per_gpu=80,
            ),
            max_instances=5,
            shutdown_after_seconds=28800,
            scaling_threshold=0.5,
            engine_args=(
                "--mem-fraction-static 0.85 "
                "--context-length 36384"
            ),
        )
        """
    )


def test_valid_vllm_chute_with_engine_args() -> None:
    assert_valid(
        """
        from chutes.chute import NodeSelector
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            readme="foo/affine-test",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            concurrency=40,
            revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            node_selector=NodeSelector(
                gpu_count=1,
                min_vram_gb_per_gpu=80,
            ),
            max_instances=5,
            shutdown_after_seconds=28800,
            scaling_threshold=0.5,
            engine_args=(
                "--mem-fraction-static 0.85 "
                "--context-length 36384"
            ),
        )
        """
    )


def test_valid_allowed_env_vars() -> None:
    assert_valid(
        """
        import os
        from chutes.chute.template.sglang import build_sglang_chute

        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        os.environ["LMCACHE_USE_EXPERIMENTAL"] = "true"

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--context-length 4096",
        )
        """
    )


def test_invalid_disallowed_env_var() -> None:
    assert_invalid(
        """
        import os
        from chutes.chute.template.sglang import build_sglang_chute

        os.environ["AWS_SECRET_ACCESS_KEY"] = "nope"

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--context-length 4096",
        )
        """,
        "Setting os.environ['AWS_SECRET_ACCESS_KEY'] is not allowed",
    )


def test_invalid_engine_args_trust_remote_code() -> None:
    assert_invalid(
        """
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--trust-remote-code true",
        )
        """,
        "engine_args string cannot contain 'trust_remote_code' or 'trust-remote-code'",
    )


def test_invalid_engine_args_concatenated_flags() -> None:
    assert_invalid(
        """
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="--mem-fraction-static 0.85--context-length 36384",
        )
        """,
        "engine_args appears to contain concatenated flags",
    )


def test_invalid_engine_args_non_string() -> None:
    assert_invalid(
        """
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args={"foo": "bar"},
        )
        """,
        "engine_args for build_vllm_chute must be a string literal",
    )


def test_invalid_image_prefix_for_builder() -> None:
    assert_invalid(
        """
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--context-length 4096",
        )
        """,
        "image must start with 'chutes/vllm'",
    )


def test_invalid_import_outside_chutes() -> None:
    assert_invalid(
        """
        import math
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--context-length 4096",
        )
        """,
        "Invalid import: math",
    )


def test_invalid_missing_chute_assignment() -> None:
    assert_invalid(
        """
        from chutes.chute.template.sglang import build_sglang_chute

        something_else = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--context-length 4096",
        )
        """,
        "Function build_sglang_chute must be assigned to variable 'chute'",
    )


def test_invalid_engine_args_f_string_obfuscation() -> None:
    assert_invalid(
        """
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args=f"--trust-remote-code true",
        )
        """,
        "f-strings are not allowed",
    )


def test_invalid_engine_args_format_obfuscation() -> None:
    assert_invalid(
        """
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="--trust-{0} true".format("remote-code"),
        )
        """,
        "Dangerous function 'format' is not allowed",
    )


def test_invalid_engine_args_join_obfuscation() -> None:
    assert_invalid(
        """
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="".join(["--trust-remote", "-code true"]),
        )
        """,
        "String .join() method is not allowed",
    )


def test_invalid_engine_args_concat_non_literal() -> None:
    assert_invalid(
        """
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--trust-" + "remote-code true",
        )
        """,
        "engine_args for build_sglang_chute must be a string literal",
    )


def test_invalid_engine_args_hex_decode_obfuscation() -> None:
    assert_invalid(
        """
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args=bytes.fromhex("2d2d74727573742d72656d6f74652d636f64652074727565"),
        )
        """,
        "engine_args for build_sglang_chute must be a string literal",
    )


def test_invalid_env_var_contains_trust() -> None:
    assert_invalid(
        """
        import os
        from chutes.chute.template.vllm import build_vllm_chute

        os.environ["VLLM_TRUST_REMOTE_CODE"] = "1"

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="--context-length 4096",
        )
        """,
        "Setting os.environ['VLLM_TRUST_REMOTE_CODE'] is not allowed",
    )


def test_invalid_os_environ_update() -> None:
    assert_invalid(
        """
        import os
        from chutes.chute.template.sglang import build_sglang_chute

        os.environ.update({"VLLM_BATCH_INVARIANT": "1"})

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--context-length 4096",
        )
        """,
        "os.environ.update is not allowed",
    )


def test_invalid_os_environ_setdefault() -> None:
    assert_invalid(
        """
        import os
        from chutes.chute.template.vllm import build_vllm_chute

        os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="--context-length 4096",
        )
        """,
        "os.environ.setdefault is not allowed",
    )
