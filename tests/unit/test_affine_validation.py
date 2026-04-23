import textwrap
from api.affine import check_affine_code, force_affine_engine_args, transform_code_for_tee


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
                "--mem-fraction-static 0.8 "
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
                "--mem-fraction-static 0.8 "
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
        "engine_args cannot contain 'trust_remote_code' or 'trust-remote-code'",
    )


def test_invalid_engine_args_concatenated_flags() -> None:
    assert_invalid(
        """
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="--mem-fraction-static 0.8--context-length 36384",
        )
        """,
        "engine_args appears to contain concatenated flags",
    )


def test_invalid_engine_args_mem_fraction_static_too_high() -> None:
    assert_invalid(
        """
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--mem-fraction-static 0.9 --context-length 36384",
        )
        """,
        "--mem-fraction-static value 0.9 is too high (must be < 0.85",
    )


def test_invalid_engine_args_mem_fraction_static_at_boundary() -> None:
    assert_invalid(
        """
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="--mem-fraction-static 0.85",
        )
        """,
        "--mem-fraction-static value 0.85 is too high (must be < 0.85",
    )


def test_invalid_engine_args_gpu_memory_utilization_too_high() -> None:
    assert_invalid(
        """
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="--gpu-memory-utilization 0.95",
        )
        """,
        "--gpu-memory-utilization value 0.95 is too high (must be < 0.85",
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


# --- force_affine_engine_args tests ---


def test_force_sglang_adds_missing_engine_args() -> None:
    code = textwrap.dedent("""
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
        )
    """)
    result = force_affine_engine_args(code)
    assert result is not None
    assert "--mem-fraction-static 0.8" in result
    assert "--chunked-prefill-size 4096" in result


def test_force_vllm_adds_missing_engine_args() -> None:
    code = textwrap.dedent("""
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
        )
    """)
    result = force_affine_engine_args(code)
    assert result is not None
    assert "--gpu-memory-utilization 0.8" in result
    assert "--max-num-batched-tokens 4096" in result


def test_force_replaces_conflicting_numeric_values() -> None:
    code = textwrap.dedent("""
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--mem-fraction-static 0.95 --chunked-prefill-size 2048",
        )
    """)
    result = force_affine_engine_args(code)
    assert result is not None
    assert "--mem-fraction-static 0.8" in result
    assert "--chunked-prefill-size 4096" in result
    assert "0.95" not in result
    assert "2048" not in result


def test_force_replaces_non_numeric_values() -> None:
    code = textwrap.dedent("""
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="--gpu-memory-utilization auto --max-num-batched-tokens auto",
        )
    """)
    result = force_affine_engine_args(code)
    assert result is not None
    assert "--gpu-memory-utilization 0.8" in result
    assert "--max-num-batched-tokens 4096" in result
    assert "auto" not in result


def test_force_preserves_other_flags() -> None:
    code = textwrap.dedent("""
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--context-length 36384 --mem-fraction-static 0.8 --chunked-prefill-size 4096",
        )
    """)
    result = force_affine_engine_args(code)
    # No change needed — should return original code.
    assert result == code
    assert "--context-length" in result


def test_force_returns_none_for_no_builder() -> None:
    code = textwrap.dedent("""
        x = 1 + 2
    """)
    assert force_affine_engine_args(code) is None


def test_force_returns_none_for_syntax_error() -> None:
    assert force_affine_engine_args("def (((") is None


def test_force_no_duplicate_flags() -> None:
    code = textwrap.dedent("""
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="--gpu-memory-utilization 0.7 --max-num-batched-tokens 2048 --context-length 4096",
        )
    """)
    result = force_affine_engine_args(code)
    assert result is not None
    assert result.count("--gpu-memory-utilization") == 1
    assert result.count("--max-num-batched-tokens") == 1


def test_force_targets_top_level_assignment_only() -> None:
    """Only the top-level assignment to a builder call should be rewritten."""
    code = textwrap.dedent("""
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--context-length 36384",
        )
    """)
    result = force_affine_engine_args(code)
    assert result is not None
    assert "--mem-fraction-static 0.8" in result
    assert "--chunked-prefill-size 4096" in result
    # The original flag is preserved.
    assert "--context-length" in result


# --- force_affine_engine_args with gpu_count (TP/DP) tests ---


def test_force_sglang_tp_dp_args() -> None:
    code = textwrap.dedent("""
        from chutes.chute import NodeSelector
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            node_selector=NodeSelector(gpu_count=4, include=["pro_6000"]),
            engine_args="--context-length 36384",
        )
    """)
    result = force_affine_engine_args(code, gpu_count=4)
    assert result is not None
    assert "--tp 1" in result
    assert "--dp 4" in result
    assert "--mem-fraction-static 0.8" in result
    assert "--chunked-prefill-size 4096" in result


def test_force_vllm_tp_dp_args() -> None:
    code = textwrap.dedent("""
        from chutes.chute import NodeSelector
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            node_selector=NodeSelector(gpu_count=8, include=["pro_6000"]),
            engine_args="--context-length 36384",
        )
    """)
    result = force_affine_engine_args(code, gpu_count=8)
    assert result is not None
    assert "--tensor-parallel-size 1" in result
    assert "--data-parallel-size 8" in result
    assert "--gpu-memory-utilization 0.8" in result
    assert "--max-num-batched-tokens 4096" in result


def test_force_tp_dp_replaces_existing_values() -> None:
    code = textwrap.dedent("""
        from chutes.chute import NodeSelector
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            node_selector=NodeSelector(gpu_count=8, include=["pro_6000"]),
            engine_args="--tp 4 --dp 2 --context-length 36384",
        )
    """)
    result = force_affine_engine_args(code, gpu_count=8)
    assert result is not None
    assert "--tp 1" in result
    assert "--dp 8" in result
    assert result.count("--tp") == 1
    assert result.count("--dp") == 1


def test_force_no_tp_dp_without_gpu_count() -> None:
    code = textwrap.dedent("""
        from chutes.chute import NodeSelector
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            node_selector=NodeSelector(gpu_count=1, include=["pro_6000"]),
            engine_args="--context-length 36384",
        )
    """)
    result = force_affine_engine_args(code)
    assert result is not None
    assert "--tp" not in result
    assert "--dp" not in result


def test_force_gpu_count_1_sets_dp_1() -> None:
    code = textwrap.dedent("""
        from chutes.chute import NodeSelector
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            node_selector=NodeSelector(gpu_count=1, include=["pro_6000"]),
            engine_args="--context-length 36384",
        )
    """)
    result = force_affine_engine_args(code, gpu_count=1)
    assert result is not None
    assert "--tp 1" in result
    assert "--dp 1" in result


# --- transform_code_for_tee tests ---


def test_transform_replaces_node_selector_and_adds_tee() -> None:
    """Existing h100 node_selector should be replaced with pro_6000, tee=True added."""
    code = textwrap.dedent("""
        from chutes.chute import NodeSelector
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            node_selector=NodeSelector(gpu_count=2, include=["h100"]),
            engine_args="--tool-call-parser qwen25 --context-length 36384",
        )
    """)
    result = transform_code_for_tee(code, gpu_count=2, is_affine=True)
    assert result is not None
    # node_selector replaced
    assert 'include=["pro_6000"]' in result or "include=['pro_6000']" in result
    assert "h100" not in result
    assert "gpu_count=2" in result
    # tee added
    assert "tee=True" in result
    # engine_args: TP/DP forced
    assert "--tp 1" in result
    assert "--dp 2" in result
    # engine_args: mem/prefill forced
    assert "--mem-fraction-static 0.8" in result
    assert "--chunked-prefill-size 4096" in result
    # original flags preserved
    assert "--tool-call-parser" in result
    assert "--context-length" in result


def test_transform_replaces_node_selector_non_affine() -> None:
    """Non-affine subnet chute: node_selector replaced, tee added, no TP/DP changes."""
    code = textwrap.dedent("""
        from chutes.chute import NodeSelector
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/turbovision-test",
            image="chutes/vllm:nightly-2026010900",
            node_selector=NodeSelector(gpu_count=4, include=["a100"]),
            engine_args="--tool-call-parser qwen25",
        )
    """)
    result = transform_code_for_tee(code, gpu_count=4, is_affine=False)
    assert result is not None
    # node_selector replaced
    assert 'include=["pro_6000"]' in result or "include=['pro_6000']" in result
    assert "a100" not in result
    assert "gpu_count=4" in result
    # tee added
    assert "tee=True" in result
    # No TP/DP or mem forced (not affine)
    assert "--tensor-parallel-size" not in result
    assert "--data-parallel-size" not in result
    # original engine_args preserved as-is
    assert "--tool-call-parser qwen25" in result


def test_transform_overwrites_existing_tee_false() -> None:
    """If tee=False exists in code, it should be replaced with tee=True."""
    code = textwrap.dedent("""
        from chutes.chute import NodeSelector
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            node_selector=NodeSelector(gpu_count=8, include=["h200"]),
            tee=False,
            engine_args="--context-length 36384",
        )
    """)
    result = transform_code_for_tee(code, gpu_count=8, is_affine=True)
    assert result is not None
    assert "tee=True" in result
    assert "tee=False" not in result
    assert "h200" not in result
    assert 'include=["pro_6000"]' in result or "include=['pro_6000']" in result
    assert "--tp 1" in result
    assert "--dp 8" in result


def test_transform_affine_multi_gpu_replaces_tp() -> None:
    """Affine chute with existing TP=4: should be overwritten to TP=1, DP=gpu_count."""
    code = textwrap.dedent("""
        from chutes.chute import NodeSelector
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            node_selector=NodeSelector(gpu_count=4, min_vram_gb_per_gpu=80),
            engine_args="--tp 4 --context-length 36384",
        )
    """)
    result = transform_code_for_tee(code, gpu_count=4, is_affine=True)
    assert result is not None
    assert "--tp 1" in result
    assert "--dp 4" in result
    assert result.count("--tp") == 1
    assert "min_vram_gb_per_gpu" not in result
    assert 'include=["pro_6000"]' in result or "include=['pro_6000']" in result
