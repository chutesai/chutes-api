"""
Tests for the TEE-ify transformation logic.
"""

import pytest
from api.chute.teeify import (
    transform_for_tee,
    _calculate_h200_gpu_count,
    _next_power_of_2,
)


class TestNextPowerOf2:
    def test_already_power_of_2(self):
        assert _next_power_of_2(1) == 1
        assert _next_power_of_2(2) == 2
        assert _next_power_of_2(4) == 4
        assert _next_power_of_2(8) == 8

    def test_not_power_of_2(self):
        assert _next_power_of_2(3) == 4
        assert _next_power_of_2(5) == 8
        assert _next_power_of_2(6) == 8
        assert _next_power_of_2(7) == 8

    def test_zero_and_negative(self):
        assert _next_power_of_2(0) == 1
        assert _next_power_of_2(-1) == 1


class TestCalculateH200GpuCount:
    def test_single_a100_gpu(self):
        """1 x A100 (80GB) -> ceil(80/140) = 1 GPU, power of 2 = 1"""
        node_selector = {"gpu_count": 1, "include": ["a100"]}
        result = _calculate_h200_gpu_count(1, node_selector)
        assert result == 1

    def test_single_gpu_mixed_include(self):
        """1 GPU with mixed include (a100, h100, h100_sxm, h100_nvl, h200).
        Min VRAM is 80GB (a100/h100), so 80/140 = 1 GPU."""
        node_selector = {
            "gpu_count": 1,
            "include": ["a100", "h100", "h100_sxm", "h100_nvl", "h200"],
        }
        result = _calculate_h200_gpu_count(1, node_selector)
        assert result == 1

    def test_two_a100_gpus(self):
        """2 x A100 (80GB each) = 160GB total -> ceil(160/140) = 2 GPUs"""
        node_selector = {"gpu_count": 2, "include": ["a100"]}
        result = _calculate_h200_gpu_count(2, node_selector)
        assert result == 2

    def test_four_a100_gpus(self):
        """4 x A100 (80GB each) = 320GB total -> ceil(320/140) = 3, power of 2 = 4"""
        node_selector = {"gpu_count": 4, "include": ["a100"]}
        result = _calculate_h200_gpu_count(4, node_selector)
        assert result == 4

    def test_eight_a100_gpus(self):
        """8 x A100 (80GB each) = 640GB total -> ceil(640/140) = 5, power of 2 = 8"""
        node_selector = {"gpu_count": 8, "include": ["a100"]}
        result = _calculate_h200_gpu_count(8, node_selector)
        assert result == 8

    def test_eight_h200_gpus_stays_at_8(self):
        """8 x H200 (140GB each) = 1120GB total -> ceil(1120/140) = 8"""
        node_selector = {"gpu_count": 8, "include": ["h200"]}
        result = _calculate_h200_gpu_count(8, node_selector)
        assert result == 8

    def test_eight_b200_gpus_capped_at_8(self):
        """8 x B200 (192GB each) = 1536GB total -> ceil(1536/140) = 11, but capped at 8"""
        node_selector = {"gpu_count": 8, "include": ["b200"]}
        result = _calculate_h200_gpu_count(8, node_selector)
        assert result == 8

    def test_min_vram_based_selector(self):
        """Node selector with min_vram_gb_per_gpu instead of include.
        min_vram=80 means supported GPUs include a100, h100, h200, etc.
        The minimum VRAM among supported GPUs determines the calculation."""
        node_selector = {"gpu_count": 2, "min_vram_gb_per_gpu": 80}
        result = _calculate_h200_gpu_count(2, node_selector)
        # With min_vram=80, supported GPUs have min VRAM of 80GB
        # 2 * 80 = 160GB total -> ceil(160/140) = 2
        assert result == 2


class TestTransformForTee:
    def test_basic_sglang_transformation(self):
        """Test basic SGLang chute transformation."""
        code = """
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute

chute = build_sglang_chute(
    username="dumbeldore",
    readme="rubizinho/Affine-small-model",
    model_name="rubizinho/Affine-small-model",
    image="chutes/sglang:nightly-20251212000",
    concurrency=20,
    revision="752156972da0be5f04201c5570c0f81db29ef558",
    node_selector=NodeSelector(
        gpu_count=1,
        include=["a100"],
    ),
    scaling_threshold=0.5,
    max_instances=4,
    shutdown_after_seconds=345600,
)
"""
        node_selector = {"gpu_count": 1, "include": ["a100"]}
        result_code, result_ns = transform_for_tee(code, node_selector)

        # Check that tee=True is added
        assert "tee=True" in result_code

        # Check that node_selector has include=["h200"]
        assert 'include=["h200"]' in result_code or "include=['h200']" in result_code

        # Check that other params are preserved
        assert "username=" in result_code
        assert "model_name=" in result_code
        assert "concurrency=20" in result_code
        assert "revision=" in result_code

        # Check returned node_selector
        assert result_ns == {"gpu_count": 1, "include": ["h200"]}

    def test_mixed_gpu_include_transformation(self):
        """Test transformation with mixed GPU includes."""
        code = """
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute

chute = build_sglang_chute(
    username="tony12345",
    readme="silentfly/Affine-qwen0121-5FgjroARbfhQjRDSgqrBwPziLANQ6dGgs3cJqdrdnwgy94Z7",
    model_name="silentfly/Affine-qwen0121-5FgjroARbfhQjRDSgqrBwPziLANQ6dGgs3cJqdrdnwgy94Z7",
    image="chutes/sglang:nightly-20251212000",
    concurrency=24,
    revision="10ac7393a1890410fc7e268bd0a16de4a00af5b6",
    node_selector=NodeSelector(
        gpu_count=1,
        include=["a100", "h100", "h100_sxm", "h100_nvl", "h200"],
    ),
    scaling_threshold=0.7,
    max_instances=1,
    shutdown_after_seconds=28800,
)
"""
        node_selector = {
            "gpu_count": 1,
            "include": ["a100", "h100", "h100_sxm", "h100_nvl", "h200"],
        }
        result_code, result_ns = transform_for_tee(code, node_selector)

        # Check that tee=True is added
        assert "tee=True" in result_code

        # Check that node_selector now only includes h200
        assert 'include=["h200"]' in result_code or "include=['h200']" in result_code

        # Old includes should not be present
        assert "a100" not in result_code or 'include=["h200"]' in result_code

        # Check returned node_selector
        assert result_ns == {"gpu_count": 1, "include": ["h200"]}

    def test_preserves_other_keywords(self):
        """Test that other keywords are preserved during transformation."""
        code = """
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute

chute = build_sglang_chute(
    username="tony12345",
    readme="jessica0911/Affine-qwen1231",
    model_name="jessica0911/Affine-qwen1231",
    image="chutes/sglang:nightly-20251212000",
    concurrency=24,
    revision="c4a3168c74972c3227a9419f0e12900b56d9b520",
    node_selector=NodeSelector(
        gpu_count=1,
        include=["a100", "h100", "h100_sxm", "h100_nvl", "h200"],
    ),
    scaling_threshold=0.7,
    max_instances=1,
    shutdown_after_seconds=28800,
)
"""
        node_selector = {
            "gpu_count": 1,
            "include": ["a100", "h100", "h100_sxm", "h100_nvl", "h200"],
        }
        result_code, _ = transform_for_tee(code, node_selector)

        # Check preserved keywords
        assert "concurrency=24" in result_code
        assert "scaling_threshold=0.7" in result_code
        assert "max_instances=1" in result_code
        assert "shutdown_after_seconds=28800" in result_code
        # AST unparser may use single or double quotes
        assert "c4a3168c74972c3227a9419f0e12900b56d9b520" in result_code

    def test_vllm_chute_transformation(self):
        """Test transformation of vLLM chute."""
        code = """
from chutes.chute import NodeSelector
from chutes.chute.template.vllm import build_vllm_chute

chute = build_vllm_chute(
    username="testuser",
    model_name="test/Affine-model",
    image="chutes/vllm:nightly-2026012000",
    concurrency=40,
    revision="abcdef1234567890abcdef1234567890abcdef12",
    node_selector=NodeSelector(
        gpu_count=2,
        include=["h100"],
    ),
)
"""
        node_selector = {"gpu_count": 2, "include": ["h100"]}
        result_code, result_ns = transform_for_tee(code, node_selector)

        assert "tee=True" in result_code
        assert 'include=["h200"]' in result_code or "include=['h200']" in result_code
        # 2 H100 (80GB each) = 160GB, ceil(160/140) = 2
        assert result_ns == {"gpu_count": 2, "include": ["h200"]}

    def test_multi_gpu_calculation(self):
        """Test that multi-GPU configurations calculate correct H200 count."""
        code = """
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute

chute = build_sglang_chute(
    username="testuser",
    model_name="test/Affine-large-model",
    image="chutes/sglang:nightly-20251212000",
    node_selector=NodeSelector(
        gpu_count=4,
        include=["a100"],
    ),
)
"""
        # 4 x A100 (80GB) = 320GB total
        # H200 has 140GB, so ceil(320/140) = 3, next power of 2 = 4
        node_selector = {"gpu_count": 4, "include": ["a100"]}
        result_code, result_ns = transform_for_tee(code, node_selector)

        assert "gpu_count=4" in result_code
        assert "tee=True" in result_code
        assert result_ns == {"gpu_count": 4, "include": ["h200"]}

    def test_code_with_os_import(self):
        """Test transformation preserves os import."""
        code = """
import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute

chute = build_sglang_chute(
    username="dumbeldore",
    readme="rubizinho/Affine-small-model",
    model_name="rubizinho/Affine-small-model",
    image="chutes/sglang:nightly-20251212000",
    concurrency=20,
    revision="752156972da0be5f04201c5570c0f81db29ef558",
    node_selector=NodeSelector(
        gpu_count=1,
        include=["a100"],
    ),
    scaling_threshold=0.5,
    max_instances=4,
    shutdown_after_seconds=345600,
)
"""
        node_selector = {"gpu_count": 1, "include": ["a100"]}
        result_code, _ = transform_for_tee(code, node_selector)

        # os import should be preserved
        assert "import os" in result_code

    def test_existing_tee_false_becomes_true(self):
        """Test that existing tee=False is changed to tee=True."""
        code = """
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute

chute = build_sglang_chute(
    username="testuser",
    model_name="test/Affine-model",
    image="chutes/sglang:nightly-20251212000",
    node_selector=NodeSelector(
        gpu_count=1,
        include=["a100"],
    ),
    tee=False,
)
"""
        node_selector = {"gpu_count": 1, "include": ["a100"]}
        result_code, _ = transform_for_tee(code, node_selector)

        assert "tee=True" in result_code
        assert "tee=False" not in result_code
