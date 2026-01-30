"""
TEE-ify transformation for affine chutes.

This module provides AST-based code transformation to convert affine chutes
to TEE-enabled versions with H200 GPU requirements.
"""

import ast
import math
from api.gpu import SUPPORTED_GPUS
from api.chute.schemas import NodeSelector


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 2 ** math.ceil(math.log2(n))


def _calculate_h200_gpu_count(original_gpu_count: int, original_node_selector: dict) -> int:
    """
    Calculate the required H200 GPU count to match original VRAM requirements.

    H200 has 140GB VRAM per GPU. The result is the smallest power of 2 that
    provides at least as much total VRAM as the original configuration,
    capped at 8 GPUs maximum.

    The original min VRAM per GPU is determined by taking the minimum VRAM
    of all GPUs in the supported_gpus list for the original node_selector.
    """
    h200_vram = SUPPORTED_GPUS["h200"]["memory"]  # 140GB

    # Get the supported GPUs from the original node selector
    ns = NodeSelector(**original_node_selector)
    supported = ns.supported_gpus

    if not supported:
        # Fallback if no supported GPUs (shouldn't happen)
        min_vram_per_gpu = 80
    else:
        # Get the minimum VRAM from the supported GPUs list
        min_vram_per_gpu = min(SUPPORTED_GPUS[gpu]["memory"] for gpu in supported)

    original_total_vram = original_gpu_count * min_vram_per_gpu
    required_gpus = math.ceil(original_total_vram / h200_vram)
    # Cap at 8 GPUs maximum
    return min(_next_power_of_2(required_gpus), 8)


class TeeifyTransformer(ast.NodeTransformer):
    """
    AST transformer to convert affine chute code for TEE deployment.

    Transforms:
    - node_selector to include=["h200"] with calculated gpu_count
    - Adds tee=True to the builder call
    """

    def __init__(self, new_gpu_count: int):
        self.new_gpu_count = new_gpu_count

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """Transform builder calls (build_sglang_chute, build_vllm_chute, Chute)."""
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name not in ("build_sglang_chute", "build_vllm_chute", "Chute"):
            return self.generic_visit(node)

        new_keywords = []
        has_tee = False
        has_node_selector = False

        for keyword in node.keywords:
            if keyword.arg == "node_selector":
                has_node_selector = True
                # Create new NodeSelector with include=["h200"] and updated gpu_count
                new_node_selector = ast.Call(
                    func=ast.Name(id="NodeSelector", ctx=ast.Load()),
                    args=[],
                    keywords=[
                        ast.keyword(
                            arg="gpu_count",
                            value=ast.Constant(value=self.new_gpu_count),
                        ),
                        ast.keyword(
                            arg="include",
                            value=ast.List(
                                elts=[ast.Constant(value="h200")],
                                ctx=ast.Load(),
                            ),
                        ),
                    ],
                )
                new_keywords.append(ast.keyword(arg="node_selector", value=new_node_selector))
            elif keyword.arg == "tee":
                has_tee = True
                # Force tee=True
                new_keywords.append(ast.keyword(arg="tee", value=ast.Constant(value=True)))
            else:
                new_keywords.append(keyword)

        # Add tee=True if not present
        if not has_tee:
            new_keywords.append(ast.keyword(arg="tee", value=ast.Constant(value=True)))

        # Add node_selector if not present
        if not has_node_selector:
            new_node_selector = ast.Call(
                func=ast.Name(id="NodeSelector", ctx=ast.Load()),
                args=[],
                keywords=[
                    ast.keyword(
                        arg="gpu_count",
                        value=ast.Constant(value=self.new_gpu_count),
                    ),
                    ast.keyword(
                        arg="include",
                        value=ast.List(
                            elts=[ast.Constant(value="h200")],
                            ctx=ast.Load(),
                        ),
                    ),
                ],
            )
            new_keywords.append(ast.keyword(arg="node_selector", value=new_node_selector))

        node.keywords = new_keywords
        return node


def transform_for_tee(code: str, original_node_selector: dict) -> tuple[str, dict]:
    """
    Transform affine chute code and node_selector for TEE deployment.

    Changes:
    1. node_selector becomes include=["h200"] with appropriate gpu_count
    2. tee=True is added/set

    Returns:
        Tuple of (transformed_code, new_node_selector_dict)
    """
    gpu_count = original_node_selector.get("gpu_count", 1)

    # Calculate new H200 GPU count based on original VRAM requirements
    new_gpu_count = _calculate_h200_gpu_count(gpu_count, original_node_selector)

    # Parse and transform the AST
    tree = ast.parse(code)
    transformer = TeeifyTransformer(new_gpu_count)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    # Convert back to code
    transformed_code = ast.unparse(tree)

    # Return both the code and the new node_selector
    new_node_selector = {"gpu_count": new_gpu_count, "include": ["h200"]}

    return transformed_code, new_node_selector
