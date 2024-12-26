"""
GPU constants, expected values, etc.
"""

import math

# GPUs that are allowed to be passed in node_selector as "include" arg,
# because they are likely to be specified and somewhat widely available.
ALLOWED_INCLUDE = {
    "4090",
    "a6000",
    "l40s",
    "a100",
    "h100",
    "h100_sxm",
    "h200",
}
SUPPORTED_GPUS = {
    "3090": {
        "model_name_check": "RTX 3090",
        "memory": 24,
        "major": 8,
        "minor": 6,
        "tensor_cores": 328,
        "processors": 82,
        "clock_rate": {"base": 1395, "boost": 1695},
        "max_threads_per_processor": 1536,
        "concurrent_kernels": True,
        "ecc": False,
        "sxm": False,
    },
    "4090": {
        "model_name_check": "RTX 4090",
        "memory": 24,
        "major": 8,
        "minor": 9,
        "tensor_cores": 512,
        "processors": 128,
        "clock_rate": {"base": 2235, "boost": 2520},
        "max_threads_per_processor": 1536,
        "concurrent_kernels": True,
        "ecc": False,
        "sxm": False,
    },
    "a5000": {
        "model_name_check": "RTX A5000",
        "memory": 24,
        "major": 8,
        "minor": 6,
        "tensor_cores": 256,
        "processors": 72,
        "clock_rate": {"base": 1170, "boost": 1695},
        "max_threads_per_processor": 1536,
        "concurrent_kernels": True,
        "ecc": True,
        "sxm": False,
    },
    "a6000": {
        "model_name_check": "RTX A6000(?! Ada)",
        "memory": 48,
        "major": 8,
        "minor": 6,
        "tensor_cores": 336,
        "processors": 84,
        "clock_rate": {"base": 1455, "boost": 1860},
        "max_threads_per_processor": 1536,
        "concurrent_kernels": True,
        "ecc": True,
        "sxm": False,
    },
    "a6000_ada": {
        "model_name_check": "RTX A6000 Ada",
        "memory": 48,
        "major": 8,
        "minor": 9,
        "tensor_cores": 568,
        "processors": 142,
        "clock_rate": {"base": 915, "boost": 2505},
        "max_threads_per_processor": 1536,
        "concurrent_kernels": True,
        "ecc": True,
        "sxm": False,
    },
    "l4": {
        "model_name_check": "L4(?!0)",
        "memory": 24,
        "major": 8,
        "minor": 9,
        "tensor_cores": 240,
        "processors": 60,
        "clock_rate": {"base": 795, "boost": 2040},
        "max_threads_per_processor": 1536,
        "concurrent_kernels": True,
        "ecc": True,
        "sxm": False,
    },
    "t4": {
        "model_name_check": "T4",
        "memory": 16,
        "major": 7,
        "minor": 5,
        "tensor_cores": 320,
        "processors": 40,
        "clock_rate": {"base": 585, "boost": 1590},
        "max_threads_per_processor": 1024,
        "concurrent_kernels": True,
        "ecc": True,
        "sxm": False,
    },
    "a10": {
        "model_name_check": "A10",
        "memory": 24,
        "major": 8,
        "minor": 6,
        "tensor_cores": 208,
        "processors": 72,
        "clock_rate": {"base": 1110, "boost": 1710},
        "max_threads_per_processor": 1536,
        "concurrent_kernels": True,
        "ecc": True,
        "sxm": False,
    },
    "a30": {
        "model_name_check": "A30",
        "memory": 24,
        "major": 8,
        "minor": 0,
        "tensor_cores": 224,
        "processors": 56,
        "clock_rate": {"base": 930, "boost": 1440},
        "max_threads_per_processor": 1536,
        "concurrent_kernels": True,
        "ecc": True,
        "sxm": False,
    },
    "a40": {
        "model_name_check": "A40",
        "memory": 48,
        "major": 8,
        "minor": 6,
        "tensor_cores": 336,
        "processors": 84,
        "clock_rate": {"base": 1305, "boost": 1740},
        "max_threads_per_processor": 1536,
        "concurrent_kernels": True,
        "ecc": True,
        "sxm": False,
    },
    "l40": {
        "model_name_check": "L40(?!S)",
        "memory": 48,
        "major": 8,
        "minor": 9,
        "tensor_cores": 568,
        "processors": 142,
        "clock_rate": {"base": 735, "boost": 2490},
        "max_threads_per_processor": 1536,
        "concurrent_kernels": True,
        "ecc": True,
        "sxm": False,
    },
    "l40s": {
        "model_name_check": "L40S",
        "memory": 48,
        "major": 8,
        "minor": 9,
        "tensor_cores": 568,
        "processors": 142,
        "clock_rate": {"base": 1065, "boost": 2520},
        "max_threads_per_processor": 1536,
        "concurrent_kernels": True,
        "ecc": True,
        "sxm": False,
    },
    "a100_40gb": {
        "model_name_check": "A100.?PCIE.?40GB",
        "memory": 40,
        "major": 8,
        "minor": 0,
        "tensor_cores": 432,
        "processors": 108,
        "clock_rate": {"base": 1065, "boost": 1410},
        "max_threads_per_processor": 2048,
        "concurrent_kernels": True,
        "ecc": True,
        "sxm": False,
    },
    "a100_40gb_sxm": {
        "model_name_check": "A100.?SXM.?40GB",
        "memory": 40,
        "major": 8,
        "minor": 0,
        "tensor_cores": 432,
        "processors": 108,
        "clock_rate": {"base": 1065, "boost": 1410},
        "max_threads_per_processor": 2048,
        "concurrent_kernels": True,
        "ecc": True,
        "sxm": True,
    },
    "a100": {
        "model_name_check": "A100.?80GB.?PCIe",
        "memory": 80,
        "major": 8,
        "minor": 0,
        "tensor_cores": 432,
        "processors": 108,
        "clock_rate": {"base": 1065, "boost": 1410},
        "max_threads_per_processor": 2048,
        "concurrent_kernels": True,
        "ecc": True,
        "sxm": False,
    },
    "a100_sxm": {
        "model_name_check": "A100.?SXM.?.?80GB",
        "memory": 80,
        "major": 8,
        "minor": 0,
        "tensor_cores": 432,
        "processors": 108,
        "clock_rate": {"base": 1275, "boost": 1410},
        "max_threads_per_processor": 2048,
        "concurrent_kernels": True,
        "ecc": True,
        "sxm": True,
    },
    "h100": {
        "model_name_check": "H100.*PCIe",
        "memory": 80,
        "major": 9,
        "minor": 0,
        "tensor_cores": 456,
        "processors": 114,
        "clock_rate": {"base": 1095, "boost": 1755},
        "max_threads_per_processor": 2048,
        "concurrent_kernels": True,
        "ecc": True,
        "sxm": False,
    },
    "h100_sxm": {
        "model_name_check": "H100.*HBM3",
        "memory": 80,
        "major": 9,
        "minor": 0,
        "tensor_cores": 528,
        "processors": 132,
        "clock_rate": {"base": 1590, "boost": 1980},
        "max_threads_per_processor": 2048,
        "concurrent_kernels": True,
        "ecc": True,
        "sxm": True,
    },
    "h200": {
        "model_name_check": " H200",
        "memory": 140,
        "major": 9,
        "minor": 0,
        "tensor_cores": 528,
        "processors": 132,
        "clock_rate": {"base": 1590, "boost": 1980},
        "max_threads_per_processor": 2048,
        "concurrent_kernels": True,
        "ecc": True,
        "sxm": True,
    },
}


def calculate_gpu_boost(gpu_info):
    """
    Generate a boost score for a particular GPU, somewhat attempting to
    reflect both pricing and real-world performance.
    """
    arch_base_multiplier = 1.3 ** (gpu_info["major"] - 7)
    arch_specific_multiplier = {
        # Hopper
        9: 1.4,
        # Ampere
        8: {
            0: 1.3,
            6: 1.0,
            9: 1.1,
        }.get(gpu_info["minor"], 1.0),
        # Turing
        7: 0.9,
    }.get(gpu_info["major"], 1.0)

    memory_multiplier = {
        9: 1.4,
        8: 1.3 if gpu_info["minor"] == 0 else 1.1,
        7: 1.0,
    }.get(gpu_info["major"], 1.0)

    memory_score = math.log2(gpu_info["memory"]) * memory_multiplier

    tensor_core_multiplier = {
        7: 0.6,
        8: {
            0: 1.3,
            6: 1.0,
            9: 1.1,
        }.get(gpu_info["minor"], 1.0),
        9: 1.5,
    }.get(gpu_info["major"], 1.0)

    tensor_score = (gpu_info["tensor_cores"] / 100) * tensor_core_multiplier

    clock_importance = 0.85 if gpu_info["major"] >= 8 and gpu_info["ecc"] else 1.0
    clock_score = (gpu_info["clock_rate"]["boost"] / 1000) * clock_importance

    ecc_multiplier = 1.15 if gpu_info["ecc"] else 1.0
    sxm_multiplier = 1.2 if gpu_info["sxm"] else 1.0

    datacenter_multiplier = 1.2 if (gpu_info["ecc"] and gpu_info["tensor_cores"] >= 300) else 1.0

    score = (
        arch_base_multiplier
        * arch_specific_multiplier
        * (memory_score + tensor_score + clock_score)
        * ecc_multiplier
        * sxm_multiplier
        * datacenter_multiplier
    )

    return score


def normalize_scores(gpu_dict):
    """
    Calculate score for each supported GPU and provide a normalized range.
    """
    scores = {}
    for model, info in gpu_dict.items():
        score = calculate_gpu_boost(info)
        scores[model] = score
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    max_score = max(scores.values())
    normalized_scores = {}
    for model, score in sorted_scores.items():
        normalized_scores[model] = score / max_score
    return normalized_scores


COMPUTE_MULTIPLIER = normalize_scores(SUPPORTED_GPUS)
COMPUTE_MIN = min(list(COMPUTE_MULTIPLIER.values()))
