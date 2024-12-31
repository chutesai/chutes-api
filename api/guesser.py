import aiohttp
import math
import json
from fastapi import APIRouter, HTTPException
from typing import Dict, Optional
from pydantic import BaseModel
from api.gpu import SUPPORTED_GPUS

router = APIRouter()


class GPURequirements(BaseModel):
    total_model_size: int
    required_gpus: int
    min_vram_per_gpu: int
    model_type: str
    quantization: Optional[str]
    num_attention_heads: int
    num_key_value_heads: Optional[int]
    hidden_size: int
    num_layers: int


class ConfigGuesser:
    def __init__(self):
        self.available_vram_sizes = sorted(
            set(gpu["memory"] for gpu in SUPPORTED_GPUS.values() if gpu["memory"] <= 80)
        )

        self.vram_overhead = {
            "llama": 2,
            "mistral": 2,
            "phi": 1.5,
            "falcon": 2,
            "gpt_neox": 2,
            "mpt": 2,
            "qwen": 2,
            "default": 2,
        }

        self.quant_multipliers = {"4bit": 0.25, "8bit": 0.5, "none": 1.0}

    def _get_min_gpu_config(self, total_vram: float, config: Dict) -> tuple[int, int]:
        """
        Calculate minimum number of GPUs needed and VRAM per GPU.
        """
        num_attention_heads = config.get("num_attention_heads", 0)
        num_kv_heads = config.get("num_key_value_heads", num_attention_heads)
        hidden_size = config.get("hidden_size", 0)

        best_gpu_count = float("inf")
        best_vram_size = float("inf")

        for gpu_count in range(1, 9):
            if (
                (num_attention_heads and num_attention_heads % gpu_count != 0)
                or (num_kv_heads and num_kv_heads % gpu_count != 0)
                or (hidden_size and hidden_size % gpu_count != 0)
            ):
                continue

            vram_per_gpu = math.ceil(total_vram / gpu_count)
            for vram_size in self.available_vram_sizes:
                if vram_size >= vram_per_gpu:
                    if gpu_count < best_gpu_count or (
                        gpu_count == best_gpu_count and vram_size < best_vram_size
                    ):
                        best_gpu_count = gpu_count
                        best_vram_size = vram_size
                    break

        if best_gpu_count == float("inf"):
            raise ValueError("Could not find a valid GPU configuration")
        return best_gpu_count, best_vram_size

    def _detect_model_type(self, config: Dict) -> str:
        """
        Detects the model architecture type from config.
        """
        model_type = config.get("model_type", "").lower()

        for arch in self.vram_overhead.keys():
            if arch in model_type:
                return arch
        return "default"

    def _detect_quantization(self, config: Dict) -> Optional[str]:
        """
        Detects if model is quantized and what format.
        """
        if config.get("quantization_config"):
            bits = config["quantization_config"].get("bits", 16)
            if bits == 4:
                return "4bit"
            elif bits == 8:
                return "8bit"
        return "none"

    async def analyze_model(self, repo_id: str, session: aiohttp.ClientSession) -> GPURequirements:
        """
        Analyzes a HuggingFace model to determine deployment requirements.
        """
        config_url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
        async with session.get(config_url) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=404, detail=f"Could not fetch config.json for {repo_id}"
                )
            try:
                config = await response.json()
            except Exception:
                config = json.loads(await response.text())

        safetensors_map_url = (
            f"https://huggingface.co/{repo_id}/raw/main/model.safetensors.index.json"
        )
        try:
            async with session.get(safetensors_map_url) as response:
                if response.status == 200:
                    try:
                        safetensors_map = await response.json()
                    except Exception:
                        safetensors_map = json.loads(await response.text())
                    total_size = safetensors_map.get("metadata", {}).get("total_size", 0)
                    if not total_size:
                        total_size = 0
                        for filename in set(safetensors_map["weight_map"].values()):
                            file_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
                            async with session.head(file_url) as head_response:
                                if 200 <= head_response.status < 400:
                                    size = int(head_response.headers.get("x-linked-size", 0))
                                    total_size += size
                else:
                    num_params = config.get("num_parameters", 0)
                    if not num_params and config.get("architectures"):
                        hidden_size = config.get("hidden_size", 0)
                        num_layers = config.get("num_hidden_layers", 0)
                        vocab_size = config.get("vocab_size", 0)
                        num_params = (hidden_size * hidden_size * 4 * num_layers) + (
                            vocab_size * hidden_size
                        )
                    total_size = num_params * 2
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error estimating model size: {str(e)}")

        total_size = math.ceil(total_size / (1024**3))
        model_type = self._detect_model_type(config)
        quantization = self._detect_quantization(config)

        base_vram = total_size + self.vram_overhead.get(model_type, self.vram_overhead["default"])
        if quantization:
            base_vram *= self.quant_multipliers.get(quantization, 1.0)
        try:
            num_gpus, vram_per_gpu = self._get_min_gpu_config(base_vram, config)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return GPURequirements(
            total_model_size=total_size,
            required_gpus=num_gpus,
            min_vram_per_gpu=vram_per_gpu,
            model_type=model_type,
            quantization=quantization,
            num_attention_heads=config.get("num_attention_heads", 0),
            num_key_value_heads=config.get("num_key_value_heads"),
            hidden_size=config.get("hidden_size", 0),
            num_layers=config.get("num_hidden_layers", 0),
        )


guesser = ConfigGuesser()


@router.get("/vllm_config", response_model=GPURequirements)
async def analyze_model(model: str):
    """
    Attempt to guess required GPU count and VRAM for a model on huggingface, assuming safetensors format.
    """
    async with aiohttp.ClientSession() as session:
        try:
            return await guesser.analyze_model(model, session)
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
