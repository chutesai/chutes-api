"""
Args for templated chutes.
"""

import re
from pydantic import BaseModel, validator
from typing import Optional
from jinja2 import Environment, PackageLoader, select_autoescape
from api.chutes.schemas import NodeSelector
from chutes.chute.template.vllm import build_vllm_chute
from chutes.chute.template.diffusion import build_diffusion_chute


env = Environment(
    loader=PackageLoader("chutes", "templates"), autoescape=select_autoescape(["html", "xml"])
)


class VLLMEngineArgs(BaseModel):
    max_model_len: Optional[int]
    enforce_eager: Optional[bool]
    num_scheduler_steps: Optional[int]


class VLLMChuteArgs(BaseModel):
    model: str
    tokenizer: Optional[str]
    logo_id: Optional[str]
    readme: Optional[str]
    public: Optional[bool] = True
    node_selector: Optional[NodeSelector]
    engine_args: Optional[VLLMEngineArgs]

    @validator("model")
    def validate_model(cls, v):
        hf_pattern = r"^[a-zA-Z0-9_\.-]+/[a-zA-Z0-9_\.-]+$"
        if re.match(hf_pattern, v):
            return v
        raise ValueError('Model must be either a valid Hugging Face model name (e.g., "org/model")')


class DiffusionPipelineArgs(BaseModel):
    use_safetensors: Optional[bool]
    variant: Optional[str]
    revision: Optional[str]


class DiffusionChuteArgs(BaseModel):
    model: str
    name: str
    logo_id: Optional[str]
    readme: Optional[str]
    public: Optional[bool] = True
    node_selector: Optional[NodeSelector]

    @validator("model")
    def validate_model(cls, v):
        hf_pattern = r"^[a-zA-Z0-9_\.-]+/[a-zA-Z0-9_\.-]+$"
        civitai_pattern = r"^https://civitai\.com/models/\d+(?:/[^/]*)?$"
        if re.match(hf_pattern, v) or re.match(civitai_pattern, v):
            return v
        raise ValueError(
            'Model must be either a valid Hugging Face model name (e.g., "org/model") '
            'or a CivitAI model URL (e.g., "https://civitai.com/models/133005")'
        )

    @validator("name")
    def validate_name(cls, v):
        if re.match(r"^[a-zA-Z0-9_\. -/]+%", v):
            return v
        raise ValueError(
            'Chute name must only contain letters, numbers, ".", "-", "_", spaces or "/"'
        )


VLLM_TEMPLATE = """from chutes.chute import NodeSelector
from chutes.chute.template.vllm import build_vllm_chute

chute = build_vllm_chute(
    username="{{ username }}",
    model_name="{{ args.model }}",
    image="{{ image }}",
    {%- if args.node_selector %}
    node_selector=NodeSelector({{ args.node_selector.model_dump() }}),
    {%- endif %}
    {%- if args.engine_args %}
    engine_args=dict(
        {%- if args.engine_args.max_model_len is not none %}
        max_model_len={{ args.engine_args.max_model_len }},
        {%- endif %}
        {%- if args.engine_args.enforce_eager is not none %}
        enforce_eager={{ args.engine_args.enforce_eager }},
        {%- endif %}
        {%- if args.engine_args.num_scheduler_steps is not none %}
        num_scheduler_steps={{ args.engine_args.num_scheduler_steps }},
        {%- endif %}
    )
    {%- endif %}
)"""

DIFFUSION_TEMPLATE = """from chutes.chute import NodeSelector
from chutes.chute.template.diffusion import build_diffusion_chute

chute = build_diffusion_chute(
    username="{{ username }}",
    name="{{ args.name }}",
    model_name_or_url="{{ args.model }}",
    image="{{ image }}",
    {%- if args.node_selector %}
    node_selector=NodeSelector({{ args.node_selector.model_dump() }}),
    {%- endif %}
    {%- if args.pipeline_args %}
    pipeline_args=dict(
        {%- if args.pipeline_args.use_safetensors is not none %}
        use_safetensors={{ args.pipeline_args.use_safetensors }},
        {%- endif %}
        {%- if args.pipeline_args.variant %}
        variant="{{ args.pipeline_args.variant }}",
        {%- endif %}
        {%- if args.pipeline_args.revision %}
        revision="{{ args.pipeline_args.revision }}",
        {%- endif %}
    )
    {%- endif %}
)"""


def build_vllm_code(args: VLLMChuteArgs, username: str, image: str) -> str:
    """
    Builds a Python script for VLLM chute creation using Jinja templating.
    """
    template = env.from_string(VLLM_TEMPLATE)
    code = template.render(args=args, username=username, image=image)
    chute = build_vllm_chute(
        username=username,
        model_name=args.model,
        image=image,
    )
    return code, chute


def build_diffusion_code(args: DiffusionChuteArgs, username: str, image: str) -> str:
    """
    Builds a Python script for Diffusion chute creation using Jinja templating.
    """
    template = env.from_string(DIFFUSION_TEMPLATE)
    code = template.render(args=args, username=username, image=image)
    chute = build_diffusion_chute(
        username=username,
        name=args.name,
        model_name_or_url=args.model,
        image=image,
    )
    return code, chute
