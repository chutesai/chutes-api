"""
Args for templated chutes.
"""

import re
from typing import List, Literal
from pydantic import BaseModel, validator, Field
from typing import Optional
from jinja2 import Environment, select_autoescape
from api.chute.schemas import NodeSelector
from chutes.chute.template.vllm import build_vllm_chute
from chutes.chute.template.diffusion import build_diffusion_chute
from chutes.chute.template.tei import build_tei_chute


env = Environment(autoescape=select_autoescape(["html", "xml"]))


class VLLMEngineArgs(BaseModel):
    tokenizer: Optional[str] = None
    max_model_len: Optional[int] = 16384
    enforce_eager: Optional[bool] = False
    num_scheduler_steps: Optional[int] = 16
    trust_remote_code: Optional[bool] = True

    @validator("tokenizer")
    def validate_hf_format(cls, v):
        if v is None:
            return v
        if re.match(r"^[a-zA-Z0-9_\.-]+/[a-zA-Z0-9_\.-]+$", v):
            return v
        raise ValueError('Model must be a valid Hugging Face repo (e.g., "org/model")')


class VLLMChuteArgs(BaseModel):
    model: str
    logo_id: Optional[str] = None
    tagline: Optional[str] = ""
    tool_description: Optional[str] = None
    readme: Optional[str] = ""
    public: Optional[bool] = True
    node_selector: Optional[NodeSelector] = None
    engine_args: Optional[VLLMEngineArgs] = None

    @validator("model")
    def validate_hf_format(cls, v):
        if re.match(r"^[a-zA-Z0-9_\.-]+/[a-zA-Z0-9_\.-]+$", v):
            return v
        raise ValueError('Model must be a valid Hugging Face repo (e.g., "org/model")')


class DiffusionPipelineArgs(BaseModel):
    use_safetensors: Optional[bool]
    variant: Optional[str]
    revision: Optional[str]


class DiffusionChuteArgs(BaseModel):
    model: str
    name: str
    logo_id: Optional[str] = None
    tagline: Optional[str] = ""
    tool_description: Optional[str] = None
    readme: Optional[str] = ""
    public: Optional[bool] = True
    node_selector: Optional[NodeSelector] = None

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
        if re.match(r"^[a-zA-Z0-9_\. -/]+$", v):
            return v
        raise ValueError(
            'Chute name must only contain letters, numbers, ".", "-", "_", spaces or "/"'
        )


class TEIChuteArgs(BaseModel):
    model: str
    endpoints: List[Literal["embed", "predict", "rerank"]] = Field(
        description="List of supported endpoints for this chute",
        min_items=1,
    )
    revision: Optional[str] = None
    logo_id: Optional[str] = None
    tagline: Optional[str] = ""
    tool_description: Optional[str] = None
    readme: Optional[str] = ""
    public: Optional[bool] = True
    node_selector: Optional[NodeSelector] = None

    @validator("model")
    def validate_model(cls, v):
        hf_pattern = r"^[a-zA-Z0-9_\.-]+/[a-zA-Z0-9_\.-]+$"
        if re.match(hf_pattern, v):
            return v
        raise ValueError('Model must be a valid Hugging Face model name (e.g., "org/model")')


VLLM_TEMPLATE = """from chutes.chute import NodeSelector
from chutes.chute.template.vllm import build_vllm_chute

chute = build_vllm_chute(
    username="{{ username }}",
    model_name="{{ args.model }}",
    image="{{ image }}",
    node_selector=NodeSelector(),
    {%- if args.engine_args %}
    engine_args=dict(
        {%- if args.engine_args.tokenizer is not none %}
        tokenizer="{{ args.engine_args.tokenizer }}",
        {%- endif %}
        {%- if args.engine_args.max_model_len is not none %}
        max_model_len={{ args.engine_args.max_model_len }},
        {%- endif %}
        {%- if args.engine_args.trust_remote_code is not none %}
        trust_remote_code={{ args.engine_args.trust_remote_code }},
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
    node_selector=NodeSelector(),
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

TEI_TEMPLATE = """from chutes.chute import NodeSelector
from chutes.chute.template.tei import build_tei_chute

chute = build_tei_chute(
    username="{{ username }}",
    model_name="{{ args.model }}",
    endpoints={{ args.endpoints | tojson }},
    image="{{ image }}",
    node_selector=NodeSelector(),
    {%- if args.revision %}
    revision="{{ args.revision }}",
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
        node_selector=NodeSelector(),
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
        node_selector=NodeSelector(),
    )
    return code, chute


def build_tei_code(args: TEIChuteArgs, username: str, image: str) -> str:
    """
    Builds a Python script for text-embeddings-inference chute creation using Jinja templating.
    """
    template = env.from_string(TEI_TEMPLATE)
    code = template.render(args=args, username=username, image=image)
    chute = build_tei_chute(
        username=username,
        model_name=args.model,
        endpoints=args.endpoints,
        image=image,
        node_selector=NodeSelector(),
        revision=args.revision,
    )
    return code, chute
