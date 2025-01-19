import asyncio
from typing import Optional
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

image = (
    Image(
        username="chutes",
        name="micro",
        tag="test0",
        readme="## Base image with cuda and python 3.12.7",
    )
    .from_base("parachutes/base-python:3.12.7")
    .add("parachute.png", "/app/parachute.png")
)

chute = Chute(
    username="test",
    name="example",
    readme="## Example Chute\n\n### Foo.\n\n```python\nprint('foo')```",
    image=image,
    node_selector=NodeSelector(
        gpu_count=1,
        # All options.
        # gpu_count: int = Field(1, ge=1, le=8)
        # min_vram_gb_per_gpu: int = Field(16, ge=16, le=80)
        # require_sxm: bool = False
        # include: Optional[List[str]] = None
        # exclude: Optional[List[str]] = None
    ),
)


class MicroArgs(BaseModel):
    foo: str = Field(..., max_length=100)
    bar: int = Field(0, gte=0, lte=100)
    baz: bool = False


class FullArgs(MicroArgs):
    bunny: Optional[str] = None
    giraffe: Optional[bool] = False
    zebra: Optional[int] = None


class ExampleOutput(BaseModel):
    foo: str
    bar: str
    baz: Optional[str]


@chute.on_startup()
async def initialize(self):
    self.billygoat = "billy"
    print("Inside the startup function!")


@chute.cord(minimal_input_schema=MicroArgs)
async def echo(self, input_args: FullArgs) -> str:
    return f"{chute.billygoat} says: {input_args}"


@chute.cord()
async def complex(self, input_args: MicroArgs) -> ExampleOutput:
    return ExampleOutput(foo=input_args.foo, bar=input_args.bar, baz=input_args.baz)


@chute.cord(
    output_content_type="image/png",
    public_api_path="/image",
    public_api_method="GET",
)
async def image(self) -> FileResponse:
    return FileResponse("parachute.png", media_type="image/png")


async def main():
    print(await echo("bar"))


if __name__ == "__main__":
    asyncio.run(main())
