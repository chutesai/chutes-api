import asyncio
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

image = (
    Image(username="test", name="micro", tag="0.0.1")
    .from_base("alpine:latest")
    .run_command("apk add python3 py3-pip shadow && ln -sf $(which python3) /usr/bin/python")
    .run_command("adduser chutes -D /home/chutes")
    .run_command("mkdir -p /app /home/chutes && chown chutes:chutes /home/chutes /app")
    .set_user("chutes")
    .set_workdir("/app")
    .run_command("python -m venv venv")
    .with_env("PATH", "/app/venv/bin:$PATH")
)

chute = Chute(
    username="test",
    name="micro",
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


@chute.on_startup()
async def initialize(self):
    self.billygoat = "billy"
    print("Inside the startup function!")


@chute.cord()
async def echo(input_str: str):
    return f"{chute.billygoat} says: {input_str}"


async def main():
    print(await echo("hello"))


if __name__ == "__main__":
    asyncio.run(main())
