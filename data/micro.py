import json
import asyncio
from chutes.image import Image
from chutes.chute import Chute, NodeSelector

image = (
    Image("micro", "0.0.1")
    .from_base("alpine:latest")
    .run_command("apk add python3 py3-pip shadow && ln -sf $(which python3) /usr/bin/python")
    .run_command("adduser chutes -D /home/chutes")
    .run_command("mkdir -p /home/chutes && chown chutes:chutes /home/chutes")
    .set_user("chutes")
    .set_workdir("/home/chutes")
    .run_command("python -m venv venv")
    .with_env("PATH", "/home/chutes/venv/bin:$PATH")
)

chute = Chute(
    name="micro",
    image=image,
    node_selector=NodeSelector(
        gpu_count=1,
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
