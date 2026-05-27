import api.database.orms  # noqa
import sys
import asyncio
import httpx
from api.database import get_session
from sqlalchemy import select
from api.instance.schemas import Instance
import api.miner_client as miner_client


async def get_logs(
    instance_id: str = None, host: str = None, port: int = None, miner_hotkey: str = None
):
    if host and port:
        if not miner_hotkey:
            if not instance_id:
                print("Must provide --miner-hotkey or --instance-id when using --host/--port")
                sys.exit(1)
            async with get_session() as session:
                instance = (
                    (
                        await session.execute(
                            select(Instance).where(Instance.instance_id == instance_id)
                        )
                    )
                    .unique()
                    .scalar_one_or_none()
                )
                miner_hotkey = instance.miner_hotkey
    else:
        if not instance_id:
            print("Must provide --instance-id or --host/--port")
            sys.exit(1)
        async with get_session() as session:
            instance = (
                (await session.execute(select(Instance).where(Instance.instance_id == instance_id)))
                .unique()
                .scalar_one_or_none()
            )
            host = instance.host
            port = next(p for p in instance.port_mappings if p["internal_port"] == 8001)[
                "external_port"
            ]
            miner_hotkey = instance.miner_hotkey

    headers, _ = miner_client.sign_request(miner_hotkey, purpose="chutes")
    client = httpx.AsyncClient(
        base_url=f"http://{host}:{port}",
        timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0),
    )
    try:
        async with client.stream("GET", "/logs/stream?backfill=1000", headers=headers) as resp:
            async for chunk in resp.aiter_text():
                cont = chunk.strip()
                if cont not in ("", "."):
                    print(cont)
    finally:
        await client.aclose()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stream logs from a chute instance")
    parser.add_argument("--instance-id", help="Instance ID to look up in the database")
    parser.add_argument("--host", help="IP address of the instance (skips DB lookup)")
    parser.add_argument("--port", type=int, help="Log port of the instance (skips DB lookup)")
    parser.add_argument(
        "--miner-hotkey", help="Miner SS58 hotkey (required with --host/--port if no --instance-id)"
    )
    # Support positional instance_id for backwards compatibility.
    parser.add_argument("instance_id_pos", nargs="?", help=argparse.SUPPRESS)
    args = parser.parse_args()

    instance_id = args.instance_id or args.instance_id_pos
    asyncio.run(
        get_logs(
            instance_id=instance_id, host=args.host, port=args.port, miner_hotkey=args.miner_hotkey
        )
    )
