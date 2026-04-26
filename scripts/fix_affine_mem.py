"""
Update engine_args for affine chutes:
- sglang: set --mem-fraction-static 0.8, add --chunked-prefill-size 4096
- vllm: set --gpu-memory-utilization 0.8, add --max-num-batched-tokens 4096

Usage:
  python scripts/fix_affine_mem.py           # dry-run (default)
  python scripts/fix_affine_mem.py --apply   # actually persist changes
"""

import ast
import re
import sys
import uuid
import asyncio
import orjson as json
from loguru import logger
from api.config import settings
from api.database import get_session
from api.chute.schemas import Chute
from sqlalchemy import select, func

ALLOWED_BUILDERS = {"build_sglang_chute", "build_vllm_chute"}


def update_engine_args(engine_args: str, image_name: str, gpu_count: int | None = None) -> str:
    if image_name == "sglang":
        mem_flag = "--mem-fraction-static"
        prefill_flag = "--chunked-prefill-size"
    elif image_name == "vllm":
        mem_flag = "--gpu-memory-utilization"
        prefill_flag = "--max-num-batched-tokens"
    else:
        raise ValueError(f"Unexpected image name: {image_name}")

    flags = [(mem_flag, "0.8"), (prefill_flag, "4096")]

    # Force TP=1 / DP=gpu_count for TEE chutes.
    if gpu_count is not None:
        if image_name == "sglang":
            flags += [("--tp", "1"), ("--dp", str(gpu_count))]
        else:
            flags += [("--tensor-parallel-size", "1"), ("--data-parallel-size", str(gpu_count))]

    for flag, default in flags:
        pattern = re.escape(flag) + r"\s+=?\s*\S+"
        if re.search(pattern, engine_args):
            engine_args = re.sub(pattern, f"{flag} {default}", engine_args)
        else:
            engine_args = engine_args.rstrip() + f" {flag} {default}"

    # Ensure no concatenated flags (e.g. "4096--context-length")
    engine_args = re.sub(r"(?<=\S)(--)", r" \1", engine_args)
    # Collapse any resulting double spaces
    engine_args = re.sub(r"  +", " ", engine_args)

    return engine_args.strip()


def find_engine_args_node(code: str) -> tuple[ast.Constant | None, str | None]:
    """
    Parse the chute code with ast and find the engine_args keyword
    in the build_sglang_chute / build_vllm_chute call.
    Returns (ast_node, current_value) or (None, None).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        func_name = None
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr
        if func_name not in ALLOWED_BUILDERS:
            continue
        for keyword in node.keywords:
            if keyword.arg == "engine_args":
                if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                    return keyword.value, keyword.value.value
    return None, None


def apply_engine_args_update(code: str, new_value: str) -> str | None:
    """
    Parse code, set engine_args to new_value, unparse back.
    Adds engine_args if it doesn't exist on the builder call.
    Returns None if no builder call found.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in ALLOWED_BUILDERS:
            continue
        for kw in node.keywords:
            if kw.arg == "engine_args":
                kw.value = ast.Constant(value=new_value)
                return ast.unparse(tree)
        node.keywords.append(ast.keyword(arg="engine_args", value=ast.Constant(value=new_value)))
        return ast.unparse(tree)
    return None


def show_diff(chute_id: str, name: str, image_name: str, old: str, new: str):
    """Print a readable diff of the engine_args change."""
    print(f"\n{'=' * 72}")
    print(f"Chute: {name}")
    print(f"ID:    {chute_id}")
    print(f"Image: {image_name}")
    print(f'  - engine_args="{old}"')
    print(f'  + engine_args="{new}"')


async def main():
    dry_run = "--apply" not in sys.argv
    if dry_run:
        logger.info("DRY RUN mode (pass --apply to persist changes)")
    else:
        logger.warning("APPLY mode: changes will be persisted")

    async with get_session() as session:
        chutes = (
            (await session.execute(select(Chute).where(Chute.name.ilike("%affine%"))))
            .unique()
            .scalars()
            .all()
        )
        logger.info(f"Found {len(chutes)} affine chutes")

        updated = 0
        skipped = 0
        for chute in chutes:
            image_name = chute.image.name
            if image_name not in ("sglang", "vllm"):
                logger.warning(
                    f"Skipping {chute.name} ({chute.chute_id}): image is {image_name}, not sglang/vllm"
                )
                skipped += 1
                continue

            ast_node, old_engine_args = find_engine_args_node(chute.code)
            old_engine_args = old_engine_args or ""
            gpu_count = chute.node_selector.get("gpu_count", 1) if chute.tee else None
            new_engine_args = update_engine_args(old_engine_args, image_name, gpu_count=gpu_count)
            if old_engine_args.strip() == new_engine_args.strip():
                logger.info(f"No changes needed for {chute.name} ({chute.chute_id})")
                skipped += 1
                continue

            show_diff(chute.chute_id, chute.name, image_name, old_engine_args, new_engine_args)

            if dry_run:
                updated += 1
                continue

            new_code = apply_engine_args_update(chute.code, new_engine_args)
            if new_code is None:
                logger.error(
                    f"Failed to apply update for {chute.name} ({chute.chute_id}): "
                    "could not find/replace engine_args in source"
                )
                skipped += 1
                continue
            chute.code = new_code
            chute.version = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{chute.image_id}:{new_code}"))
            chute.updated_at = func.now()
            await session.commit()
            await session.refresh(chute)

            await settings.redis_client.publish(
                "miner_broadcast",
                json.dumps(
                    {
                        "reason": "chute_updated",
                        "data": {
                            "chute_id": chute.chute_id,
                            "version": chute.version,
                            "job_only": not chute.cords,
                        },
                    }
                ).decode(),
            )
            logger.success(f"Updated and broadcast {chute.name}")
            updated += 1

        print(f"\n{'=' * 72}")
        mode = "Would update" if dry_run else "Updated"
        print(f"{mode}: {updated} | Skipped: {skipped} | Total: {len(chutes)}")
        if dry_run and updated > 0:
            print("Run with --apply to persist these changes.")


asyncio.run(main())
