"""
Every script a chart workload runs must exist in the API image.

Root-level scripts are copied into the image one ADD line at a time, so adding a cronjob means
editing the Dockerfile too -- and forgetting is invisible until the pod runs and dies with
"can't open file". This pins the invariant at test time instead.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO / "Dockerfile"
TEMPLATES = REPO / "charts" / "templates"

# Directories the API stage copies wholesale (`ADD --chown=chutes <dir> /app/<dir>`); anything
# inside one of these needs no per-file ADD.
WHOLESALE_DIRS = ("api", "metasync", "tokenizer", "scripts")


def _scripts_referenced_by_charts() -> set[str]:
    """Python files named in a chart container's command/args."""
    referenced = set()
    for template in TEMPLATES.glob("*.yaml"):
        text = template.read_text()
        for match in re.finditer(r"^\s*-\s+([A-Za-z0-9_./-]+\.py)\s*$", text, re.MULTILINE):
            path = match.group(1).lstrip("/")
            # ConfigMap-mounted scripts live at /app/... but are not built into the image.
            if path.startswith("app/"):
                continue
            referenced.add(path)
    return referenced


def _root_scripts_in_image() -> set[str]:
    """Root-level .py files the API stage ADDs individually."""
    api_stage = DOCKERFILE.read_text().split("FROM base AS api", 1)[-1]
    return set(re.findall(r"^ADD .*?\s([A-Za-z0-9_-]+\.py)\s+/app/", api_stage, re.MULTILINE))


def test_charts_reference_at_least_one_script():
    """Guard against the regexes silently matching nothing and the test passing vacuously."""
    assert len(_scripts_referenced_by_charts()) > 5
    assert len(_root_scripts_in_image()) > 5


@pytest.mark.parametrize("script", sorted(_scripts_referenced_by_charts()))
def test_script_is_available_in_the_image(script):
    assert (REPO / script).is_file(), f"{script} is referenced by a chart but not in the repo"

    if script.split("/")[0] in WHOLESALE_DIRS:
        return  # copied with its directory

    assert script in _root_scripts_in_image(), (
        f"{script} is a root-level script run by a chart workload, but the API stage of the "
        f"Dockerfile never ADDs it -- the pod will fail with 'can't open file'. Add:\n"
        f"    ADD --chown=chutes {script} /app/{script}"
    )
