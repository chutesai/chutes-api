"""
Drift guards for the miner OpenAPI response schemas.

The miner endpoints document their responses via ``responses={200: {"model": ...}}``
rather than ``response_model=``, so FastAPI never validates the real payload against
the declared schema. Nothing would catch a schema that quietly falls out of sync with
what the handler actually returns -- these tests do.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic.fields import ComputedFieldInfo
from sqlalchemy.orm import class_mapper

import api.database.orms  # noqa: F401  # registers the ORM classes
from api.chute.schemas import Chute
from api.miner.schemas import MinerChute, MinerMetagraphNode

with patch("ctypes.CDLL", return_value=MagicMock()):
    from api.miner.router import router as miner_router


def _model_to_dict_keys_for_chute() -> set[str]:
    """
    Statically reproduce the key set ``api.miner.router.model_to_dict`` builds for a
    Chute, so a new/removed ORM column shows up as a failure here.
    """
    keys = {column.key for column in class_mapper(Chute).columns if column.key != "env_creation"}
    keys |= {
        name
        for name, value in vars(Chute).items()
        if isinstance(getattr(value, "decorator_info", None), ComputedFieldInfo)
    }
    # Injected by model_to_dict for Chute specifically.
    keys |= {
        "image",
        "preemptible",
        "effective_compute_multiplier",
        "compute_multiplier_factors",
        "bounty",
    }
    # Stripped by model_to_dict.
    keys -= {"symmetric_key", "host", "inspecto", "port_mappings", "port", "package_hashes"}
    return {key for key in keys if not key.startswith("rint_")}


def test_miner_chute_schema_matches_model_to_dict():
    assert set(MinerChute.model_fields) == _model_to_dict_keys_for_chute()


def test_miner_metagraph_schema_matches_orm_columns():
    from api.metagraph import MetagraphNode

    # The endpoint returns ORM objects directly; jsonable_encoder emits every loaded
    # column (and drops _sa_instance_state), so the schema must cover all of them.
    columns = {column.key for column in class_mapper(MetagraphNode).columns}
    assert set(MinerMetagraphNode.model_fields) == columns


@pytest.mark.parametrize(
    "path,method",
    [
        ("/inventory", "GET"),
        ("/active_instances/", "GET"),
        ("/chutes/{chute_id}/{version}", "GET"),
        ("/stats", "GET"),
        ("/scores", "GET"),
        ("/unique_chute_history/{hotkey}", "GET"),
        ("/thrash_cooldowns", "GET"),
        ("/metagraph", "GET"),
        ("/servers/", "GET"),
    ],
)
def test_rest_endpoints_document_a_200_schema(path, method):
    """
    Every non-streaming miner endpoint should advertise a response schema. Streaming
    (SSE) endpoints are deliberately excluded -- OpenAPI models request/response bodies,
    not event streams.
    """
    routes = [r for r in miner_router.routes if r.path == path and method in r.methods]
    assert routes, f"no {method} route registered for {path}"
    route = routes[0]
    documented = route.response_model is not None or 200 in route.responses
    assert documented, f"{method} {path} has no documented 200 response schema"
