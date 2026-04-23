from types import SimpleNamespace

import pytest

from api import model_routing


def test_parse_model_parameter_trims_and_parses_suffix() -> None:
    raw, mode = model_routing.parse_model_parameter("  glm:LaTeNcY  ")
    assert raw == "glm"
    assert mode == "latency"


@pytest.mark.asyncio
async def test_resolve_alias_with_latency_suffix(monkeypatch) -> None:
    c1 = SimpleNamespace(chute_id="c1", standard_template="vllm")
    c2 = SimpleNamespace(chute_id="c2", standard_template="vllm")

    async def fake_get_one(name_or_id: str):
        return None

    async def fake_get_user_alias(user_id: str, alias: str):
        assert user_id == "u1"
        assert alias == "glm"
        return ["c1", "c2"]

    async def fake_load_chutes_map(chute_ids: list[str]):
        assert chute_ids == ["c1", "c2"]
        return {"c1": c1, "c2": c2}

    async def fake_rank_by_metric(chute_ids: list[str], chutes_map: dict, metric: str):
        assert chute_ids == ["c1", "c2"]
        assert metric == "ptps"
        return [c2, c1]

    monkeypatch.setattr(model_routing, "get_one", fake_get_one)
    monkeypatch.setattr(model_routing, "get_user_alias", fake_get_user_alias)
    monkeypatch.setattr(model_routing, "_load_chutes_map", fake_load_chutes_map)
    monkeypatch.setattr(model_routing, "_rank_by_metric", fake_rank_by_metric)

    ranked, mode = await model_routing.resolve_model_parameter("  glm:latency  ", "u1", "vllm")
    assert mode == "latency"
    assert [c.chute_id for c in ranked] == ["c2", "c1"]


@pytest.mark.asyncio
async def test_resolve_csv_expands_alias_entries(monkeypatch) -> None:
    c1 = SimpleNamespace(chute_id="c1", standard_template="vllm")
    c2 = SimpleNamespace(chute_id="c2", standard_template="vllm")
    c3 = SimpleNamespace(chute_id="zai-org/GLM-4.7-TEE", standard_template="vllm")
    captured = {}

    async def fake_get_one(name_or_id: str):
        if name_or_id == "zai-org/GLM-4.7-TEE":
            return c3
        return None

    async def fake_get_user_alias(user_id: str, alias: str):
        if alias == "glm":
            return ["c1", "c2"]
        return None

    async def fake_load_chutes_map(chute_ids: list[str]):
        captured["ids"] = chute_ids
        return {
            "c1": c1,
            "c2": c2,
            "zai-org/GLM-4.7-TEE": c3,
        }

    async def fake_rank_by_metric(chute_ids: list[str], chutes_map: dict, metric: str):
        assert metric == "otps"
        return [chutes_map[cid] for cid in chute_ids]

    monkeypatch.setattr(model_routing, "get_one", fake_get_one)
    monkeypatch.setattr(model_routing, "get_user_alias", fake_get_user_alias)
    monkeypatch.setattr(model_routing, "_load_chutes_map", fake_load_chutes_map)
    monkeypatch.setattr(model_routing, "_rank_by_metric", fake_rank_by_metric)

    ranked, mode = await model_routing.resolve_model_parameter(
        "glm,zai-org/GLM-4.7-TEE:throughput", "u1", "vllm"
    )
    assert mode == "throughput"
    assert captured["ids"] == ["c1", "c2", "zai-org/GLM-4.7-TEE"]
    assert [c.chute_id for c in ranked] == ["c1", "c2", "zai-org/GLM-4.7-TEE"]
