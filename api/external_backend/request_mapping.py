"""Dependency-light request mapping shared by HTTP and realtime relays."""

from __future__ import annotations

import copy
import math
import re
from typing import Any, Mapping, Protocol

from api.external_transport.errors import RequestRejectedError

from .mapping import extract_value


class ExternalRequestMappingError(ValueError):
    """Persisted request mapping configuration is invalid."""


class RouteLike(Protocol):
    request_config: Mapping[str, Any]
    upstream_resource_id: str


def _object(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ExternalRequestMappingError(f"{name} must be an object")
    return dict(value)


def _parameter_name(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or re.search(r"[\x00-\x20&=#?]", value)
    ):
        raise ExternalRequestMappingError(f"{label} is invalid")
    return value


def map_upstream_query_parameters(
    route: RouteLike,
    request_body: Any,
    client_query: Mapping[str, Any],
) -> dict[str, Any]:
    """Map scalar upstream query values and pin all resource selectors."""

    request_config = _object(route.request_config, "request_config")
    configured = _object(
        request_config.get("query_parameters"), "request_config.query_parameters"
    )
    context = {
        "resource": route.upstream_resource_id,
        "model": route.upstream_resource_id,
        "upstream_resource_id": route.upstream_resource_id,
    }
    sources = {"body": request_body, "query": client_query, "context": context}
    result = dict(client_query)
    for raw_name, rule in configured.items():
        name = _parameter_name(raw_name, "request_config.query_parameters name")
        if isinstance(rule, str):
            value = extract_value(sources, rule, required=True)
        elif isinstance(rule, Mapping):
            unknown = set(rule) - {"path", "value", "required"}
            if unknown:
                raise ExternalRequestMappingError(
                    "query parameter rule contains unsupported fields"
                )
            if "path" in rule and "value" in rule:
                raise ExternalRequestMappingError(
                    "query parameter rule cannot contain both path and value"
                )
            if "value" in rule:
                if "required" in rule:
                    raise ExternalRequestMappingError(
                        "literal query parameter rules cannot set required"
                    )
                value = copy.deepcopy(rule["value"])
            else:
                path = rule.get("path")
                if not isinstance(path, str) or not path:
                    raise ExternalRequestMappingError(
                        "query parameter rule requires a path or value"
                    )
                required = rule.get("required", True)
                if not isinstance(required, bool):
                    raise ExternalRequestMappingError(
                        "query parameter required must be a boolean"
                    )
                value = extract_value(sources, path, required=required)
        else:
            value = copy.deepcopy(rule)
        if name in context:
            value = context[name]
        if value is None:
            result.pop(name, None)
            continue
        if isinstance(value, (dict, list, tuple)) or not isinstance(
            value, (str, int, float, bool)
        ):
            raise RequestRejectedError("mapped query parameter is invalid")
        if isinstance(value, float) and not math.isfinite(value):
            raise RequestRejectedError("mapped query parameter is invalid")
        result[name] = value
    for name in context:
        if name in result:
            result[name] = context[name]
    resource_name = request_config.get("resource_query_parameter")
    if resource_name is not None:
        resource_name = _parameter_name(
            resource_name, "request_config.resource_query_parameter"
        )
        result[resource_name] = route.upstream_resource_id

    validated: dict[str, Any] = {}
    for raw_name, value in result.items():
        name = _parameter_name(raw_name, "outbound query parameter name")
        if isinstance(value, (dict, list, tuple)) or not isinstance(
            value, (str, int, float, bool)
        ):
            raise RequestRejectedError("mapped query parameter is invalid")
        if isinstance(value, float) and not math.isfinite(value):
            raise RequestRejectedError("mapped query parameter is invalid")
        validated[name] = value
    return validated


__all__ = ["ExternalRequestMappingError", "map_upstream_query_parameters"]
