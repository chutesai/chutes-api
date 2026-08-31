"""ORM and API schemas for externally executed Chutes."""

from __future__ import annotations

import hashlib
import math
import re
import string
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

import orjson
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    SecretStr,
    computed_field,
    field_validator,
    model_validator,
)
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from api.database import Base, generate_uuid
from api.chute.path_policy import is_reserved_canonical_chute_path
from api.chute.standard_templates import STANDARD_TEMPLATES
from api.external_transport.header_policy import requires_secret_backing
from api.payment.pricing import PricingConfigurationError, parse_pricing_rules


_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_SECRET_REFERENCE_PATTERN = re.compile(r"^secret://[A-Za-z0-9][A-Za-z0-9._/-]{0,510}$")
_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_CORD_PATH_PATTERN = re.compile(r"^(/[a-z][a-z0-9_]*)+$", re.I)
_MAX_CONFIG_JSON_DEPTH = 64
_MAX_CONFIG_JSON_NODES = 10_000
_SENSITIVE_KEYS = frozenset(
    {
        "access_key",
        "access_token",
        "api_key",
        "auth_token",
        "authorization",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "private_key",
        "secret",
        "secret_key",
    }
)
_OPERATION_SCHEMA_PATHS = frozenset(
    {
        ("realtime", "message_schema"),
        ("submission_contract", "output_schema"),
        ("websocket", "message_schema"),
    }
)


class ExternalOperationMode(str, Enum):
    SYNC = "sync"
    STREAM = "stream"
    TASK = "task"
    REALTIME = "realtime"


class ExternalOperationStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ExternalSettlementStatus(str, Enum):
    PENDING = "pending"
    SETTLED = "settled"
    NOT_BILLABLE = "not_billable"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class ExternalResultStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"


def _enum_values(enum_type: type[Enum]) -> tuple[str, ...]:
    return tuple(item.value for item in enum_type)


def _validate_identifier(value: str, field_name: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must start with a lowercase letter and contain only "
            "lowercase letters, digits, dots, underscores, or dashes"
        )
    return value


def _validate_secret_reference(value: str) -> str:
    if not _SECRET_REFERENCE_PATTERN.fullmatch(value):
        raise ValueError("credential_reference must be an opaque secret:// reference")
    path = value.removeprefix("secret://")
    if (
        value.endswith("/")
        or "//" in path
        or any(part in {".", ".."} for part in path.split("/"))
    ):
        raise ValueError("credential_reference contains an invalid path")
    return value


def _validate_base_url(value: str) -> str:
    parsed = urlsplit(str(value))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base_url must be an absolute HTTP URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query or fragment")
    return str(value).rstrip("/")


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _validate_bounded_json(value: Any, field_name: str) -> Any:
    """Require a bounded tree made only from JSON-compatible values."""

    pending: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_CONFIG_JSON_NODES or depth > _MAX_CONFIG_JSON_DEPTH:
            raise ValueError(f"{field_name} exceeds the JSON complexity limit")
        if isinstance(item, dict):
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise ValueError(f"{field_name} JSON object keys must be strings")
                pending.append((nested, depth + 1))
        elif isinstance(item, list):
            pending.extend((nested, depth + 1) for nested in item)
        elif item is None or isinstance(item, (str, bool, int)):
            continue
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{field_name} JSON numbers must be finite")
        else:
            raise ValueError(f"{field_name} must contain only JSON-compatible values")
    return value


def _validate_non_secret_value(
    value: Any,
    field_name: str,
    *,
    schema_paths: frozenset[tuple[str, ...]] = frozenset(),
) -> Any:
    _validate_bounded_json(value, field_name)
    pending: list[tuple[Any, str | None, tuple[str, ...]]] = [(value, None, ())]
    while pending:
        item, parent_key, path = pending.pop()
        if path in schema_paths:
            # JSON Schema property names commonly look credential-like.  A schema
            # is bounded metadata, not a source of credentials to inject upstream.
            if not isinstance(item, dict):
                label = ".".join((field_name, *path))
                raise ValueError(f"{label} must be an object")
            continue
        if isinstance(item, dict):
            for key, nested in item.items():
                normalized = _normalized_key(key)
                if normalized in _SENSITIVE_KEYS or (
                    parent_key
                    in {"headers", "query", "query_parameters", "static_headers"}
                    and requires_secret_backing(str(key))
                ):
                    raise ValueError(f"{field_name} must not contain credential values")
                pending.append((nested, normalized, (*path, key)))
        elif isinstance(item, list):
            pending.extend((nested, parent_key, path) for nested in item)
    return value


def _validate_non_secret_json(
    value: dict[str, Any],
    field_name: str,
    *,
    schema_paths: frozenset[tuple[str, ...]] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return _validate_non_secret_value(value, field_name, schema_paths=schema_paths)


def _validate_schema_metadata(value: dict[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return _validate_bounded_json(value, field_name)


def _validate_connection_config(value: dict[str, Any]) -> dict[str, Any]:
    value = _validate_non_secret_json(value, "connection_config")
    if "allow_insecure_http" in value and not isinstance(
        value["allow_insecure_http"], bool
    ):
        raise ValueError("connection_config.allow_insecure_http must be a boolean")
    # Imported lazily to keep the ORM/schema module free of an eager dependency
    # back into the external execution service during application startup.
    from .governance import validate_governance_config

    validate_governance_config(value.get("governance"))
    return value


def _validate_optional_non_secret_json(
    value: dict[str, Any] | None, field_name: str
) -> dict[str, Any] | None:
    if value is None:
        return None
    return _validate_non_secret_json(value, field_name)


class _RequestModel(BaseModel):
    model_config = {"extra": "forbid"}


class ExternalCord(_RequestModel):
    """Public Cord metadata without importing the hosted execution domain."""

    method: str = Field(min_length=3, max_length=7)
    path: str = Field(min_length=1, max_length=255)
    function: str = Field(min_length=1, max_length=255)
    stream: bool
    passthrough: bool = False
    public_api_path: str = Field(min_length=1, max_length=255)
    public_api_method: str = Field(min_length=3, max_length=7)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    output_content_type: str | None = Field(default=None, min_length=1, max_length=255)
    minimal_input_schema: dict[str, Any] = Field(default_factory=dict)

    @field_validator("method", "public_api_method")
    @classmethod
    def validate_method(cls, value: str, info) -> str:
        value = value.upper()
        allowed = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}
        if value not in allowed:
            raise ValueError(f"unsupported {info.field_name}")
        return value

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if not _CORD_PATH_PATTERN.fullmatch(value):
            raise ValueError("path must be an absolute Chute cord path")
        return value

    @field_validator("public_api_path")
    @classmethod
    def validate_public_api_path(cls, value: str) -> str:
        if not re.fullmatch(r"^(/[a-z][a-z0-9_-]*)+$", value, re.I):
            raise ValueError("public_api_path must be an absolute public path")
        if is_reserved_canonical_chute_path(value):
            raise ValueError(
                "public_api_path is reserved for a canonical Chute endpoint"
            )
        return value

    @field_validator("input_schema", "output_schema", "minimal_input_schema")
    @classmethod
    def validate_schema(cls, value: dict[str, Any], info) -> dict[str, Any]:
        return _validate_schema_metadata(value, info.field_name)


class ExternalUsageDescriptor(_RequestModel):
    requests: float = Field(default=1, ge=0)
    tokens: dict[str, float] = Field(default_factory=dict)
    images: dict[str, float] = Field(default_factory=dict)
    input_media_seconds: dict[str, float] = Field(default_factory=dict)
    output_media_seconds: dict[str, float] = Field(default_factory=dict)
    characters: dict[str, float] = Field(default_factory=dict)
    counts: dict[str, float] = Field(default_factory=dict)
    tools: dict[str, float] = Field(default_factory=dict)
    dimensions: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "tokens",
        "images",
        "input_media_seconds",
        "output_media_seconds",
        "characters",
        "counts",
        "tools",
    )
    @classmethod
    def validate_quantities(cls, value: dict[str, float]) -> dict[str, float]:
        for name, quantity in value.items():
            _validate_identifier(name, "usage quantity")
            if not math.isfinite(quantity) or quantity < 0:
                raise ValueError("usage quantities must be finite and non-negative")
        return value

    @field_validator("requests")
    @classmethod
    def validate_requests(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("requests must be finite and non-negative")
        return value

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_non_secret_json(value, "dimensions")


class ExternalArtifactDescriptor(_RequestModel):
    kind: str = Field(min_length=1, max_length=64)
    reference: str = Field(min_length=1, max_length=2048)
    content_type: str | None = Field(default=None, min_length=1, max_length=255)
    size_bytes: int | None = Field(default=None, ge=0)
    expires_at: datetime | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        return _validate_identifier(value, "result kind")

    @field_validator("attributes")
    @classmethod
    def validate_attributes(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_non_secret_json(value, "attributes")


class ExternalResultDescriptor(_RequestModel):
    status: ExternalResultStatus = ExternalResultStatus.COMPLETE
    artifacts: list[ExternalArtifactDescriptor] = Field(
        default_factory=list, max_length=1000
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_non_secret_json(value, "metadata")

    @model_validator(mode="after")
    def validate_partial_result(self):
        if self.status is ExternalResultStatus.PARTIAL and not self.artifacts:
            raise ValueError("a partial result must describe at least one artifact")
        return self


class ExternalErrorDescriptor(_RequestModel):
    message: str = Field(min_length=1, max_length=4096)
    code: str | None = Field(default=None, min_length=1, max_length=128)
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("details")
    @classmethod
    def validate_details(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_non_secret_json(value, "details")


class ExternalAuthHeaderTemplate(_RequestModel):
    name: str = Field(min_length=1, max_length=128)
    template: str = Field(min_length=1, max_length=1024)
    references: dict[str, str] = Field(min_length=1, max_length=32)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _HEADER_NAME_PATTERN.fullmatch(value):
            raise ValueError("auth header name is invalid")
        if value.lower() in {"host", "content-length"} or value.lower().startswith(
            "x-forwarded-"
        ):
            raise ValueError("auth header name is not configurable")
        return value

    @field_validator("template")
    @classmethod
    def validate_template(cls, value: str) -> str:
        if any(delimiter in value for delimiter in ("\r", "\n", "\x00")):
            raise ValueError("auth header template contains a control delimiter")
        return value

    @field_validator("references")
    @classmethod
    def validate_references(cls, value: dict[str, str]) -> dict[str, str]:
        for field_name, credential_name in value.items():
            _validate_identifier(field_name, "auth template field")
            _validate_identifier(credential_name, "credential name")
        return value

    @model_validator(mode="after")
    def validate_template_fields(self):
        fields: set[str] = set()
        for _literal, field_name, format_spec, conversion in string.Formatter().parse(
            self.template
        ):
            if field_name is None:
                continue
            if format_spec or conversion:
                raise ValueError(
                    "auth header templates only support simple named fields"
                )
            _validate_identifier(field_name, "auth template field")
            fields.add(field_name)
        if fields != set(self.references):
            raise ValueError(
                "auth header template fields must exactly match references"
            )
        return self


class ExternalRouteConfig(_RequestModel):
    cord_path: str = Field(min_length=1, max_length=255)
    upstream_resource_id: str = Field(min_length=1, max_length=512)
    operation_mode: ExternalOperationMode
    protocol: str = Field(min_length=1, max_length=64)
    base_url: HttpUrl | None = None
    path_template: str = Field(min_length=1, max_length=2048)
    method: str = Field(default="POST", min_length=3, max_length=7)
    request_config: dict[str, Any] = Field(default_factory=dict)
    response_config: dict[str, Any] = Field(default_factory=dict)
    operation_config: dict[str, Any] = Field(default_factory=dict)
    capabilities: dict[str, Any] = Field(default_factory=dict)

    @field_validator("cord_path")
    @classmethod
    def validate_cord_path(cls, value: str) -> str:
        if not _CORD_PATH_PATTERN.fullmatch(value):
            raise ValueError("cord_path must be an absolute Chute cord path")
        return value

    @field_validator("upstream_resource_id")
    @classmethod
    def validate_upstream_resource_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("upstream_resource_id must not be blank")
        return value

    @field_validator("protocol")
    @classmethod
    def validate_protocol(cls, value: str) -> str:
        return _validate_identifier(value, "protocol")

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: HttpUrl | None) -> HttpUrl | None:
        if value is not None:
            _validate_base_url(str(value))
        return value

    @field_validator("path_template")
    @classmethod
    def validate_path_template(cls, value: str) -> str:
        if not value.startswith("/") or any(
            token in value for token in ("://", "?", "#", "\\")
        ):
            raise ValueError(
                "path_template must be an absolute path without a query or fragment"
            )
        if any(part == ".." for part in value.split("/")):
            raise ValueError("path_template must not contain parent traversal segments")
        return value

    @field_validator("method")
    @classmethod
    def validate_method(cls, value: str) -> str:
        value = value.upper()
        if value not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}:
            raise ValueError("unsupported endpoint method")
        return value

    @field_validator(
        "request_config", "response_config", "operation_config", "capabilities"
    )
    @classmethod
    def validate_json_config(cls, value: dict[str, Any], info) -> dict[str, Any]:
        schema_paths = (
            _OPERATION_SCHEMA_PATHS
            if info.field_name == "operation_config"
            else frozenset()
        )
        return _validate_non_secret_json(
            value,
            info.field_name,
            schema_paths=schema_paths,
        )


def _validate_routes(
    value: list[dict[str, Any]] | list[ExternalRouteConfig],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("routes must contain at least one route")
    routes = [
        item
        if isinstance(item, ExternalRouteConfig)
        else ExternalRouteConfig.model_validate(item)
        for item in value
    ]
    paths = [route.cord_path for route in routes]
    if len(paths) != len(set(paths)):
        raise ValueError("routes must have unique cord_path values")
    return [route.model_dump(mode="json") for route in routes]


def _validate_credential_references(value: dict[str, str]) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError("credential_references must contain at least one reference")
    for name, reference in value.items():
        _validate_identifier(name, "credential name")
        _validate_secret_reference(reference)
    return value


def _validate_credentials(value: dict[str, SecretStr]) -> dict[str, SecretStr]:
    if not isinstance(value, dict) or not value:
        raise ValueError("credentials must contain at least one value")
    for name, credential in value.items():
        _validate_identifier(name, "credential name")
        if not credential.get_secret_value():
            raise ValueError("credential values must not be empty")
    return value


def _validate_auth_header_templates(
    value: list[dict[str, Any]] | list[ExternalAuthHeaderTemplate],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("auth_header_templates must contain at least one template")
    templates = [
        item
        if isinstance(item, ExternalAuthHeaderTemplate)
        else ExternalAuthHeaderTemplate.model_validate(item)
        for item in value
    ]
    header_names = [template.name.lower() for template in templates]
    if len(header_names) != len(set(header_names)):
        raise ValueError("auth_header_templates must have unique header names")
    return [template.model_dump(mode="json") for template in templates]


class ExternalBackendAccount(Base):
    __tablename__ = "external_backend_accounts"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "name", name="uq_external_backend_accounts_user_name"
        ),
        CheckConstraint(
            "adapter ~ '^[a-z][a-z0-9._-]{0,63}$'",
            name="ck_external_backend_accounts_adapter",
        ),
        CheckConstraint(
            "jsonb_typeof(credential_references) = 'object' "
            "AND credential_references <> '{}'::jsonb",
            name="ck_external_backend_accounts_credential_references",
        ),
        CheckConstraint(
            "jsonb_typeof(auth_header_templates) = 'array' "
            "AND jsonb_array_length(auth_header_templates) > 0",
            name="ck_external_backend_accounts_auth_header_templates",
        ),
        CheckConstraint(
            "base_url ~ '^https?://' AND base_url !~ '[?#]' "
            "AND base_url !~ '^https?://[^/]*@'",
            name="ck_external_backend_accounts_base_url",
        ),
        CheckConstraint(
            "jsonb_typeof(connection_config) = 'object'",
            name="ck_external_backend_accounts_connection_config",
        ),
        CheckConstraint(
            "jsonb_typeof(management_metadata) = 'object'",
            name="ck_external_backend_accounts_management_metadata",
        ),
        CheckConstraint(
            "NOT connection_config ?| ARRAY['access_key', 'access_token', 'api_key', "
            "'auth_token', 'authorization', 'bearer_token', 'client_secret', "
            "'credential', 'credentials', 'password', 'private_key', 'secret', "
            "'secret_key']",
            name="ck_external_backend_accounts_no_inline_credentials",
        ),
    )

    account_id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(
        String,
        ForeignKey("users.user_id", ondelete="RESTRICT"),
        nullable=False,
    )
    name = Column(String(128), nullable=False)
    adapter = Column(String(64), nullable=False)
    base_url = Column(String(2048), nullable=False)
    credential_references = Column(JSONB, nullable=False)
    auth_header_templates = Column(JSONB, nullable=False)
    connection_config = Column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    management_metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    enabled = Column(Boolean, nullable=False, default=True, server_default=text("TRUE"))
    artifact_relay_invalidated_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    user = relationship(
        "User", back_populates="external_backend_accounts", lazy="joined"
    )
    bindings = relationship(
        "ExternalChuteBinding",
        back_populates="account",
        lazy="select",
        passive_deletes=True,
    )
    operations = relationship(
        "ExternalOperation",
        back_populates="account",
        lazy="select",
        passive_deletes=True,
    )

    @validates("name")
    def validate_name(self, _, value: str) -> str:
        if not isinstance(value, str) or not value.strip() or len(value) > 128:
            raise ValueError("name must contain 1-128 characters")
        return value.strip()

    @validates("adapter")
    def validate_adapter(self, _, value: str) -> str:
        return _validate_identifier(value, "adapter")

    @validates("base_url")
    def validate_base_url(self, _, value: str) -> str:
        return _validate_base_url(value)

    @validates("credential_references")
    def validate_credential_references(
        self, _, value: dict[str, str]
    ) -> dict[str, str]:
        return _validate_credential_references(value)

    @validates("auth_header_templates")
    def validate_auth_header_templates(
        self, _, value: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return _validate_auth_header_templates(value)

    @validates("connection_config")
    def validate_connection_config(self, _, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_connection_config(value)

    @validates("management_metadata")
    def validate_management_metadata(self, _, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_non_secret_json(value, "management_metadata")


class ExternalChuteBinding(Base):
    __tablename__ = "external_chute_bindings"
    __table_args__ = (
        UniqueConstraint("chute_id", name="uq_external_chute_bindings_chute_id"),
        Index("idx_external_chute_bindings_account", "account_id"),
        CheckConstraint(
            "jsonb_typeof(routes) = 'array' AND jsonb_array_length(routes) > 0",
            name="ck_external_chute_bindings_routes",
        ),
    )

    binding_id = Column(String, primary_key=True, default=generate_uuid)
    chute_id = Column(
        String,
        ForeignKey("chutes.chute_id", ondelete="CASCADE"),
        nullable=False,
    )
    account_id = Column(
        String,
        ForeignKey("external_backend_accounts.account_id", ondelete="RESTRICT"),
        nullable=False,
    )
    routes = Column(JSONB, nullable=False)
    enabled = Column(Boolean, nullable=False, default=True, server_default=text("TRUE"))
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    chute = relationship("Chute", back_populates="external_binding", lazy="joined")
    account = relationship(
        "ExternalBackendAccount", back_populates="bindings", lazy="joined"
    )
    operations = relationship(
        "ExternalOperation",
        back_populates="binding",
        lazy="select",
        passive_deletes=True,
    )

    @validates("routes")
    def validate_routes(self, _, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _validate_routes(value)


class ExternalOperation(Base):
    __tablename__ = "external_operations"
    __table_args__ = (
        CheckConstraint(
            "operation_mode IN ('sync', 'stream', 'task', 'realtime')",
            name="ck_external_operations_operation_mode",
        ),
        CheckConstraint(
            "protocol ~ '^[a-z][a-z0-9._-]{0,63}$'",
            name="ck_external_operations_protocol",
        ),
        CheckConstraint(
            "status IN ('pending', 'submitted', 'running', 'succeeded', 'failed', "
            "'cancelled', 'expired')",
            name="ck_external_operations_status",
        ),
        CheckConstraint(
            "settlement_status IN "
            "('pending', 'settled', 'not_billable', 'failed', 'quarantined')",
            name="ck_external_operations_settlement_status",
        ),
        CheckConstraint(
            "poll_attempts >= 0",
            name="ck_external_operations_poll_attempts",
        ),
        CheckConstraint(
            "jsonb_typeof(request_metadata) = 'object'",
            name="ck_external_operations_request_metadata",
        ),
        CheckConstraint(
            "jsonb_typeof(route_snapshot) = 'object'",
            name="ck_external_operations_route_snapshot",
        ),
        CheckConstraint(
            "jsonb_typeof(upstream_metadata) = 'object'",
            name="ck_external_operations_upstream_metadata",
        ),
        CheckConstraint(
            "usage IS NULL OR jsonb_typeof(usage) = 'object'",
            name="ck_external_operations_usage",
        ),
        CheckConstraint(
            "result_descriptor IS NULL OR jsonb_typeof(result_descriptor) = 'object'",
            name="ck_external_operations_result_descriptor",
        ),
        CheckConstraint(
            "error IS NULL OR jsonb_typeof(error) = 'object'",
            name="ck_external_operations_error",
        ),
        CheckConstraint(
            "jsonb_typeof(settlement_metadata) = 'object'",
            name="ck_external_operations_settlement_metadata",
        ),
        Index("idx_external_operations_user_created", "user_id", "created_at"),
        Index("idx_external_operations_account_created", "account_id", "created_at"),
        Index("idx_external_operations_poll", "status", "next_poll_at"),
        Index("idx_external_operations_account_status", "account_id", "status"),
        Index(
            "idx_external_operations_settlement_retry",
            "settlement_status",
            "next_poll_at",
            postgresql_where=text(
                "status IN ('succeeded', 'failed', 'cancelled', 'expired') "
                "AND settlement_status IN ('pending', 'failed')"
            ),
        ),
        Index(
            "idx_external_operations_binding",
            "binding_id",
            postgresql_where=text("binding_id IS NOT NULL"),
        ),
        Index(
            "idx_external_operations_chute",
            "chute_id",
            postgresql_where=text("chute_id IS NOT NULL"),
        ),
        Index(
            "uq_external_operations_upstream_id",
            "binding_id",
            "upstream_operation_id",
            unique=True,
            postgresql_where=text("upstream_operation_id IS NOT NULL"),
        ),
        Index(
            "uq_external_operations_idempotency_key",
            "binding_id",
            "user_id",
            "idempotency_key",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),
    )

    operation_id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(
        String,
        ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True,
    )
    account_id = Column(
        String,
        ForeignKey("external_backend_accounts.account_id", ondelete="SET NULL"),
        nullable=True,
    )
    binding_id = Column(
        String,
        ForeignKey("external_chute_bindings.binding_id", ondelete="SET NULL"),
        nullable=True,
    )
    chute_id = Column(
        String,
        ForeignKey("chutes.chute_id", ondelete="SET NULL"),
        nullable=True,
    )
    cord_path = Column(String(255), nullable=False)
    operation_mode = Column(String(16), nullable=False)
    protocol = Column(String(64), nullable=False)
    status = Column(
        String(16),
        nullable=False,
        default=ExternalOperationStatus.PENDING.value,
        server_default=text("'pending'"),
    )
    settlement_status = Column(
        String(16),
        nullable=False,
        default=ExternalSettlementStatus.PENDING.value,
        server_default=text("'pending'"),
    )
    upstream_operation_id = Column(String(512), nullable=True)
    upstream_status = Column(String(128), nullable=True)
    idempotency_key = Column(String(255), nullable=True)
    route_snapshot = Column(JSONB, nullable=False)
    request_metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    upstream_metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    usage = Column(JSONB, nullable=True)
    result_descriptor = Column(JSONB, nullable=True)
    error = Column(JSONB, nullable=True)
    settlement_metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    poll_attempts = Column(Integer, nullable=False, default=0, server_default=text("0"))
    lease_owner = Column(String(255), nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    submitted_at = Column(DateTime(timezone=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    last_polled_at = Column(DateTime(timezone=True), nullable=True)
    next_poll_at = Column(DateTime(timezone=True), nullable=True)
    lease_expires_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    settled_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="external_operations", lazy="select")
    account = relationship(
        "ExternalBackendAccount", back_populates="operations", lazy="select"
    )
    binding = relationship(
        "ExternalChuteBinding", back_populates="operations", lazy="select"
    )

    @validates("operation_mode")
    def validate_operation_mode(self, _, value: str | ExternalOperationMode) -> str:
        normalized = value.value if isinstance(value, ExternalOperationMode) else value
        if normalized not in _enum_values(ExternalOperationMode):
            raise ValueError("invalid external operation mode")
        return normalized

    @validates("cord_path")
    def validate_cord_path(self, _, value: str) -> str:
        if not _CORD_PATH_PATTERN.fullmatch(value):
            raise ValueError("cord_path must be an absolute Chute cord path")
        return value

    @validates("protocol")
    def validate_protocol(self, _, value: str) -> str:
        return _validate_identifier(value, "protocol")

    @validates("status")
    def validate_status(self, _, value: str | ExternalOperationStatus) -> str:
        normalized = (
            value.value if isinstance(value, ExternalOperationStatus) else value
        )
        if normalized not in _enum_values(ExternalOperationStatus):
            raise ValueError("invalid external operation status")
        return normalized

    @validates("settlement_status")
    def validate_settlement_status(
        self, _, value: str | ExternalSettlementStatus
    ) -> str:
        normalized = (
            value.value if isinstance(value, ExternalSettlementStatus) else value
        )
        if normalized not in _enum_values(ExternalSettlementStatus):
            raise ValueError("invalid external settlement status")
        return normalized

    @validates(
        "route_snapshot", "request_metadata", "upstream_metadata", "settlement_metadata"
    )
    def validate_metadata(self, key: str, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_non_secret_json(value, key)

    @validates("usage", "result_descriptor", "error")
    def validate_optional_metadata(
        self, key: str, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        return _validate_optional_non_secret_json(value, key)


class ExternalGovernanceState(Base):
    """Trigger-maintained active and unresolved exposure for one billing scope."""

    __tablename__ = "external_governance_state"
    __table_args__ = (
        CheckConstraint(
            "scope_type IN ('user', 'account')",
            name="ck_external_governance_state_scope",
        ),
        CheckConstraint(
            "active_tasks >= 0 AND active_sync_requests >= 0 "
            "AND active_realtime >= 0 AND active_streams >= 0",
            name="ck_external_governance_state_counts",
        ),
        CheckConstraint(
            "unresolved_paygo >= 0 AND unresolved_charge >= 0",
            name="ck_external_governance_state_amounts",
        ),
    )

    scope_type = Column(String(16), primary_key=True)
    scope_id = Column(String, primary_key=True)
    active_tasks = Column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    active_sync_requests = Column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    active_realtime = Column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    active_streams = Column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    unresolved_paygo = Column(
        Numeric, nullable=False, default=0, server_default=text("0")
    )
    unresolved_charge = Column(
        Numeric, nullable=False, default=0, server_default=text("0")
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ExternalGovernanceBucket(Base):
    """Minute rollup bounding governance reads independently of operation volume."""

    __tablename__ = "external_governance_buckets"
    __table_args__ = (
        CheckConstraint(
            "scope_type IN ('user', 'account')",
            name="ck_external_governance_buckets_scope",
        ),
        CheckConstraint(
            "operation_count >= 0 AND unresolved_paygo >= 0 "
            "AND settled_paygo >= 0 AND artifact_relay_bytes >= 0",
            name="ck_external_governance_buckets_values",
        ),
        Index("idx_external_governance_buckets_expiration", "bucket_start"),
    )

    scope_type = Column(String(16), primary_key=True)
    scope_id = Column(String, primary_key=True)
    bucket_start = Column(DateTime(timezone=True), primary_key=True)
    operation_count = Column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    unresolved_paygo = Column(
        Numeric, nullable=False, default=0, server_default=text("0")
    )
    settled_paygo = Column(Numeric, nullable=False, default=0, server_default=text("0"))
    artifact_relay_bytes = Column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ExternalUsageOutbox(Base):
    """A charge that must be applied atomically before settlement completes."""

    __tablename__ = "external_usage_outbox"
    __table_args__ = (
        CheckConstraint("amount >= 0", name="ck_external_usage_outbox_amount"),
        CheckConstraint(
            "paygo_amount >= 0", name="ck_external_usage_outbox_paygo_amount"
        ),
        CheckConstraint(
            "input_tokens >= 0 AND output_tokens >= 0 AND cached_tokens >= 0",
            name="ck_external_usage_outbox_tokens",
        ),
        CheckConstraint(
            "compute_time >= 0", name="ck_external_usage_outbox_compute_time"
        ),
        CheckConstraint("attempts >= 0", name="ck_external_usage_outbox_attempts"),
        Index(
            "idx_external_usage_outbox_due",
            "next_attempt_at",
            "created_at",
        ),
    )

    event_id = Column(String(255), primary_key=True)
    operation_id = Column(
        String,
        ForeignKey("external_operations.operation_id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    )
    user_id = Column(String, nullable=False)
    chute_id = Column(String, nullable=False)
    app_id = Column(String, nullable=True)
    amount = Column(Numeric(30, 12), nullable=False)
    paygo_amount = Column(Numeric(30, 12), nullable=False)
    input_tokens = Column(
        Numeric(30, 6), nullable=False, default=0, server_default=text("0")
    )
    output_tokens = Column(
        Numeric(30, 6), nullable=False, default=0, server_default=text("0")
    )
    cached_tokens = Column(
        Numeric(30, 6), nullable=False, default=0, server_default=text("0")
    )
    compute_time = Column(Double, nullable=False, default=0, server_default=text("0"))
    track_task_completion = Column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    free_invocation = Column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    increment_invocation_quota = Column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    occurred_at = Column(DateTime(timezone=True), nullable=False)
    attempts = Column(BigInteger, nullable=False, default=0, server_default=text("0"))
    next_attempt_at = Column(DateTime(timezone=True), nullable=True)
    last_error_code = Column(String(128), nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    operation = relationship("ExternalOperation", lazy="select")


class ExternalBackendAccountCreate(_RequestModel):
    name: str = Field(min_length=1, max_length=128)
    adapter: str = Field(min_length=1, max_length=64)
    base_url: HttpUrl
    credentials: dict[str, SecretStr] = Field(min_length=1, max_length=32, repr=False)
    auth_header_templates: list[ExternalAuthHeaderTemplate] = Field(
        min_length=1, max_length=32
    )
    connection_config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value

    @field_validator("adapter")
    @classmethod
    def validate_adapter(cls, value: str) -> str:
        return _validate_identifier(value, "adapter")

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: HttpUrl) -> HttpUrl:
        _validate_base_url(str(value))
        return value

    @field_validator("credentials")
    @classmethod
    def validate_credentials(cls, value: dict[str, SecretStr]) -> dict[str, SecretStr]:
        return _validate_credentials(value)

    @field_validator("auth_header_templates")
    @classmethod
    def validate_auth_header_templates(
        cls, value: list[ExternalAuthHeaderTemplate]
    ) -> list[ExternalAuthHeaderTemplate]:
        _validate_auth_header_templates(value)
        return value

    @field_validator("connection_config")
    @classmethod
    def validate_connection_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_connection_config(value)

    @model_validator(mode="after")
    def validate_auth_references(self):
        available = set(self.credentials)
        referenced = {
            reference
            for template in self.auth_header_templates
            for reference in template.references.values()
        }
        if not referenced.issubset(available):
            raise ValueError(
                "auth header templates reference an unknown credential name"
            )
        return self


class ExternalBackendAccountUpdate(_RequestModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    base_url: HttpUrl | None = None
    credentials: dict[str, SecretStr] | None = Field(
        default=None, min_length=1, max_length=32, repr=False
    )
    remove_credentials: list[str] = Field(default_factory=list, max_length=32)
    auth_header_templates: list[ExternalAuthHeaderTemplate] | None = Field(
        default=None, min_length=1, max_length=32
    )
    connection_config: dict[str, Any] | None = None
    enabled: bool | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: HttpUrl | None) -> HttpUrl | None:
        if value is not None:
            _validate_base_url(str(value))
        return value

    @field_validator("credentials")
    @classmethod
    def validate_credentials(
        cls, value: dict[str, SecretStr] | None
    ) -> dict[str, SecretStr] | None:
        return _validate_credentials(value) if value is not None else None

    @field_validator("remove_credentials")
    @classmethod
    def validate_remove_credentials(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("remove_credentials must not contain duplicates")
        for name in value:
            _validate_identifier(name, "credential name")
        return value

    @field_validator("auth_header_templates")
    @classmethod
    def validate_auth_header_templates(
        cls, value: list[ExternalAuthHeaderTemplate] | None
    ) -> list[ExternalAuthHeaderTemplate] | None:
        if value is not None:
            _validate_auth_header_templates(value)
        return value

    @field_validator("connection_config")
    @classmethod
    def validate_connection_config(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        return _validate_connection_config(value) if value is not None else None

    @model_validator(mode="after")
    def validate_credential_changes(self):
        for field_name in ("name", "base_url", "connection_config", "enabled"):
            if (
                field_name in self.model_fields_set
                and getattr(self, field_name) is None
            ):
                raise ValueError(f"{field_name} cannot be null")
        if self.credentials and set(self.credentials) & set(self.remove_credentials):
            raise ValueError("a credential cannot be updated and removed together")
        return self


class ExternalAccountBulkCancelRequest(_RequestModel):
    reason: str = Field(min_length=1, max_length=2048)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("reason cannot be empty")
        return normalized


class ExternalCredentialForceRotateRequest(_RequestModel):
    reason: str = Field(min_length=1, max_length=2048)
    credentials: dict[str, SecretStr] = Field(min_length=1, max_length=32, repr=False)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("reason cannot be empty")
        return normalized

    @field_validator("credentials")
    @classmethod
    def validate_credentials(cls, value: dict[str, SecretStr]) -> dict[str, SecretStr]:
        return _validate_credentials(value)


class ExternalBackendAccountResponse(BaseModel):
    model_config = {"from_attributes": True}

    account_id: str
    user_id: str
    name: str
    adapter: str
    base_url: str
    credential_references: dict[str, str] = Field(exclude=True, repr=False)
    auth_header_templates: list[ExternalAuthHeaderTemplate]
    connection_config: dict[str, Any]
    management_metadata: dict[str, Any] = Field(
        default_factory=dict, exclude=True, repr=False
    )
    enabled: bool
    artifact_relay_invalidated_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    @computed_field
    @property
    def credential_configured(self) -> bool:
        return bool(self.credential_references)

    @computed_field
    @property
    def credential_names(self) -> list[str]:
        return sorted(self.credential_references)


class ExternalAccountBulkCancelResponse(BaseModel):
    account_id: str
    action_id: str
    cancel_requested: int = Field(ge=0)
    pending_deferred: int = Field(ge=0)
    task_woken: int = Field(ge=0)
    local_sessions: int = Field(ge=0)
    not_cancellable: int = Field(ge=0)


class ExternalCredentialForceRotateResponse(ExternalAccountBulkCancelResponse):
    credential_names: list[str]
    account_disabled: bool
    artifact_relays_invalidated_at: datetime


class ExternalChuteBindingCreate(_RequestModel):
    chute_id: str = Field(min_length=1, max_length=128)
    account_id: str = Field(min_length=1, max_length=128)
    routes: list[ExternalRouteConfig] = Field(min_length=1, max_length=100)
    enabled: bool = True

    @field_validator("routes")
    @classmethod
    def validate_routes(
        cls, value: list[ExternalRouteConfig]
    ) -> list[ExternalRouteConfig]:
        _validate_routes(value)
        return value


class ExternalChuteBindingUpdate(_RequestModel):
    account_id: str | None = Field(default=None, min_length=1, max_length=128)
    routes: list[ExternalRouteConfig] | None = Field(
        default=None, min_length=1, max_length=100
    )
    enabled: bool | None = None

    @field_validator("routes")
    @classmethod
    def validate_routes(
        cls, value: list[ExternalRouteConfig] | None
    ) -> list[ExternalRouteConfig] | None:
        if value is not None:
            _validate_routes(value)
        return value

    @model_validator(mode="after")
    def validate_non_null_updates(self):
        for field_name in ("account_id", "routes", "enabled"):
            if (
                field_name in self.model_fields_set
                and getattr(self, field_name) is None
            ):
                raise ValueError(f"{field_name} cannot be null")
        return self


class ExternalChuteBindingResponse(BaseModel):
    model_config = {"from_attributes": True}

    binding_id: str
    chute_id: str
    account_id: str
    routes: list[ExternalRouteConfig]
    enabled: bool
    created_at: datetime
    updated_at: datetime


def _validate_route_cords(
    cords: list[ExternalCord], routes: list[ExternalRouteConfig]
) -> None:
    cord_paths = [cord.path for cord in cords]
    route_paths = [route.cord_path for route in routes]
    if len(cord_paths) != len(set(cord_paths)):
        raise ValueError("cord paths must be unique")
    selectors = [
        (cord.public_api_path, cord.public_api_method.upper(), bool(cord.stream))
        for cord in cords
    ]
    if len(selectors) != len(set(selectors)):
        raise ValueError(
            "public Cord path, method, and stream selectors must be unique"
        )
    if len(route_paths) != len(set(route_paths)) or set(cord_paths) != set(route_paths):
        raise ValueError("routes must match the configured cord paths")


def _validate_pricing_rules(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not value:
        raise ValueError("pricing_rules must contain at least one rule")
    try:
        parse_pricing_rules(value)
    except PricingConfigurationError as exc:
        raise ValueError(str(exc)) from exc
    return value


class ExternalChuteCreate(_RequestModel):
    account_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=3, max_length=127)
    tagline: str = Field(default="", max_length=1024)
    readme: str = Field(default="", max_length=16384)
    tool_description: str | None = Field(default=None, max_length=16384)
    logo_id: str | None = None
    public: bool = False
    standard_template: str | None = None
    cords: list[ExternalCord] = Field(min_length=1, max_length=100)
    routes: list[ExternalRouteConfig] = Field(min_length=1, max_length=100)
    pricing_rules: list[dict[str, Any]] = Field(min_length=1, max_length=1000)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value

    @field_validator("standard_template")
    @classmethod
    def validate_standard_template(cls, value: str | None) -> str | None:
        if value is not None and value not in STANDARD_TEMPLATES:
            raise ValueError(f"Invalid standard template: {value}")
        return value

    @field_validator("routes")
    @classmethod
    def validate_routes(
        cls, value: list[ExternalRouteConfig]
    ) -> list[ExternalRouteConfig]:
        _validate_routes(value)
        return value

    @field_validator("pricing_rules")
    @classmethod
    def validate_pricing_rules(
        cls, value: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return _validate_pricing_rules(value)

    @model_validator(mode="after")
    def validate_route_cords(self):
        _validate_route_cords(self.cords, self.routes)
        return self


class ExternalChuteUpdate(_RequestModel):
    tagline: str | None = Field(default=None, max_length=1024)
    readme: str | None = Field(default=None, max_length=16384)
    tool_description: str | None = Field(default=None, max_length=16384)
    logo_id: str | None = None
    public: bool | None = None
    standard_template: str | None = None
    account_id: str | None = Field(default=None, min_length=1, max_length=128)
    cords: list[ExternalCord] | None = Field(default=None, min_length=1, max_length=100)
    routes: list[ExternalRouteConfig] | None = Field(
        default=None, min_length=1, max_length=100
    )
    pricing_rules: list[dict[str, Any]] | None = Field(
        default=None, min_length=1, max_length=1000
    )
    enabled: bool | None = None

    @field_validator("routes")
    @classmethod
    def validate_routes(
        cls, value: list[ExternalRouteConfig] | None
    ) -> list[ExternalRouteConfig] | None:
        if value is not None:
            _validate_routes(value)
        return value

    @field_validator("standard_template")
    @classmethod
    def validate_standard_template(cls, value: str | None) -> str | None:
        if value is not None and value not in STANDARD_TEMPLATES:
            raise ValueError(f"Invalid standard template: {value}")
        return value

    @field_validator("pricing_rules")
    @classmethod
    def validate_pricing_rules(
        cls, value: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        return _validate_pricing_rules(value) if value is not None else None

    @model_validator(mode="after")
    def validate_route_cords(self):
        for field_name in (
            "tagline",
            "readme",
            "public",
            "account_id",
            "cords",
            "routes",
            "pricing_rules",
            "enabled",
        ):
            if (
                field_name in self.model_fields_set
                and getattr(self, field_name) is None
            ):
                raise ValueError(f"{field_name} cannot be null")
        if (self.cords is None) != (self.routes is None):
            raise ValueError("cords and routes must be updated together")
        if self.cords is not None and self.routes is not None:
            _validate_route_cords(self.cords, self.routes)
            if self.pricing_rules is None:
                raise ValueError("pricing_rules are required when routes change")
        return self


class ExternalChuteResponse(BaseModel):
    model_config = {"from_attributes": True}

    chute_id: str
    user_id: str
    name: str
    tagline: str
    readme: str
    tool_description: str | None
    logo_id: str | None
    public: bool
    standard_template: str | None
    slug: str
    version: str
    execution_backend: str
    cords: list[ExternalCord]
    binding: ExternalChuteBindingResponse | None
    pricing_rules: list[dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class ExternalOperationCreate(_RequestModel):
    binding_id: str = Field(min_length=1, max_length=128)
    cord_path: str = Field(min_length=1, max_length=255)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=255)
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    expires_at: datetime | None = None

    @field_validator("cord_path")
    @classmethod
    def validate_cord_path(cls, value: str) -> str:
        if not _CORD_PATH_PATTERN.fullmatch(value):
            raise ValueError("cord_path must be an absolute Chute cord path")
        return value

    @field_validator("request_metadata")
    @classmethod
    def validate_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_non_secret_json(value, "request_metadata")


class ExternalOperationUpdate(_RequestModel):
    status: ExternalOperationStatus | None = None
    settlement_status: ExternalSettlementStatus | None = None
    upstream_operation_id: str | None = Field(
        default=None, min_length=1, max_length=512
    )
    upstream_status: str | None = Field(default=None, min_length=1, max_length=128)
    upstream_metadata: dict[str, Any] | None = None
    usage: ExternalUsageDescriptor | None = None
    result_descriptor: ExternalResultDescriptor | None = None
    error: ExternalErrorDescriptor | None = None
    settlement_metadata: dict[str, Any] | None = None
    poll_attempts: int | None = Field(default=None, ge=0)
    submitted_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    last_polled_at: datetime | None = None
    next_poll_at: datetime | None = None
    lease_owner: str | None = Field(default=None, min_length=1, max_length=255)
    lease_expires_at: datetime | None = None
    expires_at: datetime | None = None
    settled_at: datetime | None = None

    @field_validator("upstream_metadata", "settlement_metadata")
    @classmethod
    def validate_metadata(
        cls, value: dict[str, Any] | None, info
    ) -> dict[str, Any] | None:
        return _validate_optional_non_secret_json(value, info.field_name)


class ExternalSettlementRetryRequest(_RequestModel):
    reason: str = Field(min_length=1, max_length=2048)
    usage: ExternalUsageDescriptor | None = None
    pricing_snapshot: dict[str, Any] | None = None
    expected_pricing_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    customer_authorized_max_amount: Decimal | None = Field(
        default=None, ge=0, max_digits=20, decimal_places=8
    )

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("reason cannot be empty")
        return normalized

    @field_validator("pricing_snapshot")
    @classmethod
    def validate_pricing_snapshot(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        return _validate_optional_non_secret_json(value, "pricing_snapshot")

    @model_validator(mode="after")
    def validate_pricing_correction_contract(self):
        correction_fields = (
            self.pricing_snapshot,
            self.expected_pricing_sha256,
            self.customer_authorized_max_amount,
        )
        if any(value is not None for value in correction_fields) and not all(
            value is not None for value in correction_fields
        ):
            raise ValueError(
                "pricing_snapshot, expected_pricing_sha256, and "
                "customer_authorized_max_amount must be supplied together"
            )
        return self


class ExternalSettlementWriteOffRequest(_RequestModel):
    reason: str = Field(min_length=1, max_length=2048)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("reason cannot be empty")
        return normalized


class ExternalArtifactResponse(BaseModel):
    """Public artifact metadata without its private upstream locator."""

    model_config = {"from_attributes": True, "extra": "ignore"}

    kind: str
    content_type: str | None = None
    size_bytes: int | None = None
    expires_at: datetime | None = None


class ExternalResultResponse(BaseModel):
    """Public result shape; relay references and upstream metadata stay internal."""

    model_config = {"from_attributes": True, "extra": "ignore"}

    status: ExternalResultStatus = ExternalResultStatus.COMPLETE
    artifacts: list[ExternalArtifactResponse] = Field(default_factory=list)


class ExternalOperationResponse(BaseModel):
    model_config = {"from_attributes": True}

    operation_id: str
    user_id: str | None
    chute_id: str | None
    cord_path: str
    operation_mode: ExternalOperationMode
    protocol: str
    status: ExternalOperationStatus
    settlement_status: ExternalSettlementStatus
    usage: ExternalUsageDescriptor | None
    settlement_metadata: dict[str, Any] = Field(
        default_factory=dict, exclude=True, repr=False
    )
    result_descriptor: ExternalResultResponse | None
    error: ExternalErrorDescriptor | None
    created_at: datetime
    updated_at: datetime
    submitted_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    expires_at: datetime | None
    settled_at: datetime | None

    @computed_field
    @property
    def pricing_snapshot_sha256(self) -> str | None:
        pricing = self.settlement_metadata.get("pricing")
        if not isinstance(pricing, dict):
            return None
        try:
            payload = orjson.dumps(pricing, option=orjson.OPT_SORT_KEYS)
        except (TypeError, orjson.JSONEncodeError):
            return None
        return hashlib.sha256(payload).hexdigest()
