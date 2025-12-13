"""
Database models for OAuth2/IDP functionality.

Scope Format:
-------------
OAuth2 scopes follow a structured format similar to API key scopes:
- "admin" - Full access to all resources (like admin API keys)
- "{object_type}:{action}" - Access to all objects of a type with specific action
- "{object_type}:{object_id}:{action}" - Access to specific object with specific action
- "{object_type}:*:{action}" - Equivalent to {object_type}:{action}

Where:
- object_type: "chutes", "images", "invocations", "account", "billing", "secrets"
- action: "read", "write", "delete", "invoke"
- object_id: specific UUID or "*" for all

Resource Scopes:
- "chutes:read" - Read all chutes
- "chutes:invoke" - Invoke any chute
- "chutes:{id}:invoke" - Invoke specific chute
- "images:read" - Read images
- "images:write" - Create/modify images

Account Scopes (user-friendly aliases):
- "profile" / "profile:read" - Read basic profile info (username, user_id)
- "balance" / "balance:read" - Read account balance
- "billing:read" - Read billing info, payment history
- "quota" / "quota:read" - Read quota information
- "usage" / "usage:read" - Read usage statistics and invocation history
- "account:read" - Read full account details (quotas, discounts, pricing)
- "account:write" - Modify account settings
- "secrets:read" - Read secret names (not values)
- "secrets:write" - Create and manage secrets

Simple action scopes (apply to all object types):
- "read", "write", "delete", "invoke"
"""

import hashlib
import re
import secrets
import string
from typing import List, Optional, Self

from sqlalchemy import (
    Column,
    String,
    ForeignKey,
    DateTime,
    Integer,
    func,
    Boolean,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship, validates
from passlib.hash import argon2
from pydantic import BaseModel, field_validator
from api.constants import (
    MAX_REFRESH_TOKEN_LIFETIME_DAYS,
    DEFAULT_REFRESH_TOKEN_LIFETIME_DAYS,
)
from api.database import Base, generate_uuid


# Valid object types and actions for scopes
VALID_OBJECT_TYPES = ("chutes", "images", "invocations", "account", "billing", "secrets")
VALID_ACTIONS = ("read", "write", "delete", "invoke")

# Special account-related scopes (these are standalone, not object:action format)
ACCOUNT_SCOPES = {
    # Profile and basic account info
    "profile": "Read basic profile information (username, user_id)",
    "profile:read": "Read basic profile information (username, user_id)",
    # Balance and billing
    "balance": "Read account balance",
    "balance:read": "Read account balance",
    "billing:read": "Read billing information, payment history",
    # Quotas and usage
    "quota": "Read quota information and usage",
    "quota:read": "Read quota information and usage",
    "usage": "Read usage statistics and invocation history",
    "usage:read": "Read usage statistics and invocation history",
    # Account management (more privileged)
    "account:read": "Read full account details including quotas, discounts, pricing",
    "account:write": "Modify account settings",
    # Secrets
    "secrets:read": "Read secret names (not values)",
    "secrets:write": "Create and manage secrets",
}


def parse_scope(scope: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse a scope string into (object_type, object_id, action).

    Returns:
        tuple of (object_type, object_id, action) where any can be None for wildcards
    """
    if scope == "admin":
        return (None, None, None)  # Full access

    # Handle account-related scopes
    if scope in ACCOUNT_SCOPES:
        # Map simple scopes to their object:action equivalent
        if scope == "profile" or scope == "profile:read":
            return ("account", None, "read")
        if scope == "balance" or scope == "balance:read":
            return ("billing", None, "read")
        if scope == "quota" or scope == "quota:read":
            return ("account", None, "read")
        if scope == "usage" or scope == "usage:read":
            return ("invocations", None, "read")
        if scope in (
            "billing:read",
            "account:read",
            "account:write",
            "secrets:read",
            "secrets:write",
        ):
            parts = scope.split(":")
            return (parts[0], None, parts[1])

    # Handle legacy simple scopes
    if scope in VALID_ACTIONS:
        # Simple action scope applies to all object types
        return (None, None, scope)

    parts = scope.split(":")
    if len(parts) == 2:
        # Format: object_type:action
        object_type, action = parts
        if object_type in VALID_OBJECT_TYPES and action in VALID_ACTIONS:
            return (object_type, None, action)
    elif len(parts) == 3:
        # Format: object_type:object_id:action
        object_type, object_id, action = parts
        if object_type in VALID_OBJECT_TYPES and action in VALID_ACTIONS:
            if object_id == "*":
                object_id = None
            return (object_type, object_id, action)

    # Unknown scope format - return as-is for custom handling
    return (scope, None, None)


def check_scope_access(
    scopes: List[str],
    required_object_type: str,
    required_object_id: Optional[str],
    required_action: str,
) -> bool:
    """
    Check if a list of scopes grants access to a specific resource and action.

    Args:
        scopes: List of scope strings
        required_object_type: The object type being accessed (e.g., "chutes")
        required_object_id: The specific object ID (can be None for list operations)
        required_action: The action being performed (e.g., "invoke")

    Returns:
        True if access is granted, False otherwise
    """
    for scope in scopes:
        obj_type, obj_id, action = parse_scope(scope)

        # Admin scope grants full access
        if scope == "admin" or (obj_type is None and obj_id is None and action is None):
            return True

        # Simple action scope (e.g., "read", "invoke") matches any object type
        if obj_type is None and action == required_action:
            return True

        # Check object type match
        if obj_type is not None and obj_type != required_object_type:
            continue

        # Check action match
        if action is not None and action != required_action:
            continue

        # Check object ID match (None in scope means wildcard)
        if obj_id is not None and obj_id != required_object_id:
            continue

        # All checks passed
        return True

    return False


def format_scope(object_type: str, object_id: Optional[str], action: str) -> str:
    """Format a scope string from components."""
    if object_id:
        return f"{object_type}:{object_id}:{action}"
    return f"{object_type}:{action}"


def get_available_scopes() -> dict:
    """
    Get all available scopes with descriptions.
    Useful for documentation and scope selection UIs.
    """
    scopes = {
        "admin": "Full access to all resources and actions",
    }

    # Add account scopes
    scopes.update(ACCOUNT_SCOPES)

    # Add resource scopes
    resource_scopes = {
        "chutes:read": "Read chute information and list chutes",
        "chutes:write": "Create and modify chutes",
        "chutes:delete": "Delete chutes",
        "chutes:invoke": "Invoke/run chutes",
        "images:read": "Read image information and list images",
        "images:write": "Create and modify images",
        "images:delete": "Delete images",
        "invocations:read": "Read invocation history and details",
    }
    scopes.update(resource_scopes)

    return scopes


def get_scope_description(scope: str) -> str:
    """
    Get a human-readable description for a scope.
    Used for consent pages to explain what permissions are being requested.
    """
    all_scopes = get_available_scopes()

    # Direct match
    if scope in all_scopes:
        return all_scopes[scope]

    # Handle object:action format
    parts = scope.split(":")
    if len(parts) == 2:
        obj_type, action = parts
        action_verbs = {
            "read": "Read",
            "write": "Create and modify",
            "delete": "Delete",
            "invoke": "Invoke/run",
        }
        obj_names = {
            "chutes": "chutes",
            "images": "images",
            "invocations": "invocation history",
            "account": "account information",
            "billing": "billing information",
            "secrets": "secrets",
        }
        verb = action_verbs.get(action, action.capitalize())
        obj = obj_names.get(obj_type, obj_type)
        return f"{verb} {obj}"

    # Handle object:id:action format
    if len(parts) == 3:
        obj_type, obj_id, action = parts
        action_verbs = {
            "read": "Read",
            "write": "Modify",
            "delete": "Delete",
            "invoke": "Invoke",
        }
        verb = action_verbs.get(action, action.capitalize())
        return f"{verb} specific {obj_type[:-1] if obj_type.endswith('s') else obj_type} ({obj_id[:8]}...)"

    # Fallback
    return f"Access: {scope}"


def get_scope_descriptions(scopes: List[str]) -> List[str]:
    """
    Get human-readable descriptions for a list of scopes.
    Used for consent pages.
    """
    return [get_scope_description(s) for s in scopes]


# Default scopes that any app can request without explicit registration
DEFAULT_ALLOWED_SCOPES = {
    "profile",
    "openid",
    "account:read",
    "chutes:read",
    "chutes:invoke",
    "images:read",
    "invocations:read",
}

# Privileged scopes that require explicit app registration
PRIVILEGED_SCOPES = {
    "admin",
    "account:write",
    "account:delete",
    "billing:read",
    "billing:write",
    "secrets:read",
    "secrets:write",
    "secrets:delete",
    "chutes:write",
    "chutes:delete",
    "images:write",
    "images:delete",
}


def validate_requested_scopes(
    requested_scopes: List[str],
    app_allowed_scopes: Optional[List[str]],
) -> tuple[bool, Optional[str], List[str]]:
    """
    Validate that the requested scopes are allowed for this app.

    Returns (is_valid, error_message, filtered_scopes).
    - filtered_scopes contains only the valid scopes that can be granted.
    """
    all_available = get_available_scopes()
    valid_scopes = []
    app_allowed = set(app_allowed_scopes or [])

    for scope in requested_scopes:
        # Check if scope exists (allow openid/profile as standard OIDC scopes)
        if scope not in all_available and scope not in ("openid", "profile"):
            return False, f"Unknown scope: {scope}", []

        # Check if scope requires explicit registration
        if scope in PRIVILEGED_SCOPES:
            if scope not in app_allowed:
                return False, f"Scope '{scope}' requires explicit app registration", []

        valid_scopes.append(scope)

    return True, None, valid_scopes


class OAuthAppCreateRequest(BaseModel):
    """Request model for creating an OAuth application."""

    name: str
    description: Optional[str] = None
    redirect_uris: List[str]
    homepage_url: Optional[str] = None
    logo_url: Optional[str] = None
    public: bool = True
    refresh_token_lifetime_days: Optional[int] = DEFAULT_REFRESH_TOKEN_LIFETIME_DAYS
    # Scopes this app is allowed to request (if empty, allows basic scopes only)
    allowed_scopes: Optional[List[str]] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v or len(v) < 3 or len(v) > 64:
            raise ValueError("Name must be between 3 and 64 characters")
        if not re.match(r"^[\w\s\-\.]+$", v):
            raise ValueError("Name can only contain letters, numbers, spaces, hyphens, and periods")
        return v

    @field_validator("redirect_uris")
    @classmethod
    def validate_redirect_uris(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one redirect URI is required")
        if len(v) > 10:
            raise ValueError("Maximum 10 redirect URIs allowed")
        for uri in v:
            if not uri.startswith(("http://", "https://")):
                raise ValueError(f"Invalid redirect URI: {uri}")
        return v

    @field_validator("refresh_token_lifetime_days")
    @classmethod
    def validate_refresh_token_lifetime(cls, v):
        if v is not None:
            if v < 1:
                raise ValueError("Refresh token lifetime must be at least 1 day")
            if v > MAX_REFRESH_TOKEN_LIFETIME_DAYS:
                raise ValueError(
                    f"Refresh token lifetime cannot exceed {MAX_REFRESH_TOKEN_LIFETIME_DAYS} days"
                )
        return v


class OAuthAppUpdateRequest(BaseModel):
    """Request model for updating an OAuth application."""

    name: Optional[str] = None
    description: Optional[str] = None
    redirect_uris: Optional[List[str]] = None
    homepage_url: Optional[str] = None
    logo_url: Optional[str] = None
    active: Optional[bool] = None
    public: Optional[bool] = None
    refresh_token_lifetime_days: Optional[int] = None
    allowed_scopes: Optional[List[str]] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if v is not None:
            if len(v) < 3 or len(v) > 64:
                raise ValueError("Name must be between 3 and 64 characters")
            if not re.match(r"^[\w\s\-\.]+$", v):
                raise ValueError(
                    "Name can only contain letters, numbers, spaces, hyphens, and periods"
                )
        return v

    @field_validator("redirect_uris")
    @classmethod
    def validate_redirect_uris(cls, v):
        if v is not None:
            if len(v) == 0:
                raise ValueError("At least one redirect URI is required")
            if len(v) > 10:
                raise ValueError("Maximum 10 redirect URIs allowed")
            for uri in v:
                if not uri.startswith(("http://", "https://")):
                    raise ValueError(f"Invalid redirect URI: {uri}")
        return v

    @field_validator("refresh_token_lifetime_days")
    @classmethod
    def validate_refresh_token_lifetime(cls, v):
        if v is not None:
            if v < 1:
                raise ValueError("Refresh token lifetime must be at least 1 day")
            if v > MAX_REFRESH_TOKEN_LIFETIME_DAYS:
                raise ValueError(
                    f"Refresh token lifetime cannot exceed {MAX_REFRESH_TOKEN_LIFETIME_DAYS} days"
                )
        return v


class OAuthApp(Base):
    """OAuth2 Application model."""

    __tablename__ = "oauth_apps"

    app_id = Column(String, primary_key=True, default=generate_uuid)
    client_id = Column(String, unique=True, nullable=False, index=True)
    client_secret_hash = Column(String, nullable=False)
    user_id = Column(
        String,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    name = Column(String(64), nullable=False)
    description = Column(Text, nullable=True)
    redirect_uris = Column(ARRAY(String), nullable=False)
    homepage_url = Column(String, nullable=True)
    logo_url = Column(String, nullable=True)
    active = Column(Boolean, default=True, nullable=False)
    public = Column(Boolean, default=True, nullable=False)
    refresh_token_lifetime_days = Column(
        Integer, default=DEFAULT_REFRESH_TOKEN_LIFETIME_DAYS, nullable=False
    )
    # Scopes this app is allowed to request. If NULL/empty, only basic scopes allowed.
    # "admin" scope requires explicit registration.
    allowed_scopes = Column(ARRAY(String), nullable=True, default=list)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", backref="oauth_apps")
    authorizations = relationship(
        "OAuthAuthorization",
        back_populates="app",
        cascade="all, delete-orphan",
    )
    shares = relationship(
        "OAuthAppShare",
        back_populates="app",
        cascade="all, delete-orphan",
        foreign_keys="OAuthAppShare.app_id",
    )

    __table_args__ = (UniqueConstraint("name", name="constraint_oauth_app_name"),)

    @validates("name")
    def validate_name(self, _, name):
        if not name or len(name) < 3 or len(name) > 64:
            raise ValueError("Name must be between 3 and 64 characters")
        if not re.match(r"^[\w\s\-\.]+$", name):
            raise ValueError("Name can only contain letters, numbers, spaces, hyphens, and periods")
        return name

    @classmethod
    def generate_client_id(cls) -> str:
        """Generate a unique client ID."""
        return f"cid_{''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(24))}"

    @classmethod
    def generate_client_secret(cls) -> str:
        """Generate a secure client secret."""
        return f"csc_{''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(48))}"

    @classmethod
    def create(cls, user_id: str, args: OAuthAppCreateRequest) -> tuple[Self, str]:
        """Create a new OAuth application with generated credentials."""
        client_id = cls.generate_client_id()
        client_secret = cls.generate_client_secret()
        instance = cls(
            app_id=generate_uuid(),
            client_id=client_id,
            client_secret_hash=argon2.hash(client_secret),
            user_id=user_id,
            name=args.name,
            description=args.description,
            redirect_uris=args.redirect_uris,
            homepage_url=args.homepage_url,
            logo_url=args.logo_url,
            public=args.public,
            refresh_token_lifetime_days=args.refresh_token_lifetime_days
            or DEFAULT_REFRESH_TOKEN_LIFETIME_DAYS,
            allowed_scopes=args.allowed_scopes or [],
        )
        return instance, client_secret

    def verify_secret(self, secret: str) -> bool:
        """Verify the client secret."""
        return argon2.verify(secret, self.client_secret_hash)

    def regenerate_secret(self) -> str:
        """Regenerate the client secret."""
        new_secret = self.generate_client_secret()
        self.client_secret_hash = argon2.hash(new_secret)
        return new_secret

    def is_valid_redirect_uri(self, uri: str) -> bool:
        """Check if a redirect URI is registered for this app."""
        return uri in self.redirect_uris


class OAuthAppShare(Base):
    """
    Represents explicit sharing of an OAuth app with a specific user.
    Similar to ChuteShare - allows app owners to share their apps with specific users.
    """

    __tablename__ = "oauth_app_shares"

    app_id = Column(
        String,
        ForeignKey("oauth_apps.app_id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
    )
    shared_by = Column(
        String,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
    )
    shared_to = Column(
        String,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        primary_key=True,
    )
    shared_at = Column(DateTime, server_default=func.now())

    # Relationships
    app = relationship("OAuthApp", back_populates="shares")
    shared_by_user = relationship("User", foreign_keys=[shared_by])
    shared_to_user = relationship("User", foreign_keys=[shared_to])


class OAuthAppShareArgs(BaseModel):
    """Request model for sharing an OAuth app."""

    app_id_or_name: str
    user_id_or_name: str


class OAuthAuthorization(Base):
    """
    Represents a user's authorization grant to an OAuth application.
    This tracks which users have authorized which apps and with what scopes.
    """

    __tablename__ = "oauth_authorizations"

    authorization_id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(
        String,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    app_id = Column(
        String,
        ForeignKey("oauth_apps.app_id", ondelete="CASCADE"),
        nullable=False,
    )
    scopes = Column(ARRAY(String), nullable=False, default=list)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    revoked = Column(Boolean, default=False, nullable=False)
    revoked_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", backref="oauth_authorizations")
    app = relationship("OAuthApp", back_populates="authorizations")
    access_tokens = relationship(
        "OAuthAccessToken",
        back_populates="authorization",
        cascade="all, delete-orphan",
    )
    refresh_tokens = relationship(
        "OAuthRefreshToken",
        back_populates="authorization",
        cascade="all, delete-orphan",
    )

    __table_args__ = (UniqueConstraint("user_id", "app_id", name="constraint_oauth_auth_user_app"),)


class OAuthAccessToken(Base):
    """OAuth2 Access Token model."""

    __tablename__ = "oauth_access_tokens"

    token_id = Column(String, primary_key=True, default=generate_uuid)
    token_hash = Column(String, nullable=False)
    authorization_id = Column(
        String,
        ForeignKey("oauth_authorizations.authorization_id", ondelete="CASCADE"),
        nullable=False,
    )
    scopes = Column(ARRAY(String), nullable=False, default=list)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    revoked = Column(Boolean, default=False, nullable=False)

    # Relationships
    authorization = relationship("OAuthAuthorization", back_populates="access_tokens")

    @classmethod
    def generate_token(cls, token_id: str) -> str:
        """
        Generate a secure access token with embedded token_id for O(1) lookup.
        Format: cak_{token_id}.{secret}
        """
        secret = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(48))
        return f"cak_{token_id}.{secret}"

    @classmethod
    def hash_token(cls, token: str) -> str:
        """Hash the secret portion of a token for secure storage (argon2)."""
        # Only hash the secret part, not the token_id
        parts = token.split(".", 1)
        if len(parts) != 2:
            return argon2.hash(token)
        return argon2.hash(parts[1])

    @classmethod
    def parse_token(cls, token: str) -> tuple[Optional[str], Optional[str]]:
        """
        Parse a token string into (token_id, secret).
        Returns (None, None) if format is invalid.
        """
        if not token.startswith("cak_"):
            return None, None
        rest = token[4:]  # Remove "cak_" prefix
        parts = rest.split(".", 1)
        if len(parts) != 2:
            return None, None
        return parts[0], parts[1]

    def verify_secret(self, token: str) -> bool:
        """Verify the secret portion of a token against the stored hash."""
        _, secret = self.parse_token(token)
        if not secret:
            return False
        try:
            return argon2.verify(secret, self.token_hash)
        except Exception:
            return False

    @staticmethod
    def could_be_valid(token: str) -> bool:
        """Fast check for token validity format."""
        if not token.startswith("cak_"):
            return False
        rest = token[4:]
        parts = rest.split(".", 1)
        if len(parts) != 2:
            return False
        token_id, secret = parts
        # token_id is UUID (36 chars), secret is 48 chars alphanumeric
        return (
            len(token_id) == 36
            and len(secret) == 48
            and re.match(r"^[a-zA-Z0-9]+$", secret) is not None
        )


class OAuthRefreshToken(Base):
    """OAuth2 Refresh Token model."""

    __tablename__ = "oauth_refresh_tokens"

    token_id = Column(String, primary_key=True, default=generate_uuid)
    token_hash = Column(String, nullable=False)
    authorization_id = Column(
        String,
        ForeignKey("oauth_authorizations.authorization_id", ondelete="CASCADE"),
        nullable=False,
    )
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    revoked = Column(Boolean, default=False, nullable=False)
    used = Column(Boolean, default=False, nullable=False)

    # Relationships
    authorization = relationship("OAuthAuthorization", back_populates="refresh_tokens")

    @classmethod
    def generate_token(cls, token_id: str) -> str:
        """
        Generate a secure refresh token with embedded token_id for O(1) lookup.
        Format: crt_{token_id}.{secret}
        """
        secret = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(48))
        return f"crt_{token_id}.{secret}"

    @classmethod
    def hash_token(cls, token: str) -> str:
        """Hash the secret portion of a token for secure storage (argon2)."""
        parts = token.split(".", 1)
        if len(parts) != 2:
            return argon2.hash(token)
        return argon2.hash(parts[1])

    @classmethod
    def parse_token(cls, token: str) -> tuple[Optional[str], Optional[str]]:
        """
        Parse a token string into (token_id, secret).
        Returns (None, None) if format is invalid.
        """
        if not token.startswith("crt_"):
            return None, None
        rest = token[4:]  # Remove "crt_" prefix
        parts = rest.split(".", 1)
        if len(parts) != 2:
            return None, None
        return parts[0], parts[1]

    def verify_secret(self, token: str) -> bool:
        """Verify the secret portion of a token against the stored hash."""
        _, secret = self.parse_token(token)
        if not secret:
            return False
        try:
            return argon2.verify(secret, self.token_hash)
        except Exception:
            return False

    @staticmethod
    def could_be_valid(token: str) -> bool:
        """Fast check for token validity format."""
        if not token.startswith("crt_"):
            return False
        rest = token[4:]
        parts = rest.split(".", 1)
        if len(parts) != 2:
            return False
        token_id, secret = parts
        return (
            len(token_id) == 36
            and len(secret) == 48
            and re.match(r"^[a-zA-Z0-9]+$", secret) is not None
        )


class OAuthAuthorizationCode(BaseModel):
    """
    Authorization code for OAuth2 authorization code flow.
    Stored in Redis (ephemeral, single-use) rather than the database.
    """

    app_id: str
    user_id: str
    redirect_uri: str
    scopes: List[str] = []
    state: Optional[str] = None
    code_challenge: Optional[str] = None  # PKCE support
    code_challenge_method: Optional[str] = None  # plain or S256

    @staticmethod
    def generate_code() -> str:
        """Generate a secure authorization code."""
        return "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(64))

    @staticmethod
    def hash_code(code: str) -> str:
        """Hash a code for use as Redis key (SHA256, not argon2 - ephemeral data)."""
        return hashlib.sha256(code.encode()).hexdigest()

    @staticmethod
    def redis_key(code_hash: str) -> str:
        """Get the Redis key for a code hash."""
        return f"idp:auth_code:{code_hash}"

    def to_json(self) -> str:
        """Serialize to JSON for Redis storage."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str | bytes) -> "OAuthAuthorizationCode":
        """Deserialize from Redis."""
        if isinstance(data, bytes):
            data = data.decode()
        return cls.model_validate_json(data)
