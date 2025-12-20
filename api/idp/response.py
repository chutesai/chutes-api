"""
Response models for OAuth2/IDP functionality.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class OAuthAppResponse(BaseModel):
    """Response model for an OAuth application (public info)."""

    app_id: str
    client_id: str
    user_id: str
    name: str
    description: Optional[str] = None
    redirect_uris: List[str]
    homepage_url: Optional[str] = None
    logo_url: Optional[str] = None
    active: bool
    public: bool
    refresh_token_lifetime_days: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class OAuthAppCreationResponse(OAuthAppResponse):
    """Response model when creating an OAuth application (includes secret)."""

    client_secret: Optional[str] = None


class OAuthAppSecretRegenerateResponse(BaseModel):
    """Response model when regenerating a client secret."""

    app_id: str
    client_id: str
    client_secret: str


class OAuthAuthorizationResponse(BaseModel):
    """Response model for a user's authorization to an app."""

    authorization_id: str
    app_id: str
    app_name: str
    app_description: Optional[str] = None
    app_logo_url: Optional[str] = None
    scopes: List[str]
    created_at: datetime
    revoked: bool

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """OAuth2 token response following RFC 6749."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    scope: Optional[str] = None


class TokenErrorResponse(BaseModel):
    """OAuth2 error response following RFC 6749."""

    error: str
    error_description: Optional[str] = None
    error_uri: Optional[str] = None


class AuthorizeRequest(BaseModel):
    """OAuth2 authorization request parameters."""

    response_type: str
    client_id: str
    redirect_uri: str
    scope: Optional[str] = None
    state: Optional[str] = None
    code_challenge: Optional[str] = None  # PKCE
    code_challenge_method: Optional[str] = None  # PKCE


class TokenRequest(BaseModel):
    """OAuth2 token request parameters."""

    grant_type: str
    code: Optional[str] = None
    redirect_uri: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    refresh_token: Optional[str] = None
    code_verifier: Optional[str] = None  # PKCE


class UserInfoResponse(BaseModel):
    """User info response for authenticated users."""

    sub: str  # user_id
    username: str
    created_at: datetime

    class Config:
        from_attributes = True
