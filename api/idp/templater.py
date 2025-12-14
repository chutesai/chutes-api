"""
HTML templates for OAuth2/IDP login and authorization pages.
Uses Jinja2 templates styled to match chutes.ai design.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# Set up Jinja2 environment
_template_dir = Path(__file__).parent / "templates"
_env = Environment(
    loader=FileSystemLoader(_template_dir),
    autoescape=True,
)


def _format_lifetime_text(days: int) -> str:
    """Format refresh token lifetime as human-readable text."""
    if days == 1:
        return "1 day"
    elif days < 7:
        return f"{days} days"
    elif days == 7:
        return "1 week"
    elif days < 30:
        weeks = days // 7
        return f"{weeks} week{'s' if weeks > 1 else ''}"
    else:
        return "30 days"


def login_page(
    client_id: str,
    redirect_uri: str,
    state: str = "",
    scope: str = "",
    app_name: str = "",
    app_description: str = "",
    nonce: str = "",
    error: str = "",
    code_challenge: str = "",
    code_challenge_method: str = "",
    create_account_url: str = "",
    login_url: str = "",
) -> str:
    """Generate the login page HTML."""
    template = _env.get_template("login.jinja2")
    return template.render(
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        scope=scope,
        app_name=app_name,
        app_description=app_description,
        nonce=nonce,
        error=error,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        create_account_url=create_account_url,
        login_url=login_url,
    )


def authorize_page(
    client_id: str,
    redirect_uri: str,
    state: str = "",
    scope: str = "",
    app_name: str = "",
    app_description: str = "",
    app_logo_url: str = "",
    user_name: str = "",
    scopes: list = None,
    code_challenge: str = "",
    code_challenge_method: str = "",
    refresh_token_lifetime_days: int = 30,
) -> str:
    """Generate the authorization consent page HTML."""
    scopes = scopes or ["Read your profile information"]
    lifetime_text = _format_lifetime_text(refresh_token_lifetime_days)

    template = _env.get_template("authorize.jinja2")
    return template.render(
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        scope=scope,
        app_name=app_name,
        app_description=app_description,
        app_logo_url=app_logo_url,
        user_name=user_name,
        scopes=scopes,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        lifetime_text=lifetime_text,
    )


def error_page(error: str, error_description: str = "") -> str:
    """Generate an error page HTML."""
    template = _env.get_template("error.jinja2")
    return template.render(
        error=error,
        error_description=error_description,
    )


def success_page(message: str, redirect_url: str = "") -> str:
    """Generate a success page HTML."""
    template = _env.get_template("success.jinja2")
    return template.render(
        message=message,
        redirect_url=redirect_url,
    )
