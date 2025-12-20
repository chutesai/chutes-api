"""
HTML templates for user registration pages.
Uses Jinja2 templates.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# Set up Jinja2 environment
_template_dir = Path(__file__).parent / "templates"
_env = Environment(
    loader=FileSystemLoader(_template_dir),
    autoescape=True,
)


def registration_token_form(hcaptcha_sitekey: str) -> str:
    """Generate the registration token request form with hCaptcha."""
    template = _env.get_template("registration_token_form.jinja2")
    return template.render(hcaptcha_sitekey=hcaptcha_sitekey)


def registration_token_success(token: str) -> str:
    """Generate the registration token success page."""
    template = _env.get_template("registration_token_success.jinja2")
    return template.render(token=token)


def error_page(message: str) -> str:
    """Generate an error page."""
    template = _env.get_template("error.jinja2")
    return template.render(message=message)
