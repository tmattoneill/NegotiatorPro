"""
Environment Validator

Validates required environment variables based on APP_ENV setting.
- In production: fails startup if critical secrets are missing
- In development: warns but allows defaults
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_app_env() -> str:
    """Get current environment (development or production)"""
    return os.getenv("APP_ENV", "development").lower()


def is_production() -> bool:
    """Check if running in production mode"""
    return get_app_env() == "production"


def get_required_env(
    var_name: str,
    default: Optional[str] = None,
    secret: bool = False
) -> str:
    """
    Get environment variable with production enforcement.

    Args:
        var_name: Name of the environment variable
        default: Default value for development mode
        secret: If True, this is a security-critical variable

    Returns:
        The environment variable value

    Raises:
        ValueError: In production, if a required secret is not set
    """
    value = os.getenv(var_name)

    if value:
        return value

    if is_production() and secret:
        raise ValueError(
            f"CRITICAL: {var_name} must be set in production environment. "
            f"Set APP_ENV=development to use defaults for local development."
        )

    if default is not None:
        if secret:
            logger.warning(
                f"WARNING: {var_name} not set, using default. "
                f"This is INSECURE for production use!"
            )
        return default

    raise ValueError(f"{var_name} is required and has no default value")


def validate_startup_environment() -> dict:
    """
    Validate all critical environment variables at startup.

    Returns:
        Dict with validation results

    Raises:
        ValueError: In production, if critical variables are missing
    """
    env = get_app_env()
    issues = []

    # Critical security variables
    critical_vars = [
        "JWT_SECRET_KEY",
        "ADMIN_PASSWORD",
        "POSTGRES_PASSWORD",
    ]

    for var in critical_vars:
        if not os.getenv(var):
            if is_production():
                issues.append(f"{var} is required in production")
            else:
                logger.warning(f"{var} not set - using insecure default for development")

    if issues:
        raise ValueError(
            f"Environment validation failed:\n" +
            "\n".join(f"  - {issue}" for issue in issues)
        )

    logger.info(f"Environment validated: APP_ENV={env}")
    return {"env": env, "production": is_production()}
