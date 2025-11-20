"""
Configuration API Routes

Provides endpoints for retrieving application configuration.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
import logging

from ...config_loader import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/config", tags=["configuration"])


@router.get("/")
async def get_full_config():
    """
    Get full application configuration.

    Returns:
        Complete config.json contents
    """
    return config._config


@router.get("/app")
async def get_app_info():
    """
    Get application information.

    Returns:
        App name, version, description
    """
    return config.get_app_info()


@router.get("/ui")
async def get_ui_config():
    """
    Get UI configuration including theme, features, and limits.

    Returns:
        UI configuration object
    """
    return config.get_ui_config()


@router.get("/llm-models")
async def get_llm_models(provider: Optional[str] = None, tier: Optional[str] = None):
    """
    Get LLM model configurations.

    Args:
        provider: Optional filter by provider (openai, anthropic, ollama)
        tier: Optional filter by tier (default, premium)

    Returns:
        List of model configurations or provider configurations
    """
    try:
        if tier:
            # Get models by tier across all providers
            models = config.get_models_by_tier(tier)
            return {"models": models, "count": len(models)}

        if provider:
            # Get models for specific provider
            provider_config = config.get_provider_config(provider)
            if not provider_config:
                raise HTTPException(status_code=404, detail=f"Provider '{provider}' not found")
            return provider_config

        # Get all provider configurations
        return {"providers": config.get_llm_models()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving LLM models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm-models/{model_id}")
async def get_model_by_id(model_id: str):
    """
    Get detailed information about a specific model.

    Args:
        model_id: Model identifier (e.g., 'gpt-4o-mini')

    Returns:
        Model configuration including provider info
    """
    model = config.get_model_by_id(model_id)

    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return model


@router.get("/providers")
async def get_providers(enabled_only: bool = True):
    """
    Get list of available providers.

    Args:
        enabled_only: Only return enabled providers

    Returns:
        List of provider configurations
    """
    providers = config.get_llm_models()

    if enabled_only:
        providers = [p for p in providers if p.get("enabled", True)]

    return {
        "providers": providers,
        "count": len(providers)
    }


@router.get("/defaults")
async def get_defaults():
    """
    Get default configuration values.

    Returns:
        Default settings including models, temperature, etc.
    """
    return config.get_defaults()


@router.get("/defaults/models")
async def get_default_models():
    """
    Get default and premium model configurations.

    Returns:
        Default and premium model settings
    """
    return {
        "default": config.get_default_model(),
        "premium": config.get_premium_model()
    }


@router.get("/negotiation")
async def get_negotiation_config():
    """
    Get negotiation framework configuration.

    Returns:
        PLEASE framework settings and analysis elements
    """
    return config.get_negotiation_config()


@router.get("/security")
async def get_security_config():
    """
    Get public security configuration.

    Returns:
        Password requirements and session settings (sensitive values excluded)
    """
    security = config.get_security_config()

    # Only return public security settings
    return {
        "password_min_length": security.get("password_min_length", 8),
        "session_timeout_minutes": security.get("session_timeout_minutes", 60),
        "max_login_attempts": security.get("max_login_attempts", 5),
        "require_email_verification": security.get("require_email_verification", False)
    }


@router.post("/reload")
async def reload_config():
    """
    Reload configuration from disk.

    Returns:
        Success message with new config version
    """
    try:
        config.reload()
        return {
            "message": "Configuration reloaded successfully",
            "app_version": config.get("app.version")
        }
    except Exception as e:
        logger.error(f"Error reloading config: {e}")
        raise HTTPException(status_code=500, detail="Failed to reload configuration")
