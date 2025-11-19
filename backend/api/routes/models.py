"""
Models API Routes

Provides endpoints for retrieving available LLM models and backends.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List
import logging

# Import the same RAG system function used by chat endpoint
from .chat import get_rag_system

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
async def get_available_models():
    """
    Get all available LLM models organized by backend.

    Returns:
        Dict with backend IDs as keys and model information as values.
        Each backend includes: name, enabled status, and list of models.
    """
    try:
        rag = get_rag_system()
        backend_manager = rag.backend_manager

        # Get all available backends from the backend manager
        all_backends = backend_manager.get_available_backends()

        # Build response dict
        models_by_backend = {}
        for backend in all_backends:
            # Check if backend is enabled in user config
            is_enabled = backend_manager.user_config.get("backend_settings", {}).get(
                backend.id, {}
            ).get("enabled", False)

            # Convert ModelInfo objects to dicts
            models_list = [
                {
                    "id": model.id,
                    "name": model.name,
                    "description": model.description
                }
                for model in backend.models
            ]

            models_by_backend[backend.id] = {
                "name": backend.name,
                "enabled": is_enabled,
                "models": models_list
            }

        # Only return enabled backends
        enabled_backends = {
            backend_id: backend_info
            for backend_id, backend_info in models_by_backend.items()
            if backend_info["enabled"]
        }

        # If no backends enabled, check for API keys and enable OpenAI as fallback
        if not enabled_backends:
            import os
            has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
            if has_openai_key and "openai" in models_by_backend:
                models_by_backend["openai"]["enabled"] = True
                return {"openai": models_by_backend["openai"]}

        return enabled_backends if enabled_backends else models_by_backend

    except Exception as e:
        logger.error(f"Error in get_available_models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve models: {str(e)}")


@router.get("/models/current")
async def get_current_model_config():
    """
    Get the currently configured default and premium models.

    Returns:
        Dict with current default and premium model configurations.
    """
    try:
        rag = get_rag_system()
        backend_manager = rag.backend_manager

        default_config = backend_manager.get_active_model_config("default")
        premium_config = backend_manager.get_active_model_config("premium")

        return {
            "default": {
                "backend": default_config.get("backend", "openai"),
                "model": default_config.get("model", "gpt-4o-mini")
            },
            "premium": {
                "backend": premium_config.get("backend", "openai"),
                "model": premium_config.get("model", "o3-mini")
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve current config: {str(e)}")
