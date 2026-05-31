"""
Models API Routes

Provides endpoints for retrieving available LLM models and backends.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional
import logging
import os
import requests

# Import the same RAG system function used by chat endpoint
from .chat import get_rag_system
from ...user_profile import UserProfileManager

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

            # For Ollama backends, dynamically fetch available models
            if backend.provider == "ollama":
                import os
                if backend.id == "ollama":
                    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                else:  # ollama-cloud
                    base_url = os.getenv("OLLAMA_CLOUD_URL", "https://ollama.com")

                dynamic_models = backend_manager.get_ollama_available_models(base_url)
                if dynamic_models:
                    models_list = [
                        {
                            "id": model.id,
                            "name": model.name,
                            "description": model.description
                        }
                        for model in dynamic_models
                    ]
                    models_by_backend[backend.id] = {
                        "name": backend.name,
                        "enabled": is_enabled,
                        "models": models_list,
                        "available": True
                    }
                else:
                    # Ollama not reachable - report error to user
                    models_by_backend[backend.id] = {
                        "name": backend.name,
                        "enabled": is_enabled,
                        "models": [],
                        "available": False,
                        "error": f"Could not connect to Ollama at {base_url}. Please ensure Ollama is running."
                    }
                continue
            else:
                # Convert ModelInfo objects to dicts for non-Ollama backends
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


@router.get("/models/available-for-user")
async def get_available_providers_for_user(user_id: Optional[str] = Query(None)):
    """
    Get available providers filtered by user's API keys.

    Returns only providers for which:
    1. User has a valid API key configured, OR
    2. System has a valid API key in environment, OR
    3. Provider is Ollama (local) and is running, OR
    4. Provider is RunPod (always shown as fallback)

    Args:
        user_id: Optional user ID to check user-specific API keys

    Returns:
        Dict with available providers and their models, filtered by API key availability.
    """
    try:
        rag = get_rag_system()
        backend_manager = rag.backend_manager

        # Get user's API keys if user_id provided
        user_has_openai_key = False
        user_has_anthropic_key = False

        if user_id:
            try:
                user_keys = await UserProfileManager.get_user_api_keys(user_id)
                user_has_openai_key = bool(user_keys.get("openai_api_key"))
                user_has_anthropic_key = bool(user_keys.get("anthropic_api_key"))
            except Exception as e:
                logger.warning(f"Could not fetch user API keys: {e}")

        # Check system API keys
        system_has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
        system_has_anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))
        system_has_runpod_key = bool(os.getenv("RUNPOD_API_KEY"))

        # Check if Ollama is running locally
        ollama_available = False
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            response = requests.get(f"{ollama_base_url}/api/tags", timeout=2)
            ollama_available = response.status_code == 200
        except:
            ollama_available = False

        # Build response with filtered providers
        available_providers = {}
        all_backends = backend_manager.get_available_backends()

        for backend in all_backends:
            provider_available = False
            provider_models = []
            provider_error = None

            if backend.id == "openai":
                # OpenAI available if user or system has key
                provider_available = user_has_openai_key or system_has_openai_key
                if provider_available:
                    provider_models = [
                        {"id": model.id, "name": model.name, "description": model.description}
                        for model in backend.models
                    ]

            elif backend.id == "anthropic":
                # Anthropic available if user or system has key
                provider_available = user_has_anthropic_key or system_has_anthropic_key
                if provider_available:
                    provider_models = [
                        {"id": model.id, "name": model.name, "description": model.description}
                        for model in backend.models
                    ]

            elif backend.id == "ollama":
                # Ollama (local) available if running
                provider_available = ollama_available
                if provider_available:
                    dynamic_models = backend_manager.get_ollama_available_models(ollama_base_url)
                    provider_models = [
                        {"id": model.id, "name": model.name, "description": model.description}
                        for model in dynamic_models
                    ]
                else:
                    provider_error = "Ollama is not running locally. Start Ollama to use local models."

            elif backend.id == "ollama-cloud":
                # Ollama cloud available if API key set
                ollama_cloud_key = bool(os.getenv("OLLAMA_API_KEY"))
                provider_available = ollama_cloud_key
                if provider_available:
                    provider_models = [
                        {"id": model.id, "name": model.name, "description": model.description}
                        for model in backend.models
                    ]

            elif backend.id == "deepseek":
                system_has_deepseek_key = bool(os.getenv("DEEPSEEK_API_KEY"))
                provider_available = system_has_deepseek_key
                if provider_available:
                    provider_models = [
                        {"id": model.id, "name": model.name, "description": model.description}
                        for model in backend.models
                    ]

            elif backend.id == "runpod":
                # RunPod always available as fallback (if API key exists)
                provider_available = system_has_runpod_key
                if provider_available:
                    provider_models = [
                        {"id": model.id, "name": model.name, "description": model.description}
                        for model in backend.models
                    ]

            # Always include RunPod as fallback option (marked as fallback)
            if backend.id == "runpod":
                available_providers[backend.id] = {
                    "name": backend.name,
                    "available": provider_available,
                    "models": provider_models if provider_available else [
                        {"id": "basic", "name": "Basic Model", "description": "Basic fallback model via RunPod"}
                    ],
                    "is_fallback": True,
                    "requires_key": not provider_available
                }
            elif provider_available:
                available_providers[backend.id] = {
                    "name": backend.name,
                    "available": True,
                    "models": provider_models,
                    "is_fallback": False
                }
            elif backend.id == "ollama" and not ollama_available:
                # Include Ollama with error message so users know why it's not available
                available_providers[backend.id] = {
                    "name": backend.name,
                    "available": False,
                    "models": [],
                    "is_fallback": False,
                    "error": provider_error
                }

        # Add metadata about what keys the user has
        response = {
            "providers": available_providers,
            "user_api_keys": {
                "has_openai": user_has_openai_key,
                "has_anthropic": user_has_anthropic_key
            },
            "system_api_keys": {
                "has_openai": system_has_openai_key,
                "has_anthropic": system_has_anthropic_key,
                "has_runpod": system_has_runpod_key
            },
            "ollama_local_available": ollama_available
        }

        # Debug logging
        logger.info(f"[available-for-user] user_id={user_id}")
        logger.info(f"[available-for-user] user_keys: openai={user_has_openai_key}, anthropic={user_has_anthropic_key}")
        logger.info(f"[available-for-user] system_keys: openai={system_has_openai_key}, anthropic={system_has_anthropic_key}, runpod={system_has_runpod_key}")
        logger.info(f"[available-for-user] ollama_available={ollama_available}")
        logger.info(f"[available-for-user] providers returned: {list(available_providers.keys())}")
        for pid, pinfo in available_providers.items():
            logger.info(f"[available-for-user]   {pid}: available={pinfo.get('available')}, is_fallback={pinfo.get('is_fallback')}, models={len(pinfo.get('models', []))}")

        return response

    except Exception as e:
        logger.error(f"Error in get_available_providers_for_user: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve providers: {str(e)}")
