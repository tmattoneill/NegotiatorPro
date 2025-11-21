"""
LLM Backend Configuration System

This module provides a unified interface for managing multiple LLM backends
including OpenAI, Anthropic Claude, and Ollama (local and cloud).
"""

import os
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a specific model"""
    id: str
    name: str
    description: str
    supports_temperature: bool = True
    supports_max_tokens: bool = True
    supports_streaming: bool = True
    default_temperature: float = 0.3
    max_context_length: int = 4096
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


@dataclass
class BackendConfig:
    """Configuration for an LLM backend"""
    id: str
    name: str
    provider: str  # 'openai', 'anthropic', 'ollama'
    enabled: bool = True
    api_key_env_var: Optional[str] = None
    base_url_env_var: Optional[str] = None
    default_base_url: Optional[str] = None
    models: List[ModelInfo] = field(default_factory=list)
    requires_api_key: bool = True


class LLMBackendManager:
    """Manages multiple LLM backends and their configurations"""

    # Define all supported backends and their models
    BACKENDS = {
        "openai": BackendConfig(
            id="openai",
            name="OpenAI",
            provider="openai",
            api_key_env_var="OPENAI_API_KEY",
            base_url_env_var="OPENAI_BASE_URL",
            default_base_url="https://api.openai.com/v1",
            requires_api_key=True,
            models=[
                ModelInfo(
                    id="gpt-4.1-2025-04-14",
                    name="GPT-4.1 (Default)",
                    description="Latest GPT-4.1 model with enhanced capabilities",
                    max_context_length=128000,
                    cost_per_1k_input=2.0,
                    cost_per_1k_output=8.0
                ),
                ModelInfo(
                    id="gpt-4o-mini",
                    name="GPT-4o Mini",
                    description="Fast and cost-effective model for most tasks",
                    max_context_length=128000,
                    cost_per_1k_input=0.15,
                    cost_per_1k_output=0.60
                ),
                ModelInfo(
                    id="gpt-5.1-2025-11-13",
                    name="GPT-5.1",
                    description="Latest GPT-5.1 model",
                    max_context_length=128000,
                    cost_per_1k_input=3.0,
                    cost_per_1k_output=12.0
                ),
            ]
        ),
        "anthropic": BackendConfig(
            id="anthropic",
            name="Anthropic Claude",
            provider="anthropic",
            api_key_env_var="ANTHROPIC_API_KEY",
            base_url_env_var="ANTHROPIC_BASE_URL",
            default_base_url="https://api.anthropic.com",
            requires_api_key=True,
            models=[
                ModelInfo(
                    id="claude-sonnet-4-5-20250929",
                    name="Claude Sonnet 4.5",
                    description="Latest Claude Sonnet with enhanced capabilities",
                    max_context_length=200000,
                    cost_per_1k_input=3.0,
                    cost_per_1k_output=15.0
                ),
                ModelInfo(
                    id="claude-haiku-4-5-20251001",
                    name="Claude Haiku 4.5",
                    description="Latest Claude Haiku - fast and efficient",
                    max_context_length=200000,
                    cost_per_1k_input=0.25,
                    cost_per_1k_output=1.25
                ),
            ]
        ),
        "ollama": BackendConfig(
            id="ollama",
            name="Ollama (Local)",
            provider="ollama",
            api_key_env_var=None,
            base_url_env_var="OLLAMA_BASE_URL",
            default_base_url="http://localhost:11434",
            requires_api_key=False,
            models=[
                # Only include locally installed models
                ModelInfo(
                    id="granite3.1-moe:latest",
                    name="Granite 3.1 MoE",
                    description="IBM's Granite 3.1 mixture-of-experts model",
                    max_context_length=8192,
                ),
                ModelInfo(
                    id="granite3.1-moe:1b",
                    name="Granite 3.1 MoE 1B",
                    description="IBM's compact Granite 3.1 MoE model",
                    max_context_length=8192,
                ),
            ]
        ),
        "ollama-cloud": BackendConfig(
            id="ollama-cloud",
            name="Ollama (Cloud)",
            provider="ollama",
            api_key_env_var="OLLAMA_API_KEY",
            base_url_env_var="OLLAMA_CLOUD_URL",
            default_base_url="https://ollama.com",  # Ollama cloud endpoint
            requires_api_key=True,
            models=[
                # Cloud-hosted Ollama models (from ollama list output)
                ModelInfo(
                    id="glm-4.6:cloud",
                    name="GLM-4.6 Cloud",
                    description="Zhipu AI's GLM-4.6 language model (cloud-hosted)",
                    max_context_length=128000,
                ),
                ModelInfo(
                    id="gpt-oss:20b-cloud",
                    name="GPT-OSS 20B Cloud",
                    description="Medium GPT-OSS model (cloud-hosted)",
                    max_context_length=32768,
                ),
                ModelInfo(
                    id="gpt-oss:120b-cloud",
                    name="GPT-OSS 120B Cloud",
                    description="Large GPT-OSS model (cloud-hosted)",
                    max_context_length=32768,
                ),
                ModelInfo(
                    id="qwen3-coder:480b-cloud",
                    name="Qwen3 Coder 480B Cloud",
                    description="Alibaba's massive Qwen3 coding model (cloud-hosted)",
                    max_context_length=32768,
                ),
            ]
        ),
    }

    def __init__(self, config_file: str = "llm_backend_config.json"):
        """Initialize the backend manager"""
        self.config_file = config_file
        self.user_config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load user configuration from file"""
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded LLM backend configuration from {self.config_file}")
                return config
            except Exception as e:
                logger.error(f"Error loading backend config: {e}")
                return self._get_default_config()
        else:
            logger.info("No existing backend config found, creating default")
            config = self._get_default_config()
            self.save_config(config)
            return config

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration - defaults to Ollama (local) unless user has API keys"""
        return {
            "active_backend": "ollama",
            "active_models": {
                "default": {
                    "backend": "ollama",
                    "model": "llama3.1:8b"
                },
                "premium": {
                    "backend": "ollama",
                    "model": "llama3.1:70b"
                }
            },
            "backend_settings": {
                "openai": {"enabled": False},
                "anthropic": {"enabled": False},
                "ollama": {"enabled": True},
                "ollama-cloud": {"enabled": False}
            }
        }

    def save_config(self, config: Optional[Dict[str, Any]] = None):
        """Save configuration to file"""
        if config is None:
            config = self.user_config

        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved LLM backend configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving backend config: {e}")

    def get_available_backends(self) -> List[BackendConfig]:
        """Get list of available backends"""
        return list(self.BACKENDS.values())

    def get_enabled_backends(self) -> List[BackendConfig]:
        """Get list of enabled backends"""
        enabled = []
        for backend_id, backend in self.BACKENDS.items():
            if self.user_config.get("backend_settings", {}).get(backend_id, {}).get("enabled", False):
                enabled.append(backend)
        return enabled

    def get_backend(self, backend_id: str) -> Optional[BackendConfig]:
        """Get backend configuration by ID"""
        return self.BACKENDS.get(backend_id)

    def get_model_info(self, backend_id: str, model_id: str) -> Optional[ModelInfo]:
        """Get model information"""
        backend = self.get_backend(backend_id)
        if backend:
            for model in backend.models:
                if model.id == model_id:
                    return model
        return None

    def get_ollama_available_models(self, base_url: str = None) -> List[ModelInfo]:
        """Query Ollama for available models"""
        if base_url is None:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        try:
            # Query Ollama API for available models
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                data = response.json()
                models = []
                for model_data in data.get("models", []):
                    model_name = model_data.get("name", "")
                    # Extract size info if available
                    size_bytes = model_data.get("size", 0)
                    size_gb = size_bytes / (1024**3) if size_bytes > 0 else 0

                    # Create ModelInfo for each available model
                    models.append(ModelInfo(
                        id=model_name,
                        name=model_name,
                        description=f"Ollama model ({size_gb:.1f}GB)" if size_gb > 0 else "Ollama model",
                        max_context_length=128000,  # Default, may vary by model
                    ))
                logger.info(f"Found {len(models)} Ollama models at {base_url}")
                return models
            else:
                logger.warning(f"Ollama API returned status {response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to Ollama at {base_url}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error querying Ollama models: {e}")
            return []

    def get_active_model_config(self, model_type: str = "default") -> Dict[str, str]:
        """Get active model configuration (backend + model)"""
        active_models = self.user_config.get("active_models", {})
        if model_type in active_models:
            return active_models[model_type]

        # Fallback to default
        return active_models.get("default", {
            "backend": "openai",
            "model": "gpt-4o-mini"
        })

    def set_active_model(self, model_type: str, backend_id: str, model_id: str):
        """Set active model for a model type (default/premium)"""
        if "active_models" not in self.user_config:
            self.user_config["active_models"] = {}

        self.user_config["active_models"][model_type] = {
            "backend": backend_id,
            "model": model_id
        }
        self.save_config()
        logger.info(f"Set {model_type} model to {backend_id}/{model_id}")

    def enable_backend(self, backend_id: str, enabled: bool = True):
        """Enable or disable a backend"""
        if "backend_settings" not in self.user_config:
            self.user_config["backend_settings"] = {}

        if backend_id not in self.user_config["backend_settings"]:
            self.user_config["backend_settings"][backend_id] = {}

        self.user_config["backend_settings"][backend_id]["enabled"] = enabled
        self.save_config()
        logger.info(f"{'Enabled' if enabled else 'Disabled'} backend: {backend_id}")

    def get_llm_kwargs(self, backend_id: str, model_id: str) -> Dict[str, Any]:
        """Get kwargs for creating an LLM instance"""
        backend = self.get_backend(backend_id)
        model_info = self.get_model_info(backend_id, model_id)

        if not backend or not model_info:
            logger.error(f"Backend {backend_id} or model {model_id} not found")
            return {}

        kwargs = {"model": model_id}

        # Add temperature if supported
        if model_info.supports_temperature:
            kwargs["temperature"] = model_info.default_temperature

        # Add max_tokens if supported
        if model_info.supports_max_tokens:
            kwargs["max_tokens"] = None  # Let model decide

        # Add streaming if supported
        if model_info.supports_streaming:
            kwargs["streaming"] = False  # Default to non-streaming

        # Add API key if required
        if backend.requires_api_key and backend.api_key_env_var:
            api_key = os.getenv(backend.api_key_env_var)
            if api_key:
                kwargs["api_key"] = api_key
            else:
                logger.warning(f"API key not found for {backend_id}: {backend.api_key_env_var}")

        # Add base URL for Ollama or custom endpoints
        if backend.base_url_env_var:
            base_url = os.getenv(backend.base_url_env_var, backend.default_base_url)
            if base_url:
                kwargs["base_url"] = base_url

        # Filter out None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        return kwargs

    def create_llm_instance(self, backend_id: str, model_id: str):
        """Create an LLM instance based on backend and model"""
        backend = self.get_backend(backend_id)

        if not backend:
            raise ValueError(f"Unknown backend: {backend_id}")

        kwargs = self.get_llm_kwargs(backend_id, model_id)

        # Create appropriate LLM instance based on provider
        if backend.provider == "openai":
            from langchain_openai import ChatOpenAI
            logger.info(f"Creating ChatOpenAI with kwargs: {kwargs}")
            return ChatOpenAI(**kwargs)

        elif backend.provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
                # ChatAnthropic uses 'model_name' parameter (not 'model')
                # But we need to handle other parameters differently
                anthropic_kwargs = {
                    "model_name": kwargs.get("model"),  # ChatAnthropic expects model_name
                    "temperature": kwargs.get("temperature", 0.3),
                }

                # Add API key if present
                if "api_key" in kwargs:
                    anthropic_kwargs["anthropic_api_key"] = kwargs["api_key"]

                # Add base URL if present
                if "base_url" in kwargs:
                    anthropic_kwargs["base_url"] = kwargs["base_url"]

                # Remove streaming for Anthropic to avoid issues
                # Filter out None values
                anthropic_kwargs = {k: v for k, v in anthropic_kwargs.items() if v is not None}

                logger.info(f"Creating ChatAnthropic with kwargs: {anthropic_kwargs}")
                return ChatAnthropic(**anthropic_kwargs)
            except ImportError:
                raise ImportError("langchain-anthropic not installed. Install with: pip install langchain-anthropic")

        elif backend.provider == "ollama":
            try:
                # Use the newer langchain-ollama package (langchain-community version is deprecated)
                from langchain_ollama import ChatOllama

                # ChatOllama parameter mapping
                ollama_kwargs = {
                    "model": kwargs.get("model"),
                    "base_url": kwargs.get("base_url", "http://localhost:11434"),
                }

                # Add temperature if present
                if "temperature" in kwargs:
                    ollama_kwargs["temperature"] = kwargs["temperature"]

                # Add authentication headers for cloud instances
                if backend_id == "ollama-cloud" and "api_key" in kwargs:
                    ollama_kwargs["headers"] = {
                        "Authorization": f"Bearer {kwargs['api_key']}"
                    }
                    logger.info("Added Bearer token authentication for Ollama cloud")

                # Filter out None values
                ollama_kwargs = {k: v for k, v in ollama_kwargs.items() if v is not None}

                logger.info(f"Creating ChatOllama with kwargs: {ollama_kwargs}")
                return ChatOllama(**ollama_kwargs)
            except ImportError:
                raise ImportError("Ollama support requires langchain-ollama. Install with: pip install langchain-ollama")

        else:
            raise ValueError(f"Unknown provider: {backend.provider}")

    def get_status_report(self) -> str:
        """Get a status report of all backends"""
        report = ["🔌 LLM Backend Status\n"]

        # Active models
        report.append("📍 Active Models:")
        for model_type in ["default", "premium"]:
            config = self.get_active_model_config(model_type)
            backend_id = config.get("backend", "unknown")
            model_id = config.get("model", "unknown")
            backend = self.get_backend(backend_id)
            model_info = self.get_model_info(backend_id, model_id)

            backend_name = backend.name if backend else backend_id
            model_name = model_info.name if model_info else model_id

            report.append(f"  • {model_type.capitalize()}: {backend_name} - {model_name}")

        report.append("\n🌐 Backend Status:")

        for backend_id, backend in self.BACKENDS.items():
            enabled = self.user_config.get("backend_settings", {}).get(backend_id, {}).get("enabled", False)
            status = "✅ Enabled" if enabled else "❌ Disabled"

            # Check API key status
            api_status = ""
            if backend.requires_api_key and backend.api_key_env_var:
                api_key = os.getenv(backend.api_key_env_var)
                api_status = " | 🔑 API Key: " + ("✅" if api_key else "❌ Missing")

            report.append(f"  • {backend.name}: {status}{api_status}")
            report.append(f"    Models: {len(backend.models)} available")

        return "\n".join(report)

    def get_all_models_for_ui(self) -> Dict[str, List[tuple]]:
        """Get all models organized by backend for UI display"""
        models_by_backend = {}

        for backend_id, backend in self.BACKENDS.items():
            models = []

            # For Ollama backends, get actual available models
            if backend.provider == "ollama":
                base_url = None
                if backend_id == "ollama-cloud":
                    base_url = os.getenv("OLLAMA_CLOUD_URL", "https://ollama.com")
                else:
                    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

                available_models = self.get_ollama_available_models(base_url)
                for model in available_models:
                    display_name = f"{model.name} - {model.description}"
                    models.append((model.id, display_name))
            else:
                # For OpenAI and Anthropic, use predefined models
                for model in backend.models:
                    display_name = f"{model.name} - {model.description}"
                    models.append((model.id, display_name))

            models_by_backend[backend_id] = {
                "name": backend.name,
                "models": models,
                "enabled": self.user_config.get("backend_settings", {}).get(backend_id, {}).get("enabled", False)
            }

        return models_by_backend


# Global instance
backend_manager = LLMBackendManager()
