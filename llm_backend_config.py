"""
LLM Backend Configuration System

This module provides a unified interface for managing multiple LLM backends
including OpenAI, Anthropic Claude, and Ollama (local and cloud).
"""

import os
import json
import logging
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
                    id="gpt-4o",
                    name="GPT-4o",
                    description="Most advanced OpenAI model with vision capabilities",
                    max_context_length=128000,
                    cost_per_1k_input=5.0,
                    cost_per_1k_output=15.0
                ),
                ModelInfo(
                    id="gpt-4o-mini",
                    name="GPT-4o Mini",
                    description="Fast and cost-effective model for most tasks",
                    max_context_length=128000,
                    cost_per_1k_input=0.15,
                    cost_per_1k_output=0.6
                ),
                ModelInfo(
                    id="o3-mini",
                    name="O3 Mini",
                    description="Advanced reasoning model (no temperature control)",
                    supports_temperature=False,
                    max_context_length=128000,
                    cost_per_1k_input=1.0,
                    cost_per_1k_output=4.0
                ),
                ModelInfo(
                    id="gpt-4-turbo",
                    name="GPT-4 Turbo",
                    description="High-performance GPT-4 variant",
                    max_context_length=128000,
                    cost_per_1k_input=10.0,
                    cost_per_1k_output=30.0
                ),
                ModelInfo(
                    id="gpt-4",
                    name="GPT-4",
                    description="Original GPT-4 model",
                    max_context_length=8192,
                    cost_per_1k_input=30.0,
                    cost_per_1k_output=60.0
                ),
                ModelInfo(
                    id="gpt-3.5-turbo",
                    name="GPT-3.5 Turbo",
                    description="Fast and economical model",
                    max_context_length=16385,
                    cost_per_1k_input=0.5,
                    cost_per_1k_output=1.5
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
                    id="claude-3-5-sonnet-20241022",
                    name="Claude 3.5 Sonnet",
                    description="Most intelligent Claude model with extended thinking",
                    max_context_length=200000,
                    cost_per_1k_input=3.0,
                    cost_per_1k_output=15.0
                ),
                ModelInfo(
                    id="claude-3-opus-20240229",
                    name="Claude 3 Opus",
                    description="Most capable Claude 3 model for complex tasks",
                    max_context_length=200000,
                    cost_per_1k_input=15.0,
                    cost_per_1k_output=75.0
                ),
                ModelInfo(
                    id="claude-3-sonnet-20240229",
                    name="Claude 3 Sonnet",
                    description="Balanced intelligence and speed",
                    max_context_length=200000,
                    cost_per_1k_input=3.0,
                    cost_per_1k_output=15.0
                ),
                ModelInfo(
                    id="claude-3-haiku-20240307",
                    name="Claude 3 Haiku",
                    description="Fastest and most compact Claude model",
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
                ModelInfo(
                    id="llama3.1:70b",
                    name="Llama 3.1 70B",
                    description="Meta's largest Llama 3.1 model",
                    max_context_length=128000,
                ),
                ModelInfo(
                    id="llama3.1:8b",
                    name="Llama 3.1 8B",
                    description="Efficient Llama 3.1 model",
                    max_context_length=128000,
                ),
                ModelInfo(
                    id="llama3:70b",
                    name="Llama 3 70B",
                    description="Meta's Llama 3 70B model",
                    max_context_length=8192,
                ),
                ModelInfo(
                    id="llama3:8b",
                    name="Llama 3 8B",
                    description="Meta's Llama 3 8B model",
                    max_context_length=8192,
                ),
                ModelInfo(
                    id="mistral:7b",
                    name="Mistral 7B",
                    description="Mistral's efficient 7B model",
                    max_context_length=8192,
                ),
                ModelInfo(
                    id="mixtral:8x7b",
                    name="Mixtral 8x7B",
                    description="Mistral's mixture-of-experts model",
                    max_context_length=32768,
                ),
                ModelInfo(
                    id="qwen2.5:72b",
                    name="Qwen 2.5 72B",
                    description="Alibaba's Qwen 2.5 large model",
                    max_context_length=128000,
                ),
                ModelInfo(
                    id="qwen2.5:14b",
                    name="Qwen 2.5 14B",
                    description="Alibaba's Qwen 2.5 medium model",
                    max_context_length=128000,
                ),
                ModelInfo(
                    id="phi3:14b",
                    name="Phi-3 14B",
                    description="Microsoft's Phi-3 model",
                    max_context_length=128000,
                ),
                ModelInfo(
                    id="gemma2:27b",
                    name="Gemma 2 27B",
                    description="Google's Gemma 2 large model",
                    max_context_length=8192,
                ),
                ModelInfo(
                    id="gemma2:9b",
                    name="Gemma 2 9B",
                    description="Google's Gemma 2 medium model",
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
            default_base_url=None,  # User must specify
            requires_api_key=True,
            models=[
                # Same models as local Ollama, but running on cloud instance
                ModelInfo(
                    id="llama3.1:70b",
                    name="Llama 3.1 70B",
                    description="Meta's largest Llama 3.1 model (Cloud)",
                    max_context_length=128000,
                ),
                ModelInfo(
                    id="llama3.1:8b",
                    name="Llama 3.1 8B",
                    description="Efficient Llama 3.1 model (Cloud)",
                    max_context_length=128000,
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
        """Get default configuration"""
        return {
            "active_backend": "openai",
            "active_models": {
                "default": {
                    "backend": "openai",
                    "model": "gpt-4o-mini"
                },
                "premium": {
                    "backend": "openai",
                    "model": "o3-mini"
                }
            },
            "backend_settings": {
                "openai": {"enabled": True},
                "anthropic": {"enabled": False},
                "ollama": {"enabled": False},
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
                # Modern langchain-anthropic uses 'model' parameter (not 'model_name')
                # But we need to handle other parameters differently
                anthropic_kwargs = {
                    "model": kwargs.get("model"),  # Use 'model' not 'model_name'
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
                # IMPORTANT: Use ChatOllama for chat completion format, not base Ollama class
                from langchain_community.chat_models import ChatOllama

                # ChatOllama parameter mapping
                ollama_kwargs = {
                    "model": kwargs.get("model"),
                    "base_url": kwargs.get("base_url", "http://localhost:11434"),
                }

                # Add temperature if present
                if "temperature" in kwargs:
                    ollama_kwargs["temperature"] = kwargs["temperature"]

                # Filter out None values
                ollama_kwargs = {k: v for k, v in ollama_kwargs.items() if v is not None}

                logger.info(f"Creating ChatOllama with kwargs: {ollama_kwargs}")
                return ChatOllama(**ollama_kwargs)
            except ImportError:
                raise ImportError("Ollama support requires langchain-community. Install with: pip install langchain-community")

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
