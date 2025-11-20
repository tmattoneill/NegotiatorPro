"""
Configuration Loader

Loads and manages application configuration from config.json.
Provides easy access to UI settings, LLM models, and system-wide defaults.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Singleton class for loading and accessing application configuration.

    Usage:
        config = ConfigLoader()
        models = config.get_llm_models()
        default_model = config.get_default_model()
    """

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _load_config(self):
        """Load configuration from config.json file."""
        try:
            # Look for config.json in project root
            config_path = Path(__file__).parent.parent / "config.json"

            if not config_path.exists():
                logger.warning(f"Config file not found at {config_path}, using defaults")
                self._config = self._get_default_config()
                return

            with open(config_path, 'r') as f:
                self._config = json.load(f)

            logger.info(f"Configuration loaded from {config_path}")

        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return minimal default configuration if config.json is missing."""
        return {
            "app": {
                "name": "NegotiatorPro",
                "version": "1.0.0"
            },
            "llm_models": [],
            "defaults": {
                "default_provider": "openai",
                "default_model": "gpt-4o-mini"
            }
        }

    def reload(self):
        """Reload configuration from disk."""
        self._config = None
        self._load_config()

    # ========== General Config Access ==========

    def get(self, key: str, default: Any = None) -> Any:
        """Get any config value by key path (e.g., 'app.name')."""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default

    def get_app_info(self) -> Dict[str, str]:
        """Get application information."""
        return self._config.get("app", {})

    # ========== LLM Model Access ==========

    def get_llm_models(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get LLM model configurations.

        Args:
            provider: Optional provider filter (e.g., 'openai', 'anthropic')

        Returns:
            List of model configurations
        """
        models = self._config.get("llm_models", [])

        if provider:
            # Filter by provider
            for provider_config in models:
                if provider_config.get("provider") == provider:
                    return provider_config.get("models", [])
            return []

        return models

    def get_provider_config(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific provider."""
        for provider_config in self._config.get("llm_models", []):
            if provider_config.get("provider") == provider:
                return provider_config
        return None

    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled provider IDs."""
        return [
            p.get("provider")
            for p in self._config.get("llm_models", [])
            if p.get("enabled", True)
        ]

    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Find a model by its ID across all providers."""
        for provider_config in self._config.get("llm_models", []):
            for model in provider_config.get("models", []):
                if model.get("id") == model_id:
                    # Include provider info in response
                    return {
                        **model,
                        "provider": provider_config.get("provider"),
                        "provider_name": provider_config.get("provider_prettyname")
                    }
        return None

    def get_models_by_tier(self, tier: str) -> List[Dict[str, Any]]:
        """
        Get all models of a specific tier (e.g., 'default', 'premium').

        Args:
            tier: Model tier ('default' or 'premium')

        Returns:
            List of models matching the tier
        """
        matching_models = []

        for provider_config in self._config.get("llm_models", []):
            if not provider_config.get("enabled", True):
                continue

            for model in provider_config.get("models", []):
                if model.get("tier") == tier:
                    matching_models.append({
                        **model,
                        "provider": provider_config.get("provider"),
                        "provider_name": provider_config.get("provider_prettyname")
                    })

        return matching_models

    # ========== Defaults ==========

    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return self._config.get("defaults", {})

    def get_default_model(self) -> Dict[str, str]:
        """Get default model configuration."""
        defaults = self.get_defaults()
        return {
            "provider": defaults.get("default_provider", "openai"),
            "model": defaults.get("default_model", "gpt-4o-mini")
        }

    def get_premium_model(self) -> Dict[str, str]:
        """Get premium model configuration."""
        defaults = self.get_defaults()
        return {
            "provider": defaults.get("premium_provider", "openai"),
            "model": defaults.get("premium_model", "o3-mini")
        }

    # ========== UI Settings ==========

    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration."""
        return self._config.get("ui", {})

    def get_ui_features(self) -> Dict[str, bool]:
        """Get enabled UI features."""
        return self._config.get("ui", {}).get("features", {})

    def get_ui_limits(self) -> Dict[str, int]:
        """Get UI limits (max message length, etc.)."""
        return self._config.get("ui", {}).get("limits", {})

    # ========== Negotiation Framework ==========

    def get_negotiation_config(self) -> Dict[str, Any]:
        """Get negotiation framework configuration."""
        return self._config.get("negotiation", {})

    def get_analysis_elements(self) -> List[str]:
        """Get list of negotiation analysis elements."""
        return self._config.get("negotiation", {}).get("analysis_elements", [])

    def get_self_assessment_criteria(self) -> List[str]:
        """Get self-assessment criteria for negotiations."""
        return self._config.get("negotiation", {}).get("self_assessment_criteria", [])

    # ========== Security Settings ==========

    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return self._config.get("security", {})

    def get_password_requirements(self) -> Dict[str, Any]:
        """Get password security requirements."""
        security = self.get_security_config()
        return {
            "min_length": security.get("password_min_length", 8),
            "require_uppercase": security.get("require_uppercase", False),
            "require_lowercase": security.get("require_lowercase", False),
            "require_numbers": security.get("require_numbers", False),
            "require_special_chars": security.get("require_special_chars", False)
        }

    # ========== Admin Settings ==========

    def get_admin_config(self) -> Dict[str, Any]:
        """Get admin configuration."""
        return self._config.get("admin", {})

    def get_default_admin_credentials(self) -> Dict[str, str]:
        """Get default admin user credentials."""
        admin = self.get_admin_config()
        return {
            "username": admin.get("default_username", "admin"),
            "email": admin.get("default_email", "test@example.com")
        }


# Global singleton instance
config = ConfigLoader()


# Convenience functions for common operations
def get_llm_models(provider: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get LLM models (convenience function)."""
    return config.get_llm_models(provider)


def get_default_model() -> Dict[str, str]:
    """Get default model (convenience function)."""
    return config.get_default_model()


def get_premium_model() -> Dict[str, str]:
    """Get premium model (convenience function)."""
    return config.get_premium_model()


def get_enabled_providers() -> List[str]:
    """Get enabled providers (convenience function)."""
    return config.get_enabled_providers()
