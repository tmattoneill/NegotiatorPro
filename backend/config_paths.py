"""
Configuration Paths Management

Centralizes all configuration file paths to support Docker-based deployment
with data mounted at /data or falling back to local development paths.
"""
import os
from pathlib import Path


class ConfigPaths:
    """
    Manages configuration file paths for Docker and local development.

    In Docker: Uses /app/config (mapped to ./data/config on host)
    In local dev: Uses project root directory
    """

    def __init__(self):
        """Initialize configuration paths."""
        # Detect if running in Docker
        self.in_docker = os.path.exists("/.dockerenv")

        # Base directories
        if self.in_docker:
            self.config_dir = Path("/app/config")
            self.data_dir = Path("/app")
        else:
            # Local development - use project root
            self.config_dir = Path.cwd()
            self.data_dir = Path.cwd()

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

    @property
    def admin_config(self) -> str:
        """Path to admin_config.json"""
        return str(self.config_dir / "admin_config.json")

    @property
    def admin_sessions(self) -> str:
        """Path to admin_sessions.json"""
        return str(self.config_dir / "admin_sessions.json")

    @property
    def usage_stats(self) -> str:
        """Path to usage_stats.json"""
        return str(self.config_dir / "usage_stats.json")

    @property
    def llm_backend_config(self) -> str:
        """Path to llm_backend_config.json"""
        return str(self.config_dir / "llm_backend_config.json")

    @property
    def embedding_config(self) -> str:
        """Path to embedding_config.json"""
        return str(self.config_dir / "embedding_config.json")

    @property
    def prompt_config(self) -> str:
        """Path to prompt_config.json"""
        return str(self.config_dir / "prompt_config.json")

    @property
    def vectorstore_dir(self) -> str:
        """Path to vectorstore directory"""
        if self.in_docker:
            return "/app/vectorstore"
        else:
            return str(self.data_dir / "vectorstore")

    @property
    def uploads_dir(self) -> str:
        """Path to uploads directory"""
        if self.in_docker:
            return "/app/uploads"
        else:
            return str(self.data_dir / "uploads")

    @property
    def sources_dir(self) -> str:
        """Path to sources directory"""
        if self.in_docker:
            return "/app/sources"
        else:
            return str(self.data_dir / "sources")


# Global instance
config_paths = ConfigPaths()
