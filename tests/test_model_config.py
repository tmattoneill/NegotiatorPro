"""
Tests for ModelConfig class from main.py
"""
import pytest
import sys
from pathlib import Path

# Import ModelConfig from main.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import ModelConfig


class TestModelConfig:
    """Test ModelConfig functionality"""

    def test_gpt4o_mini_config(self):
        """Test gpt-4o-mini configuration"""
        config = ModelConfig.get_model_kwargs("gpt-4o-mini")

        assert config["model"] == "gpt-4o-mini"
        assert "temperature" in config
        assert config["temperature"] == 0.3

    def test_o3_mini_config(self):
        """Test o3-mini configuration (no temperature)"""
        config = ModelConfig.get_model_kwargs("o3-mini")

        assert config["model"] == "o3-mini"
        # o3-mini should NOT have temperature parameter
        assert "temperature" not in config

    def test_gpt4_config(self):
        """Test gpt-4 configuration"""
        config = ModelConfig.get_model_kwargs("gpt-4")

        assert config["model"] == "gpt-4"
        assert "temperature" in config
        assert config["temperature"] == 0.3

    def test_gpt35_turbo_config(self):
        """Test gpt-3.5-turbo configuration"""
        config = ModelConfig.get_model_kwargs("gpt-3.5-turbo")

        assert config["model"] == "gpt-3.5-turbo"
        assert "temperature" in config

    def test_unknown_model_fallback(self):
        """Test unknown model uses fallback configuration"""
        config = ModelConfig.get_model_kwargs("unknown-model")

        assert config["model"] == "unknown-model"
        # Should have default temperature
        assert "temperature" in config

    def test_model_configs_dictionary_exists(self):
        """Test that MODEL_CONFIGS dictionary is defined"""
        assert hasattr(ModelConfig, "MODEL_CONFIGS")
        assert isinstance(ModelConfig.MODEL_CONFIGS, dict)

    def test_all_defined_models_have_model_key(self):
        """Test that all model configs have 'model' key"""
        for model_name, config in ModelConfig.MODEL_CONFIGS.items():
            assert "model" in config
            assert config["model"] == model_name

    def test_config_filters_none_values(self):
        """Test that None values are filtered from config"""
        config = ModelConfig.get_model_kwargs("gpt-4o-mini")

        # Should not contain any None values
        for key, value in config.items():
            assert value is not None

    def test_o3_mini_specific_parameters(self):
        """Test o3-mini has model-specific parameter handling"""
        o3_config = ModelConfig.get_model_kwargs("o3-mini")
        gpt4o_config = ModelConfig.get_model_kwargs("gpt-4o-mini")

        # o3-mini should not have temperature
        assert "temperature" not in o3_config

        # gpt-4o-mini should have temperature
        assert "temperature" in gpt4o_config

    def test_supported_models_list(self):
        """Test that common models are supported"""
        supported_models = list(ModelConfig.MODEL_CONFIGS.keys())

        assert "gpt-4o-mini" in supported_models
        assert "o3-mini" in supported_models
        assert "gpt-4" in supported_models
        assert "gpt-3.5-turbo" in supported_models
