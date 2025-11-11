"""
Combined tests for text_preprocessor, prompt_manager, and embedding_config modules
"""
import pytest
import json
from pathlib import Path
from text_preprocessor import TextPreprocessor
from prompt_manager import PromptManager
from embedding_config import EmbeddingConfig


class TestTextPreprocessor:
    """Test TextPreprocessor functionality"""

    def test_init(self):
        """Test TextPreprocessor initialization"""
        preprocessor = TextPreprocessor()
        assert preprocessor is not None

    def test_preprocess_basic_text(self):
        """Test basic text preprocessing"""
        preprocessor = TextPreprocessor()
        text = "Hello, this is a test message."

        result = preprocessor.preprocess(text)

        assert "processed_text" in result
        assert "original_tokens" in result
        assert "processed_tokens" in result
        assert isinstance(result["processed_text"], str)

    def test_preprocess_removes_extra_whitespace(self):
        """Test that extra whitespace is removed"""
        preprocessor = TextPreprocessor()
        text = "Hello    world    with   extra   spaces"

        result = preprocessor.preprocess(text)
        processed = result["processed_text"]

        # Should have normalized whitespace
        assert "    " not in processed or len(processed) < len(text)

    def test_token_counting(self):
        """Test token counting is reasonable"""
        preprocessor = TextPreprocessor()
        text = "This is a test sentence with some words."

        result = preprocessor.preprocess(text)

        assert result["original_tokens"] > 0
        assert result["processed_tokens"] > 0

    def test_cost_calculation(self):
        """Test cost calculation"""
        preprocessor = TextPreprocessor()
        text = "This is a test message for cost calculation."

        result = preprocessor.preprocess(text)

        if "estimated_cost_savings" in result:
            assert result["estimated_cost_savings"] >= 0

    def test_empty_text_handling(self):
        """Test handling of empty text"""
        preprocessor = TextPreprocessor()
        text = ""

        result = preprocessor.preprocess(text)

        assert "processed_text" in result
        assert result["processed_text"] == ""

    def test_long_text_processing(self):
        """Test processing of long text"""
        preprocessor = TextPreprocessor()
        text = "This is a test sentence. " * 100

        result = preprocessor.preprocess(text)

        assert result["original_tokens"] > 0
        assert "processed_text" in result


class TestPromptManager:
    """Test PromptManager functionality"""

    def test_init(self, temp_dir, monkeypatch):
        """Test PromptManager initialization"""
        monkeypatch.chdir(temp_dir)
        manager = PromptManager()
        assert manager is not None

    def test_get_raw_prompts(self, temp_dir, monkeypatch):
        """Test getting raw prompts"""
        monkeypatch.chdir(temp_dir)
        manager = PromptManager()

        prompts = manager.get_raw_prompts()

        assert isinstance(prompts, dict)
        assert "system" in prompts or "user" in prompts

    def test_update_system_prompt(self, temp_dir, monkeypatch):
        """Test updating system prompt"""
        monkeypatch.chdir(temp_dir)
        manager = PromptManager()

        test_prompt = "You are a helpful negotiation assistant."
        manager.update_system_prompt(test_prompt)

        # Verify prompt was saved
        prompts = manager.get_raw_prompts()
        assert prompts["system"] == test_prompt

    def test_update_user_prompt(self, temp_dir, monkeypatch):
        """Test updating user prompt"""
        monkeypatch.chdir(temp_dir)
        manager = PromptManager()

        test_prompt = "Question: {question}"
        manager.update_user_prompt(test_prompt)

        # Verify prompt was saved
        prompts = manager.get_raw_prompts()
        assert prompts["user"] == test_prompt

    def test_get_prompts_for_chat(self, temp_dir, monkeypatch):
        """Test getting formatted prompts for chat"""
        monkeypatch.chdir(temp_dir)
        manager = PromptManager()

        question = "How do I negotiate a salary?"
        context = "Some relevant context about negotiations."

        system_prompt, user_prompt = manager.get_prompts_for_chat(question, context)

        assert isinstance(system_prompt, str)
        assert isinstance(user_prompt, str)
        assert len(system_prompt) > 0
        assert len(user_prompt) > 0

    def test_prompt_persistence(self, temp_dir, monkeypatch):
        """Test that prompts persist across instances"""
        monkeypatch.chdir(temp_dir)

        # Create first instance and set prompt
        manager1 = PromptManager()
        test_prompt = "Test system prompt"
        manager1.update_system_prompt(test_prompt)

        # Create second instance and verify prompt
        manager2 = PromptManager()
        prompts = manager2.get_raw_prompts()
        assert prompts["system"] == test_prompt

    def test_placeholder_replacement(self, temp_dir, monkeypatch):
        """Test that placeholders are replaced in prompts"""
        monkeypatch.chdir(temp_dir)
        manager = PromptManager()

        # Set prompts with placeholders
        manager.update_system_prompt("Context: {context}")
        manager.update_user_prompt("Question: {question}")

        question = "Test question"
        context = "Test context"

        system_prompt, user_prompt = manager.get_prompts_for_chat(question, context)

        assert "{context}" not in system_prompt
        assert "{question}" not in user_prompt
        assert "Test context" in system_prompt
        assert "Test question" in user_prompt


class TestEmbeddingConfig:
    """Test EmbeddingConfig functionality"""

    def test_init(self, temp_dir, monkeypatch):
        """Test EmbeddingConfig initialization"""
        monkeypatch.chdir(temp_dir)
        config = EmbeddingConfig()
        assert config is not None

    def test_get_embedding_kwargs(self, temp_dir, monkeypatch):
        """Test getting embedding kwargs"""
        monkeypatch.chdir(temp_dir)
        config = EmbeddingConfig()

        kwargs = config.get_embedding_kwargs()

        assert isinstance(kwargs, dict)
        assert "model" in kwargs

    def test_default_embedding_model(self, temp_dir, monkeypatch):
        """Test default embedding model is set"""
        monkeypatch.chdir(temp_dir)
        config = EmbeddingConfig()

        kwargs = config.get_embedding_kwargs()

        # Default should be text-embedding-3-large
        assert kwargs["model"] == "text-embedding-3-large"

    def test_get_current_model(self, temp_dir, monkeypatch):
        """Test getting current embedding model"""
        monkeypatch.chdir(temp_dir)
        config = EmbeddingConfig()

        model = config.get_current_model()

        assert isinstance(model, str)
        assert len(model) > 0

    def test_validate_compatibility(self, temp_dir, monkeypatch):
        """Test compatibility validation"""
        monkeypatch.chdir(temp_dir)
        config = EmbeddingConfig()

        compatibility = config.validate_compatibility()

        assert isinstance(compatibility, dict)
        assert "compatible" in compatibility

    def test_get_status_report(self, temp_dir, monkeypatch):
        """Test getting status report"""
        monkeypatch.chdir(temp_dir)
        config = EmbeddingConfig()

        report = config.get_status_report()

        assert isinstance(report, str)
        assert len(report) > 0

    def test_config_persistence(self, temp_dir, monkeypatch):
        """Test that config persists"""
        monkeypatch.chdir(temp_dir)

        # Create instance (should create config file)
        config1 = EmbeddingConfig()
        model1 = config1.get_current_model()

        # Create new instance (should load existing config)
        config2 = EmbeddingConfig()
        model2 = config2.get_current_model()

        assert model1 == model2
