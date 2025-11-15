"""
NegotiatorPro Backend Module

This package contains all backend logic for the NegotiatorPro RAG system,
including LLM configuration, document management, admin functionality, and
the core RAG engine.
"""

from .admin_config import AdminConfig
from .document_manager import DocumentManager
from .embedding_config import EmbeddingConfig
from .llm_backend_config import LLMBackendManager, backend_manager
from .prompt_manager import PromptManager
from .text_preprocessor import TextPreprocessor
from .rag_engine import EnhancedNegotiationRAG, ModelConfig

__all__ = [
    "AdminConfig",
    "DocumentManager",
    "EmbeddingConfig",
    "EnhancedNegotiationRAG",
    "LLMBackendManager",
    "backend_manager",
    "ModelConfig",
    "PromptManager",
    "TextPreprocessor",
]
