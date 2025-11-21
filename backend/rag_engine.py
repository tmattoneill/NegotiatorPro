"""
RAG Engine Module

This module contains the core RAG (Retrieval-Augmented Generation) logic
for the NegotiatorPro system, extracted from main.py for better separation
of concerns and to prepare for React frontend migration.
"""

import os
import logging
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from .admin_config import AdminConfig
from .document_manager import DocumentManager
from .embedding_config import EmbeddingConfig
from .text_preprocessor import TextPreprocessor
from .prompt_manager import PromptManager
from .llm_backend_config import backend_manager
from .config_loader import config

logger = logging.getLogger(__name__)


# =============================================================================
# Global Model Configuration
# =============================================================================
# Logical role-based model configurations for the RAG system.
# Keys are logical roles, values are OpenAI-compatible kwargs.
# New code should use get_model_kwargs() with these logical keys.

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Primary chat model for user answers
    "chat_default": {
        "model": "gpt-4o",
        "temperature": 0.2,
        "max_tokens": None,
    },
    # Cheaper/faster chat model for simple queries
    "chat_mini": {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": None,
    },
    # Reasoning model for complex analysis (o-series don't support temperature)
    "rag_reasoning": {
        "model": "o4-mini",
        "reasoning_effort": "medium",
    },
    # Default embedding model for vectorstore
    "embedding_default": {
        "model": "text-embedding-3-small",
    },
    # Legacy models for backwards compatibility
    "legacy_gpt4": {
        "model": "gpt-4",
        "temperature": 0.3,
        "max_tokens": None,
    },
    "legacy_gpt35": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.3,
        "max_tokens": None,
    },
}

# Mapping from raw model names to logical keys for legacy compatibility
_MODEL_NAME_TO_KEY: Dict[str, str] = {
    "gpt-4o": "chat_default",
    "gpt-4o-mini": "chat_mini",
    "o4-mini": "rag_reasoning",
    "o3-mini": "rag_reasoning",  # Treat o3-mini as reasoning model
    "gpt-4": "legacy_gpt4",
    "gpt-3.5-turbo": "legacy_gpt35",
    "text-embedding-3-small": "embedding_default",
    "text-embedding-3-large": "embedding_default",
    "text-embedding-ada-002": "embedding_default",
}


def get_model_kwargs(model_key: str) -> Dict[str, Any]:
    """
    Get cleaned model kwargs for a logical model key.

    Args:
        model_key: Logical key from MODEL_CONFIGS (e.g., "chat_default", "chat_mini")

    Returns:
        Cleaned dict of kwargs suitable for passing to ChatOpenAI or similar.

    Raises:
        KeyError: If model_key is not found in MODEL_CONFIGS.
    """
    if model_key not in MODEL_CONFIGS:
        raise KeyError(f"Unknown model key '{model_key}'. Available: {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[model_key].copy()

    # Clean up the config:
    # 1. Remove temperature for o-series reasoning models (they don't support it)
    model_name = cfg.get("model", "")
    if model_name.startswith("o") and "temperature" in cfg:
        del cfg["temperature"]
        logger.debug(f"Removed temperature for reasoning model {model_name}")

    # 2. Remove max_tokens if None (don't send null to API)
    if cfg.get("max_tokens") is None:
        cfg.pop("max_tokens", None)

    logger.info(f"get_model_kwargs({model_key}): {cfg}")
    return cfg


def get_model_kwargs_legacy(model_name: str) -> Dict[str, Any]:
    """
    Legacy method for backwards compatibility.

    DEPRECATED: New code should use get_model_kwargs() with logical keys.

    This wrapper accepts either:
    - Logical keys (e.g., "chat_default") - passed directly to get_model_kwargs
    - Raw model names (e.g., "gpt-4o-mini") - mapped to logical key first

    Args:
        model_name: Either a logical key or a raw OpenAI model name.

    Returns:
        Cleaned dict of kwargs.
    """
    # Check if it's already a logical key
    if model_name in MODEL_CONFIGS:
        return get_model_kwargs(model_name)

    # Map raw model name to logical key
    model_key = _MODEL_NAME_TO_KEY.get(model_name)
    if model_key:
        logger.debug(f"Legacy mapping: {model_name} -> {model_key}")
        return get_model_kwargs(model_key)

    # Unknown model - return sensible defaults with warning
    logger.warning(f"Unknown model '{model_name}', using default config")
    return {"model": model_name, "temperature": 0.3}


class ModelConfig:
    """Model configuration middleware to handle different model parameters and backends"""

    def __init__(self):
        """Initialize with backend manager"""
        self.backend_manager = backend_manager

    @staticmethod
    def get_model_kwargs_legacy(model_name: str) -> Dict[str, Any]:
        """
        Legacy static method wrapper for backwards compatibility.

        DEPRECATED: Use module-level get_model_kwargs() with logical keys instead.
        """
        return get_model_kwargs_legacy(model_name)

    def get_model_kwargs_for_backend(self, backend_id: str, model_id: str) -> Dict[str, Any]:
        """Get appropriate kwargs for a specific backend and model via backend manager."""
        return self.backend_manager.get_llm_kwargs(backend_id, model_id)

    def create_llm(self, backend_id: str, model_id: str):
        """Create an LLM instance for the specified backend and model"""
        return self.backend_manager.create_llm_instance(backend_id, model_id)


class EnhancedNegotiationRAG:
    """
    Enhanced RAG system for negotiation advice.

    This class handles:
    - Document loading and processing
    - Vector store creation and management
    - LLM configuration and invocation
    - Context retrieval and question answering
    """

    def __init__(self):
        self.vectorstore = None
        self.default_qa_chain = None
        self.premium_qa_chain = None
        self.admin_config = AdminConfig()
        self.document_manager = DocumentManager()
        self.embedding_config = EmbeddingConfig()
        self.text_preprocessor = TextPreprocessor()
        self.prompt_manager = PromptManager()
        self.backend_manager = backend_manager
        self.model_config = ModelConfig()

    def load_documents(self) -> List[Any]:
        """Load and process documents from sources directory"""
        logger.info("Starting document loading...")
        docs = []

        # Get all supported documents
        documents_info = self.document_manager.list_source_documents()
        logger.info(f"Found {len(documents_info)} supported documents")

        for doc_info in documents_info:
            logger.info(f"Processing {doc_info['filename']}")
            start_time = time.time()

            try:
                file_path = doc_info['path']
                ext = doc_info['extension']

                # Load based on file type
                if ext == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif ext == '.txt':
                    loader = TextLoader(file_path, encoding='utf-8')
                elif ext == '.docx':
                    loader = Docx2txtLoader(file_path)
                elif ext == '.doc':
                    # Skip .doc files if unstructured is not available
                    try:
                        from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                        loader = UnstructuredWordDocumentLoader(file_path)
                    except ImportError:
                        logger.warning(f"Skipping .doc file {doc_info['filename']} - unstructured package not available")
                        continue
                else:
                    logger.warning(f"Unsupported file type: {ext}")
                    continue

                documents = loader.load()

                # Add source metadata
                for doc in documents:
                    doc.metadata['source_file'] = doc_info['filename']
                    doc.metadata['file_type'] = doc_info['type']

                docs.extend(documents)
                end_time = time.time()
                logger.info(f"Loaded {len(documents)} pages from {doc_info['filename']} in {end_time-start_time:.2f}s")

            except Exception as e:
                logger.error(f"Error loading {doc_info['filename']}: {e}")
                continue

        logger.info(f"Total documents loaded: {len(docs)} pages")
        return docs

    def create_chunks(self, documents: List[Any]) -> List[Any]:
        """Split documents into chunks"""
        logger.info("Starting text chunking...")
        start_time = time.time()

        # Get chunk settings from config
        chunk_size = config.get("rag.chunk_size", 1000)
        chunk_overlap = config.get("rag.chunk_overlap", 200)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)

        end_time = time.time()
        logger.info(f"Created {len(chunks)} text chunks in {end_time-start_time:.2f}s")
        return chunks

    def create_vectorstore(self, chunks: List[Any]):
        """Create FAISS vectorstore from document chunks"""
        logger.info("Starting vector embedding creation...")
        start_time = time.time()

        try:
            logger.info("Initializing OpenAI embeddings...")
            embedding_kwargs = self.embedding_config.get_embedding_kwargs()
            embeddings = OpenAIEmbeddings(**embedding_kwargs)
            logger.info(f"Using embedding model: {embedding_kwargs.get('model')}")

            logger.info(f"Creating FAISS vectorstore from {len(chunks)} chunks...")
            vectorstore = FAISS.from_documents(chunks, embeddings)

            end_time = time.time()
            logger.info(f"Vector store created successfully in {end_time-start_time:.2f}s")
            return vectorstore

        except Exception as e:
            logger.error(f"Error creating vectorstore: {e}")
            raise

    def save_vectorstore(self):
        """Save vectorstore to disk"""
        if self.vectorstore:
            try:
                logger.info("Saving vectorstore to disk...")
                self.vectorstore.save_local("vectorstore")
                logger.info("Vector store saved successfully")
            except Exception as e:
                logger.error(f"Error saving vectorstore: {e}")

    def load_vectorstore(self) -> bool:
        """Load vectorstore from disk"""
        try:
            logger.info("Attempting to load existing vectorstore...")
            embedding_kwargs = self.embedding_config.get_embedding_kwargs()
            embeddings = OpenAIEmbeddings(**embedding_kwargs)
            logger.info(f"Loading vectorstore with embedding model: {embedding_kwargs.get('model')}")
            self.vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
            logger.info("Vector store loaded successfully from disk")
            return True
        except Exception as e:
            logger.info(f"No existing vector store found: {e}")
            return False

    def regenerate_vectorstore(self) -> Dict[str, Any]:
        """Regenerate vectorstore from current documents"""
        logger.info("Starting vectorstore regeneration...")

        try:
            # Remove existing vectorstore
            vectorstore_path = Path("vectorstore")
            if vectorstore_path.exists():
                shutil.rmtree(vectorstore_path)
                logger.info("Removed existing vectorstore")

            # Reload documents and create new vectorstore
            documents = self.load_documents()
            if not documents:
                return {
                    "success": False,
                    "message": "No documents found to process"
                }

            chunks = self.create_chunks(documents)
            self.vectorstore = self.create_vectorstore(chunks)
            self.save_vectorstore()

            # Recreate LLM instances
            self.setup_llms()

            return {
                "success": True,
                "message": f"Vectorstore regenerated successfully with {len(documents)} documents and {len(chunks)} chunks"
            }

        except Exception as e:
            logger.error(f"Error regenerating vectorstore: {e}")
            return {
                "success": False,
                "message": f"Error regenerating vectorstore: {str(e)}"
            }

    def get_relevant_context(self, question: str, k: int = None) -> str:
        """Retrieve relevant context from vectorstore for the given question"""
        try:
            if not self.vectorstore:
                return "No knowledge base available."

            # Get k from config if not specified
            if k is None:
                k = config.get("rag.retrieval_k", 5)

            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            relevant_docs = retriever.invoke(question)

            # Combine retrieved documents into context
            context_parts = []
            for i, doc in enumerate(relevant_docs):
                context_parts.append(f"Source {i+1}: {doc.page_content}")

            return "\n\n".join(context_parts) if context_parts else "No relevant context found."

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return "Error retrieving context from knowledge base."

    def setup_llms(self):
        """Setup LLM instances for both models using backend manager"""
        logger.info("Setting up LLM instances...")

        # Get active model configurations from backend manager
        default_model_config = self.backend_manager.get_active_model_config("default")
        premium_model_config = self.backend_manager.get_active_model_config("premium")

        default_backend = default_model_config.get("backend", "openai")
        default_model = default_model_config.get("model", "gpt-4o-mini")
        premium_backend = premium_model_config.get("backend", "openai")
        premium_model = premium_model_config.get("model", "o3-mini")

        logger.info(f"Default model: {default_backend}/{default_model}")
        logger.info(f"Premium model: {premium_backend}/{premium_model}")

        # Create LLM instances using the model config
        try:
            self.default_llm = self.model_config.create_llm(default_backend, default_model)
            logger.info(f"Default LLM initialized: {default_backend}/{default_model}")
        except Exception as e:
            logger.error(f"Error creating default LLM: {e}")
            fallback_cfg = get_model_kwargs("chat_mini")
            logger.warning(f"Falling back to OpenAI {fallback_cfg['model']}")
            self.default_llm = ChatOpenAI(**fallback_cfg)

        try:
            self.premium_llm = self.model_config.create_llm(premium_backend, premium_model)
            logger.info(f"Premium LLM initialized: {premium_backend}/{premium_model}")
        except Exception as e:
            logger.error(f"Error creating premium LLM: {e}")
            fallback_cfg = get_model_kwargs("rag_reasoning")
            logger.warning(f"Falling back to OpenAI {fallback_cfg['model']}")
            self.premium_llm = ChatOpenAI(**fallback_cfg)

    def setup_system(self):
        """Initialize the RAG system"""
        logger.info("Starting RAG system setup...")
        setup_start = time.time()

        # Check embedding configuration compatibility
        compatibility = self.embedding_config.validate_compatibility()
        if not compatibility["compatible"]:
            logger.error("Embedding configuration incompatible!")
            for error in compatibility["errors"]:
                logger.error(f"  • {error}")
            logger.warning("Recommend rebuilding vectorstore with correct model")

        # Log current configuration
        logger.info(f"Embedding config: {self.embedding_config.get_current_model()}")

        # Try to load existing vectorstore first
        if not self.load_vectorstore():
            logger.info("Creating new vectorstore...")
            # Create new vectorstore
            documents = self.load_documents()
            if documents:
                chunks = self.create_chunks(documents)
                self.vectorstore = self.create_vectorstore(chunks)
                self.save_vectorstore()
            else:
                logger.warning("No documents found, vectorstore not created")
                # Don't return - still need to setup LLMs even without documents

        # Setup LLMs regardless of whether documents are loaded
        self.setup_llms()

        setup_end = time.time()
        logger.info(f"RAG system setup complete in {setup_end-setup_start:.2f}s!")

    def get_advice(
        self,
        question: str,
        use_premium_model: bool = False,
        use_preprocessing: bool = True,
        override_backend: Optional[str] = None,
        override_model: Optional[str] = None
    ) -> str:
        """
        Get negotiation advice based on the question using proper chat completion.

        Args:
            question: The user's negotiation question
            use_premium_model: Whether to use the premium model
            use_preprocessing: Whether to preprocess the input text
            override_backend: Optional backend override
            override_model: Optional model override

        Returns:
            The AI's response as a string
        """
        # Check for test/ping prompts first - skip full RAG workflow
        if self.prompt_manager.is_test_prompt(question):
            logger.info("Test prompt detected - returning quick response")
            return self.prompt_manager.get_test_response()

        if not hasattr(self, 'default_llm') or not hasattr(self, 'premium_llm'):
            return "System not initialized properly. Please check if documents are loaded."

        # Preprocess the question if enabled
        preprocessing_info = None
        if use_preprocessing and len(question.strip()) > 100:  # Only preprocess longer texts
            preprocessing_result = self.text_preprocessor.preprocess(question)
            question = preprocessing_result['processed_text']
            preprocessing_info = preprocessing_result
            logger.info(f"Preprocessing saved {preprocessing_result['tokens_saved']} tokens ({preprocessing_result['reduction_percentage']:.1f}% reduction)")

        try:
            # Select appropriate LLM based on model choice
            if override_backend and override_model:
                # User selected custom provider/model from dropdown
                llm = self.model_config.create_llm(override_backend, override_model)
                model_name = f"{override_backend}/{override_model}"
                logger.info(f"Using user-selected model: {model_name}")
            elif use_premium_model:
                llm = self.premium_llm
                model_config = self.backend_manager.get_active_model_config("premium")
                model_name = f"{model_config.get('backend', 'openai')}/{model_config.get('model', 'o3-mini')}"
                logger.info(f"Using premium model: {model_name}")
            else:
                llm = self.default_llm
                model_config = self.backend_manager.get_active_model_config("default")
                model_name = f"{model_config.get('backend', 'openai')}/{model_config.get('model', 'gpt-4o-mini')}"
                logger.info(f"Using default model: {model_name}")

            # Get relevant context from vectorstore
            context = self.get_relevant_context(question)

            # Get system and user prompts from prompt manager
            system_prompt, user_prompt = self.prompt_manager.get_prompts_for_chat(
                question=question,
                context=context
            )

            # Create messages for chat completion
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            logger.info(f"Sending chat completion request with {len(system_prompt)} char system prompt and {len(user_prompt)} char user prompt")

            # Call the LLM with proper chat format
            response = llm.invoke(messages)

            # Log usage (simplified - in production you'd get actual token counts)
            self.admin_config.log_usage(model_name, 1000)  # Placeholder token count

            # Extract content from response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)

        except Exception as e:
            import traceback
            logger.error(f"Error getting advice: {repr(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error args: {e.args}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return f"Error getting advice: {repr(e)}"
