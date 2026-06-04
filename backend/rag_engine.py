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
from .runpod_llm import ChatRunPod, is_runpod_available

logger = logging.getLogger(__name__)


class LLMGenerationError(Exception):
    """Raised when the LLM fails to generate advice (bad key, upstream down, etc.).

    The chat route maps this to an HTTP error so the frontend renders a proper
    error state instead of displaying the exception text as an assistant message.
    """


class MissingAPIKeyError(LLMGenerationError):
    """Raised when the selected provider has no usable key in the user's profile.

    This is a user-configuration problem, not an upstream failure, so the chat
    route maps it to a 400 (with the provider-specific message) rather than a
    502. Subclasses LLMGenerationError so existing handlers still catch it.
    """


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

    def create_llm(self, backend_id: str, model_id: str, api_key: Optional[str] = None):
        """Create an LLM instance for the specified backend and model"""
        return self.backend_manager.create_llm_instance(backend_id, model_id, api_key=api_key)


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

    def get_relevant_context(
        self,
        question: str,
        k: int = None,
        tags_filter: Optional[List[str]] = None,
    ) -> str:
        """Retrieve relevant context from vectorstore for the given question.

        When tags_filter is set (e.g. ['sales'] or ['negotiation']), only
        chunks whose 'tags' metadata field overlaps with the filter list are
        returned. Uses FAISS post-retrieval metadata filtering.
        """
        try:
            if not self.vectorstore:
                return "No knowledge base available."

            if k is None:
                k = config.get("rag.retrieval_k", 5)

            if tags_filter:
                filter_fn = lambda meta: bool(
                    set(meta.get("tags", [])) & set(tags_filter)
                )
                # Fetch many more candidates so post-retrieval tag filtering
                # leaves enough results. The corpus is imbalanced (negotiation
                # outnumbers sales ~5:1), so fetch_k must be high enough that
                # the minority tag still surfaces in the top-N by similarity.
                fetch_k = max(k * 40, 200)
                relevant_docs = self.vectorstore.similarity_search(
                    question, k=k, fetch_k=fetch_k, filter=filter_fn
                )
            else:
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
                relevant_docs = retriever.invoke(question)

            context_parts = []
            for i, doc in enumerate(relevant_docs):
                context_parts.append(f"Source {i+1}: {doc.page_content}")

            return "\n\n".join(context_parts) if context_parts else "No relevant context found."

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return "Error retrieving context from knowledge base."

    def _create_llm_with_fallback(self, backend: str, model: str, model_type: str) -> Any:
        """
        Create an LLM instance with cascading fallback support.

        Fallback order:
        1. Configured backend/model
        2. OpenAI (if OPENAI_API_KEY available)
        3. RunPod basic model (if RUNPOD_API_KEY available)

        Args:
            backend: The backend ID to use
            model: The model ID to use
            model_type: Either "default" or "premium" (for logging)

        Returns:
            An LLM instance
        """
        # Try configured backend first
        try:
            llm = self.model_config.create_llm(backend, model)
            logger.info(f"{model_type.capitalize()} LLM initialized: {backend}/{model}")
            return llm
        except Exception as e:
            logger.warning(f"Error creating {model_type} LLM ({backend}/{model}): {e}")

        # Fallback to OpenAI if API key is available
        if os.getenv("OPENAI_API_KEY"):
            try:
                fallback_model = "chat_mini" if model_type == "default" else "rag_reasoning"
                fallback_cfg = get_model_kwargs(fallback_model)
                logger.warning(f"Falling back to OpenAI {fallback_cfg['model']}")
                return ChatOpenAI(**fallback_cfg)
            except Exception as e:
                logger.warning(f"OpenAI fallback failed: {e}")

        # Ultimate fallback to RunPod basic model
        if is_runpod_available():
            try:
                logger.warning("Falling back to RunPod basic model")
                return ChatRunPod()
            except Exception as e:
                logger.error(f"RunPod fallback failed: {e}")

        # If all fallbacks fail, raise an error
        raise RuntimeError(
            f"Failed to create {model_type} LLM. No backends available. "
            "Please configure at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, "
            "OLLAMA_BASE_URL, or RUNPOD_API_KEY"
        )

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

        # Create LLM instances with fallback support
        self.default_llm = self._create_llm_with_fallback(default_backend, default_model, "default")
        self.premium_llm = self._create_llm_with_fallback(premium_backend, premium_model, "premium")

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

    @staticmethod
    def _extract_usage(response: Any) -> Dict[str, int]:
        """Pull real token counts from a LangChain chat response.

        LangChain standardises `usage_metadata` across providers with
        input/output/total tokens; Anthropic adds cache read/creation counts
        under input_token_details. Returns zeros when a backend doesn't report
        usage (e.g. some Ollama builds).
        """
        um = getattr(response, "usage_metadata", None)
        if not isinstance(um, dict):
            # Some backends omit usage; tests mock the LLM (usage_metadata is a
            # Mock, not a dict). Treat anything non-dict as "no usage reported".
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                    "cache_read": 0, "cache_creation": 0}
        input_tokens = int(um.get("input_tokens", 0) or 0)
        output_tokens = int(um.get("output_tokens", 0) or 0)
        total = int(um.get("total_tokens", input_tokens + output_tokens) or 0)
        details = um.get("input_token_details", {}) or {}
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total,
            "cache_read": int(details.get("cache_read", 0) or 0),
            "cache_creation": int(details.get("cache_creation", 0) or 0),
        }

    def get_advice(
        self,
        question: str,
        use_premium_model: bool = False,
        use_preprocessing: bool = True,
        override_backend: Optional[str] = None,
        override_model: Optional[str] = None,
        mode: str = "auto",
        user_api_keys: Optional[Dict[str, str]] = None,
        return_usage: bool = False,
    ):
        """
        Get negotiation advice based on the question using proper chat completion.

        Args:
            question: The user's negotiation question
            use_premium_model: Whether to use the premium model
            use_preprocessing: Whether to preprocess the input text
            override_backend: Optional backend override
            override_model: Optional model override
            return_usage: When True, return (answer, usage_dict) instead of just
                the answer string. usage_dict has input/output/total token counts
                and Anthropic cache_read/cache_creation when available.

        Returns:
            The AI's response as a string, or (answer, usage) when return_usage.
        """
        if not hasattr(self, 'default_llm') or not hasattr(self, 'premium_llm'):
            raise LLMGenerationError("RAG system not initialized — vectorstore or LLMs unavailable.")

        def resolve_user_key(backend_id: str) -> Optional[str]:
            """Resolve a provider's key from the USER's profile only.

            User negotiations must never fall back to a system env key — if the
            selected provider needs a key the user hasn't configured, fail
            clearly instead of silently using a system key (which also gets sent
            to the wrong provider's endpoint).
            """
            key = (user_api_keys or {}).get(backend_id)
            backend = self.backend_manager.get_backend(backend_id)
            if backend and backend.requires_api_key and not key:
                raise MissingAPIKeyError(
                    f"No {backend.name} API key found in your profile. "
                    f"Add one in Settings to use this provider."
                )
            return key

        # Check for test/ping prompts - call LLM but skip RAG context retrieval
        if self.prompt_manager.is_test_prompt(question):
            logger.info("Test prompt detected - using simplified LLM call (no RAG)")
            try:
                # Use the same LLM selection logic as regular prompts
                if override_backend and override_model:
                    api_key = resolve_user_key(override_backend)
                    llm = self.model_config.create_llm(override_backend, override_model, api_key=api_key)
                elif use_premium_model:
                    llm = self.premium_llm
                else:
                    llm = self.default_llm

                messages = [
                    {"role": "system", "content": self.prompt_manager.get_test_system_prompt()},
                    {"role": "user", "content": self.prompt_manager.get_test_user_prompt(question)}
                ]
                response = llm.invoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                usage = self._extract_usage(response)
                return (content, usage) if return_usage else content
            except Exception as e:
                logger.error(f"Test prompt LLM call failed: {e}")
                raise LLMGenerationError(f"Connection test failed: {e}") from e

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
                api_key = resolve_user_key(override_backend)
                llm = self.model_config.create_llm(override_backend, override_model, api_key=api_key)
                backend_id = override_backend
                model_name = f"{override_backend}/{override_model}"
                logger.info(f"Using user-selected model: {model_name}")
            elif use_premium_model:
                llm = self.premium_llm
                model_config = self.backend_manager.get_active_model_config("premium")
                backend_id = model_config.get('backend', 'openai')
                model_name = f"{backend_id}/{model_config.get('model', 'o3-mini')}"
                logger.info(f"Using premium model: {model_name}")
            else:
                llm = self.default_llm
                model_config = self.backend_manager.get_active_model_config("default")
                backend_id = model_config.get('backend', 'openai')
                model_name = f"{backend_id}/{model_config.get('model', 'gpt-4o-mini')}"
                logger.info(f"Using default model: {model_name}")

            # Resolve mode → tags_filter for scoped retrieval
            tags_filter: Optional[List[str]] = None
            if mode == "sales":
                tags_filter = ["sales"]
            elif mode == "negotiation":
                tags_filter = ["negotiation"]

            # Get relevant context from vectorstore
            context = self.get_relevant_context(question, tags_filter=tags_filter)

            # Split the prompt into a static prefix (meta + persona, identical
            # every request) and the per-request context. mode selects which
            # persona stacks on the meta prompt; mode=auto uses meta only.
            static_system, context_block, user_prompt = self.prompt_manager.get_prompt_parts(
                question=question,
                context=context,
                mode=mode,
            )

            # Provider-aware system message for prompt caching.
            #   anthropic: mark the static prefix with cache_control so it's
            #     cached across requests; context goes in a second, uncached
            #     block after the breakpoint.
            #   openai: a single static-first string lets OpenAI auto-cache the
            #     identical prefix (>1024 tokens).
            #   ollama/others: no caching; the string form is equivalent.
            if backend_id == "anthropic":
                system_content = [
                    {"type": "text", "text": static_system,
                     "cache_control": {"type": "ephemeral"}},
                ]
                if context_block:
                    system_content.append({"type": "text", "text": context_block})
                system_message = {"role": "system", "content": system_content}
                sys_len = len(static_system) + len(context_block)
            else:
                system_text = (
                    f"{static_system.rstrip()}\n\n{context_block}\n"
                    if context_block else static_system
                )
                system_message = {"role": "system", "content": system_text}
                sys_len = len(system_text)

            # Create messages for chat completion
            messages = [
                system_message,
                {"role": "user", "content": user_prompt},
            ]

            logger.info(f"Sending chat completion request with {sys_len} char system prompt and {len(user_prompt)} char user prompt (backend={backend_id})")

            # Call the LLM with proper chat format
            response = llm.invoke(messages)

            # Real token metering from the response (replaces the old placeholder
            # of 1000). cost stays 0.0 for now — the per-model pricing fields are
            # mislabeled (per-1M values in a *_per_1k field), so deriving cost
            # from them would be 1000x off. Tokens are what we need to verify
            # caching; cost is a follow-up once pricing units are fixed.
            usage = self._extract_usage(response)
            self.admin_config.log_usage(model_name, usage["total_tokens"])
            logger.info(
                "Token usage [%s] input=%d output=%d total=%d cache_read=%d cache_creation=%d",
                model_name, usage["input_tokens"], usage["output_tokens"],
                usage["total_tokens"], usage["cache_read"], usage["cache_creation"],
            )

            content = response.content if hasattr(response, 'content') else str(response)
            return (content, usage) if return_usage else content

        except LLMGenerationError:
            raise
        except Exception as e:
            import traceback
            logger.error(f"Error getting advice: {repr(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error args: {e.args}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise LLMGenerationError(f"Error getting advice: {repr(e)}") from e
