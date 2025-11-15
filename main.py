import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
import glob
import logging
import time
import json
import shutil
from pathlib import Path
import uuid
from datetime import datetime

# Import our admin components
from admin_config import AdminConfig
from document_manager import DocumentManager
from embedding_config import EmbeddingConfig
from text_preprocessor import TextPreprocessor
from prompt_manager import PromptManager
from llm_backend_config import LLMBackendManager, backend_manager

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

class ModelConfig:
    """Model configuration middleware to handle different model parameters and backends"""

    def __init__(self):
        """Initialize with backend manager"""
        self.backend_manager = backend_manager

    @staticmethod
    def get_model_kwargs_legacy(model_name):
        """Legacy method for backwards compatibility - use get_model_kwargs instead"""
        MODEL_CONFIGS = {
            "gpt-4o-mini": {
                "model": "gpt-4o-mini",
                "temperature": 0.3,
                "max_tokens": None
            },
            "o3-mini": {
                "model": "o3-mini",
                # o3 models don't support temperature parameter
            },
            "gpt-4": {
                "model": "gpt-4",
                "temperature": 0.3,
                "max_tokens": None
            },
            "gpt-3.5-turbo": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.3,
                "max_tokens": None
            }
        }

        if model_name not in MODEL_CONFIGS:
            logger.warning(f"Unknown model {model_name}, using default config")
            return {"model": model_name, "temperature": 0.3}

        config = MODEL_CONFIGS[model_name].copy()
        # Filter out None values to avoid passing them to ChatOpenAI
        config = {k: v for k, v in config.items() if v is not None}
        logger.info(f"Using config for {model_name}: {config}")
        return config

    def get_model_kwargs(self, backend_id: str, model_id: str):
        """Get appropriate kwargs for a specific backend and model"""
        return self.backend_manager.get_llm_kwargs(backend_id, model_id)

    def create_llm(self, backend_id: str, model_id: str):
        """Create an LLM instance for the specified backend and model"""
        return self.backend_manager.create_llm_instance(backend_id, model_id)

class EnhancedNegotiationRAG:
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
        
    def load_documents(self):
        """Load and process PDF documents"""
        logger.info("Starting PDF document loading...")
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
    
    def create_chunks(self, documents):
        """Split documents into chunks"""
        logger.info("Starting text chunking...")
        start_time = time.time()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        end_time = time.time()
        logger.info(f"Created {len(chunks)} text chunks in {end_time-start_time:.2f}s")
        return chunks
    
    def create_vectorstore(self, chunks):
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
    
    def load_vectorstore(self):
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
    
    def regenerate_vectorstore(self):
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
    
    def get_relevant_context(self, question: str, k: int = 5) -> str:
        """Retrieve relevant context from vectorstore for the given question"""
        try:
            if not self.vectorstore:
                return "No knowledge base available."
            
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
            logger.info(f"✅ Default LLM initialized: {default_backend}/{default_model}")
        except Exception as e:
            logger.error(f"Error creating default LLM: {e}")
            logger.warning("Falling back to OpenAI gpt-4o-mini")
            self.default_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        try:
            self.premium_llm = self.model_config.create_llm(premium_backend, premium_model)
            logger.info(f"✅ Premium LLM initialized: {premium_backend}/{premium_model}")
        except Exception as e:
            logger.error(f"Error creating premium LLM: {e}")
            logger.warning("Falling back to OpenAI o3-mini")
            self.premium_llm = ChatOpenAI(model="o3-mini")
    
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
    
    def get_advice(self, question, use_premium_model=False, use_preprocessing=True, override_backend=None, override_model=None):
        """Get negotiation advice based on the question using proper chat completion"""
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

# Initialize the enhanced RAG system
rag_system = EnhancedNegotiationRAG()

def create_admin_interface_content():
    """Create admin interface content"""
    
    def authenticate_admin(password):
        """Authenticate admin user"""
        if rag_system.admin_config.verify_password(password):
            session_id = str(uuid.uuid4())
            rag_system.admin_config.create_session(session_id)
            return True, session_id, "Authentication successful"
        return False, "", "Invalid password"
    
    def check_admin_session(session_id):
        """Check if admin session is valid - AUTHENTICATION DISABLED"""
        # Authentication disabled for easier access
        return True
    
    def save_system_prompt(prompt, session_id):
        """Save system prompt"""
        if not check_admin_session(session_id):
            return "Session expired. Please log in again."
        
        rag_system.prompt_manager.update_system_prompt(prompt)
        return "System prompt saved successfully"
    
    def get_system_prompt(session_id):
        """Get current system prompt"""
        if not check_admin_session(session_id):
            return "Session expired. Please log in again."
        prompts = rag_system.prompt_manager.get_raw_prompts()
        return prompts.get("system", "")
    
    def save_user_prompt(prompt, session_id):
        """Save user prompt template"""
        if not check_admin_session(session_id):
            return "Session expired. Please log in again."
        
        rag_system.prompt_manager.update_user_prompt(prompt)
        return "User prompt template saved successfully"
    
    def get_user_prompt(session_id):
        """Get current user prompt template"""
        if not check_admin_session(session_id):
            return "Session expired. Please log in again."
        prompts = rag_system.prompt_manager.get_raw_prompts()
        return prompts.get("user", "")
    
    def upload_document(files, session_id):
        """Handle document upload"""
        if not check_admin_session(session_id):
            return "Session expired. Please log in again.", refresh_document_list(session_id)
        
        if not files:
            return "No files selected", refresh_document_list(session_id)
        
        results = []
        for file in files:
            if hasattr(file, 'name'):  # Gradio file object
                result = rag_system.document_manager.save_uploaded_file(file.name, Path(file.name).name)
            else:  # File path string
                result = rag_system.document_manager.save_uploaded_file(file, Path(file).name)
            
            if result["success"]:
                results.append(f"✅ {result['message']}")
            else:
                results.append(f"❌ {result['message']}")
        
        return "\n".join(results), refresh_document_list(session_id)
    
    def refresh_document_list(session_id):
        """Refresh document list"""
        if not check_admin_session(session_id):
            return "Session expired. Please log in again."
        
        documents = rag_system.document_manager.list_source_documents()
        if not documents:
            return "No documents found"
        
        doc_list = []
        for doc in documents:
            doc_list.append(f"📄 {doc['filename']} ({doc['size_mb']}MB) - {doc['type']}")
        
        return "\n".join(doc_list)
    
    def regenerate_vectorstore(session_id):
        """Regenerate vectorstore"""
        if not check_admin_session(session_id):
            return "Session expired. Please log in again."
        
        result = rag_system.regenerate_vectorstore()
        if result["success"]:
            return f"✅ {result['message']}"
        else:
            return f"❌ {result['message']}"
    
    def get_usage_stats(session_id):
        """Get usage statistics"""
        if not check_admin_session(session_id):
            return "Session expired. Please log in again."
        
        stats = rag_system.admin_config.get_usage_summary(30)
        
        summary = f"""📊 Usage Statistics (Last 30 Days)
        
Total Requests: {stats['total_requests']}
Total Tokens: {stats['total_tokens']:,}
Total Cost: ${stats['total_cost']:.4f}

Model Breakdown:"""
        
        for model, model_stats in stats['models'].items():
            summary += f"\n  {model}: {model_stats['requests']} requests, {model_stats['tokens']:,} tokens, ${model_stats['cost']:.4f}"
        
        return summary
    
    def get_embedding_status(session_id):
        """Get embedding configuration status"""
        if not check_admin_session(session_id):
            return "Session expired. Please log in again."
        
        return rag_system.embedding_config.get_status_report()
    
    def change_admin_password(current_password, new_password, confirm_password, session_id):
        """Change admin password"""
        if not check_admin_session(session_id):
            return "Session expired. Please log in again."

        if not rag_system.admin_config.verify_password(current_password):
            return "Current password is incorrect"

        if new_password != confirm_password:
            return "New passwords don't match"

        if len(new_password) < 6:
            return "New password must be at least 6 characters"

        rag_system.admin_config.change_password(new_password)
        return "Password changed successfully"

    def get_backend_status(session_id):
        """Get backend configuration status"""
        if not check_admin_session(session_id):
            return "Session expired. Please log in again."

        return rag_system.backend_manager.get_status_report()

    def get_backend_models(backend_id):
        """Get available models for a backend"""
        backend = rag_system.backend_manager.get_backend(backend_id)
        if not backend:
            return []

        models = []
        for model in backend.models:
            # Gradio dropdown format: (label, value)
            models.append((f"{model.name} - {model.description}", model.id))
        return models

    def set_default_model(backend_id, model_id, session_id):
        """Set default model"""
        if not check_admin_session(session_id):
            return "Session expired. Please log in again."

        try:
            rag_system.backend_manager.set_active_model("default", backend_id, model_id)
            # Reinitialize LLMs
            rag_system.setup_llms()
            return f"✅ Default model set to {backend_id}/{model_id}"
        except Exception as e:
            return f"❌ Error setting default model: {str(e)}"

    def set_premium_model(backend_id, model_id, session_id):
        """Set premium model"""
        if not check_admin_session(session_id):
            return "Session expired. Please log in again."

        try:
            rag_system.backend_manager.set_active_model("premium", backend_id, model_id)
            # Reinitialize LLMs
            rag_system.setup_llms()
            return f"✅ Premium model set to {backend_id}/{model_id}"
        except Exception as e:
            return f"❌ Error setting premium model: {str(e)}"

    def enable_backend(backend_id, enabled, session_id):
        """Enable or disable a backend"""
        if not check_admin_session(session_id):
            return "Session expired. Please log in again."

        try:
            rag_system.backend_manager.enable_backend(backend_id, enabled)
            status = "enabled" if enabled else "disabled"
            return f"✅ Backend {backend_id} {status}"
        except Exception as e:
            return f"❌ Error updating backend: {str(e)}"
    
    # Admin interface content (no wrapping Blocks)
    # Note: Authentication disabled for easier access
    session_state = gr.State("admin-session")  # Dummy session for compatibility

    # Admin content (authentication removed)
    with gr.Group(visible=True) as admin_content:
        gr.Markdown("### Admin Dashboard")
        gr.Markdown("*Manage system configuration and monitor usage*")
        
        with gr.Tabs():
            # System Configuration
            with gr.Tab("Configuration"):
                gr.Markdown("### System Prompt")
                gr.Markdown("*Define the AI's role and behavior. Use `{context}` for knowledge base content.*")
                system_prompt_text = gr.Textbox(
                    label="System Prompt Template",
                    lines=12,
                    placeholder="Enter the system prompt template...",
                    show_label=False
                )
                with gr.Row():
                    load_system_btn = gr.Button("Load Current", size="sm")
                    save_system_btn = gr.Button("Save Changes", variant="primary")
                system_status = gr.Textbox(label="Status", interactive=False, show_label=False)

                gr.Markdown("---")
                gr.Markdown("### User Prompt Template")
                gr.Markdown("*Format the user's question. Use `{question}` as placeholder.*")
                user_prompt_text = gr.Textbox(
                    label="User Prompt Template",
                    lines=5,
                    placeholder="Enter user prompt template...",
                    show_label=False
                )
                with gr.Row():
                    load_user_btn = gr.Button("Load Current", size="sm")
                    save_user_btn = gr.Button("Save Changes", variant="primary")
                user_status = gr.Textbox(label="Status", interactive=False, show_label=False)

            # Backend Configuration
            with gr.Tab("🤖 LLM Backends"):
                gr.Markdown("### Backend Status")
                refresh_backend_btn = gr.Button("🔄 Refresh Status", size="sm")
                backend_status_display = gr.Textbox(
                    label="Backend Status",
                    lines=15,
                    interactive=False,
                    show_label=False
                )

                gr.Markdown("---")
                gr.Markdown("### Configure Default Model")
                gr.Markdown("*This model is used for regular queries*")

                with gr.Row():
                    default_backend_dropdown = gr.Dropdown(
                        choices=[
                            ("OpenAI", "openai"),
                            ("Anthropic Claude", "anthropic"),
                            ("Ollama (Local)", "ollama"),
                            ("Ollama (Cloud)", "ollama-cloud")
                        ],
                        label="Backend",
                        value="openai",
                        interactive=True
                    )

                default_model_dropdown = gr.Dropdown(
                    choices=[],
                    label="Model",
                    interactive=True
                )

                set_default_model_btn = gr.Button("💾 Set Default Model", variant="primary")
                default_model_status = gr.Textbox(label="Status", interactive=False, show_label=False)

                gr.Markdown("---")
                gr.Markdown("### Configure Premium Model")
                gr.Markdown("*This model is used when 'Premium Model' is selected*")

                with gr.Row():
                    premium_backend_dropdown = gr.Dropdown(
                        choices=[
                            ("OpenAI", "openai"),
                            ("Anthropic Claude", "anthropic"),
                            ("Ollama (Local)", "ollama"),
                            ("Ollama (Cloud)", "ollama-cloud")
                        ],
                        label="Backend",
                        value="openai",
                        interactive=True
                    )

                premium_model_dropdown = gr.Dropdown(
                    choices=[],
                    label="Model",
                    interactive=True
                )

                set_premium_model_btn = gr.Button("💾 Set Premium Model", variant="primary")
                premium_model_status = gr.Textbox(label="Status", interactive=False, show_label=False)

                gr.Markdown("---")
                gr.Markdown("### API Key Configuration")
                gr.Markdown("""
**Required Environment Variables:**

- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic**: `ANTHROPIC_API_KEY`
- **Ollama Local**: No API key needed (default: http://localhost:11434)
- **Ollama Cloud**: `OLLAMA_API_KEY` and `OLLAMA_CLOUD_URL`

Set these in your `.env` file and restart the application.
                """)

            # Document Management
            with gr.Tab("Documents"):
                gr.Markdown("### Upload Knowledge Base")
                file_upload = gr.File(
                    label="Supported: PDF, TXT, DOC, DOCX",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".doc", ".docx"]
                )
                upload_btn = gr.Button("Upload Files", variant="primary")
                upload_status = gr.Textbox(label="Status", interactive=False, show_label=False)

                gr.Markdown("---")
                gr.Markdown("### Current Library")
                refresh_docs_btn = gr.Button("Refresh", size="sm")
                document_list = gr.Textbox(
                    label="Documents",
                    lines=10,
                    interactive=False,
                    show_label=False
                )

                gr.Markdown("---")
                gr.Markdown("### Vector Database")
                gr.Markdown("*Rebuild the knowledge base index when documents change*")
                regenerate_btn = gr.Button("Rebuild Index", variant="secondary")
                vectorstore_status = gr.Textbox(label="Status", interactive=False, show_label=False)

            # Usage Statistics
            with gr.Tab("Analytics"):
                gr.Markdown("### Usage Statistics")
                refresh_stats_btn = gr.Button("Refresh Stats", size="sm")
                usage_display = gr.Textbox(
                    label="Stats",
                    lines=15,
                    interactive=False,
                    show_label=False
                )

                gr.Markdown("---")
                gr.Markdown("### Embedding Configuration")
                refresh_embedding_btn = gr.Button("Check Status", size="sm")
                embedding_status = gr.Textbox(
                    label="Embedding Status",
                    lines=10,
                    interactive=False,
                    show_label=False
                )

            # Admin Settings
            with gr.Tab("Security"):
                gr.Markdown("### Change Admin Password")
                current_pwd = gr.Textbox(
                    label="Current Password",
                    type="password",
                    placeholder="Enter current password"
                )
                new_pwd = gr.Textbox(
                    label="New Password",
                    type="password",
                    placeholder="Enter new password"
                )
                confirm_pwd = gr.Textbox(
                    label="Confirm Password",
                    type="password",
                    placeholder="Confirm new password"
                )
                change_pwd_btn = gr.Button("Update Password", variant="primary")
                pwd_status = gr.Textbox(label="Status", interactive=False, show_label=False)

    # Event handlers
    # Note: Authentication removed - admin panel is now publicly accessible

    # System prompt handlers
    load_system_btn.click(get_system_prompt, inputs=[session_state], outputs=[system_prompt_text])
    save_system_btn.click(save_system_prompt, inputs=[system_prompt_text, session_state], outputs=[system_status])
    
    # User prompt handlers
    load_user_btn.click(get_user_prompt, inputs=[session_state], outputs=[user_prompt_text])
    save_user_btn.click(save_user_prompt, inputs=[user_prompt_text, session_state], outputs=[user_status])
    
    # Document handlers
    upload_btn.click(upload_document, inputs=[file_upload, session_state], outputs=[upload_status, document_list])
    refresh_docs_btn.click(refresh_document_list, inputs=[session_state], outputs=[document_list])
    regenerate_btn.click(regenerate_vectorstore, inputs=[session_state], outputs=[vectorstore_status])
    
    # Usage stats
    refresh_stats_btn.click(get_usage_stats, inputs=[session_state], outputs=[usage_display])
    refresh_embedding_btn.click(get_embedding_status, inputs=[session_state], outputs=[embedding_status])
    
    # Password change
    change_pwd_btn.click(
        change_admin_password,
        inputs=[current_pwd, new_pwd, confirm_pwd, session_state],
        outputs=[pwd_status]
    )

    # Backend configuration handlers
    def update_default_models(backend_id):
        """Update model dropdown when backend changes"""
        models = get_backend_models(backend_id)
        # models is list of (label, value) tuples, get first value
        return gr.Dropdown(choices=models, value=models[0][1] if models else None)

    def update_premium_models(backend_id):
        """Update model dropdown when backend changes"""
        models = get_backend_models(backend_id)
        # models is list of (label, value) tuples, get first value
        return gr.Dropdown(choices=models, value=models[0][1] if models else None)

    refresh_backend_btn.click(get_backend_status, inputs=[session_state], outputs=[backend_status_display])

    default_backend_dropdown.change(
        update_default_models,
        inputs=[default_backend_dropdown],
        outputs=[default_model_dropdown]
    )

    premium_backend_dropdown.change(
        update_premium_models,
        inputs=[premium_backend_dropdown],
        outputs=[premium_model_dropdown]
    )

    set_default_model_btn.click(
        set_default_model,
        inputs=[default_backend_dropdown, default_model_dropdown, session_state],
        outputs=[default_model_status]
    )

    set_premium_model_btn.click(
        set_premium_model,
        inputs=[premium_backend_dropdown, premium_model_dropdown, session_state],
        outputs=[premium_model_status]
    )

def create_main_interface_content():
    """Create main user interface content"""

    def get_provider_choices():
        """Get list of available provider choices"""
        return [
            ("OpenAI", "openai"),
            ("Anthropic Claude", "anthropic"),
            ("Ollama (Local)", "ollama"),
            ("Ollama (Cloud)", "ollama-cloud")
        ]

    def get_model_choices(provider):
        """Get list of model choices for a given provider"""
        backend = rag_system.backend_manager.get_backend(provider)
        if backend:
            return [(model.name, model.id) for model in backend.models]
        return []

    def get_default_provider():
        """Get default provider from config"""
        default_config = rag_system.backend_manager.get_active_model_config("default")
        return default_config.get("backend", "openai")

    def get_default_model(provider):
        """Get default model for a provider"""
        default_config = rag_system.backend_manager.get_active_model_config("default")
        if default_config.get("backend") == provider:
            return default_config.get("model")
        # Return first model for this provider
        models = get_model_choices(provider)
        return models[0][1] if models else None

    def update_model_choices(provider):
        """Update model dropdown based on selected provider"""
        choices = get_model_choices(provider)
        if choices:
            return gr.Dropdown(choices=choices, value=choices[0][1])
        return gr.Dropdown(choices=[], value=None)

    def negotiate_advisor(question, partner_context="", use_premium=False, use_preprocessing=True, provider=None, model=None):
        """Main function for the Gradio interface"""
        # Apply default user prompt if configured
        default_prompt = rag_system.admin_config.get_default_user_prompt()
        if default_prompt and not question.strip():
            question = default_prompt

        if partner_context.strip():
            enhanced_question = f"Context about my negotiation partner: {partner_context}\n\nMy question: {question}"
        else:
            enhanced_question = question

        if not enhanced_question.strip():
            return "Please enter a negotiation question.", "Ready • Please enter a question", ""

        # Use provided provider/model or fall back to configured defaults
        if provider is None or model is None:
            if use_premium:
                config = rag_system.backend_manager.get_active_model_config("premium")
            else:
                config = rag_system.backend_manager.get_active_model_config("default")
            provider = config.get("backend", "openai")
            model = config.get("model", "gpt-4o-mini")

        # Get model name for display
        model_info = rag_system.backend_manager.get_model_info(provider, model)
        display_model_name = model_info.name if model_info else model
        status = f"⏳ Thinking with {display_model_name}..."

        # Get advice with preprocessing option and custom model selection
        advice = rag_system.get_advice(
            enhanced_question,
            use_premium_model=use_premium,
            use_preprocessing=use_preprocessing,
            override_backend=provider,
            override_model=model
        )

        # Generate preprocessing stats if preprocessing was used
        preprocessing_stats = ""
        if use_preprocessing and len(enhanced_question.strip()) > 100:
            try:
                preprocessing_result = rag_system.text_preprocessor.preprocess(enhanced_question)
                preprocessing_stats = f"""
### 📊 Optimization Stats

- **Original tokens:** {preprocessing_result['original_tokens']:,}
- **Optimized tokens:** {preprocessing_result['processed_tokens']:,}
- **Tokens saved:** {preprocessing_result['tokens_saved']:,} ({preprocessing_result['reduction_percentage']:.1f}% reduction)
- **Cost savings:** ${preprocessing_result['estimated_cost_savings']:.4f}
- **Characters removed:** {preprocessing_result['character_reduction']:,}
"""
            except Exception as e:
                preprocessing_stats = f"Optimization stats unavailable: {str(e)}"

        # Update final status
        final_status = f"✓ Complete • Used {display_model_name}"

        return advice, final_status, preprocessing_stats
    
    # Main interface content (no wrapping Blocks)
    with gr.Row():
        # Sidebar-style column for inputs
        with gr.Column(scale=2):
            gr.Markdown("### Your Question")

            question = gr.Textbox(
                label="Negotiation challenge",
                placeholder="How should I respond to a lowball offer?",
                lines=5,
                max_lines=8,
                show_label=False
            )

            partner_info = gr.Textbox(
                label="Context (optional)",
                placeholder="Additional context about the other party...",
                lines=3,
                max_lines=5,
                show_label=False
            )

            gr.Markdown("### Settings")

            with gr.Group():
                # Provider and model selection
                default_provider = get_default_provider()
                default_models = get_model_choices(default_provider)
                default_model_value = get_default_model(default_provider)

                provider_dropdown = gr.Dropdown(
                    choices=get_provider_choices(),
                    value=default_provider,
                    label="Provider",
                    info="Select LLM provider"
                )

                model_dropdown = gr.Dropdown(
                    choices=default_models,
                    value=default_model_value,
                    label="Model",
                    info="Select model to use"
                )

                use_premium_model = gr.Checkbox(
                    label="Use Premium Model",
                    value=False,
                    info="Override with premium model from admin config"
                )

                use_preprocessing = gr.Checkbox(
                    label="Optimize Text",
                    value=True,
                    info="Reduce tokens and costs"
                )

            submit_btn = gr.Button("Get Advice", variant="primary", size="lg")

            model_status = gr.Textbox(
                label="Status",
                value="Ready",
                interactive=False,
                max_lines=1,
                show_label=False
            )

        # Main chat-style column for output
        with gr.Column(scale=3):
            gr.Markdown("### Response")

            advice_output = gr.Markdown(
                value="",
                label="Response"
            )

            preprocessing_stats = gr.Markdown(
                value="",
                visible=True
            )
    
    # Update model dropdown when provider changes
    provider_dropdown.change(
        fn=update_model_choices,
        inputs=[provider_dropdown],
        outputs=[model_dropdown]
    )

    submit_btn.click(
        fn=negotiate_advisor,
        inputs=[question, partner_info, use_premium_model, use_preprocessing, provider_dropdown, model_dropdown],
        outputs=[advice_output, model_status, preprocessing_stats]
    )
    
    # Example questions section - minimal and clean
    gr.Markdown("---")
    gr.Markdown("### Examples")

    example_questions = [
        "How do I respond to 'That's my final offer'?",
        "What's the best way to make the first offer?",
        "How can I build rapport with a difficult negotiator?",
        "They're using high-pressure tactics. What should I do?",
        "How do I negotiate when I have less leverage?",
        "What questions should I ask to understand their interests?"
    ]

    with gr.Row():
        for i in range(0, len(example_questions), 3):
            with gr.Column():
                if i < len(example_questions):
                    def create_example_handler(example_text):
                        def handler():
                            return example_text, "", False, True, "Ready • Using gpt-4o-mini"
                        return handler

                    gr.Button(example_questions[i], size="sm").click(
                        create_example_handler(example_questions[i]),
                        outputs=[question, partner_info, use_premium_model, use_preprocessing, model_status]
                    )
                if i+1 < len(example_questions):
                    def create_example_handler2(example_text):
                        def handler():
                            return example_text, "", False, True, "Ready • Using gpt-4o-mini"
                        return handler

                    gr.Button(example_questions[i+1], size="sm").click(
                        create_example_handler2(example_questions[i+1]),
                        outputs=[question, partner_info, use_premium_model, use_preprocessing, model_status]
                    )
                if i+2 < len(example_questions):
                    def create_example_handler3(example_text):
                        def handler():
                            return example_text, "", False, True, "Ready • Using gpt-4o-mini"
                        return handler

                    gr.Button(example_questions[i+2], size="sm").click(
                        create_example_handler3(example_questions[i+2]),
                        outputs=[question, partner_info, use_premium_model, use_preprocessing, model_status]
                    )

    gr.Markdown("---")
    gr.Markdown("*Powered by expert negotiation literature including 'Getting to Yes', 'Never Split the Difference', and more.*")

if __name__ == "__main__":
    logger.info("=== Starting Enhanced Negotiation RAG System ===")
    
    try:
        logger.info("Initializing RAG system...")
        rag_system.setup_system()
        
        logger.info("Creating combined interface...")

        # Create single interface with both main and admin functionality
        # Using default Gradio theme
        with gr.Blocks(
            title="NegotiatorPro"
        ) as combined_demo:
            # Clean minimal header
            gr.Markdown("# NegotiatorPro")
            gr.Markdown("*AI negotiation guidance from expert strategies*")

            with gr.Tabs():
                with gr.Tab("Chat"):
                    # Embed main interface content directly
                    create_main_interface_content()

                with gr.Tab("Admin"):
                    # Embed admin interface content directly
                    create_admin_interface_content()
        
        logger.info("Launching application...")
        combined_demo.launch(
            share=True, 
            server_name="0.0.0.0", 
            server_port=7860, 
            show_api=False,
            favicon_path="static/favicon.ico"
        )
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise