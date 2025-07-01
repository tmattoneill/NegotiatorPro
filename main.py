import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

class ModelConfig:
    """Model configuration middleware to handle different model parameters"""
    
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
    
    @staticmethod
    def get_model_kwargs(model_name):
        """Get appropriate kwargs for a specific model"""
        if model_name not in ModelConfig.MODEL_CONFIGS:
            logger.warning(f"Unknown model {model_name}, using default config")
            return {"model": model_name, "temperature": 0.3}
        
        config = ModelConfig.MODEL_CONFIGS[model_name].copy()
        # Filter out None values to avoid passing them to ChatOpenAI
        config = {k: v for k, v in config.items() if v is not None}
        logger.info(f"Using config for {model_name}: {config}")
        return config

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
        """Setup LLM instances for both models"""
        logger.info("Setting up LLM instances...")
        
        # Create LLM instances for both models with model-specific parameters
        default_config = ModelConfig.get_model_kwargs("gpt-4o-mini")
        premium_config = ModelConfig.get_model_kwargs("o3-mini")
        
        self.default_llm = ChatOpenAI(**default_config)
        self.premium_llm = ChatOpenAI(**premium_config)
    
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
                return
        
        self.setup_llms()
        
        setup_end = time.time()
        logger.info(f"RAG system setup complete in {setup_end-setup_start:.2f}s!")
    
    def get_advice(self, question, use_premium_model=False, use_preprocessing=True):
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
            if use_premium_model:
                llm = self.premium_llm
                model_name = "o3-mini"
                logger.info("Using o3-mini model for this query")
            else:
                llm = self.default_llm
                model_name = "gpt-4o-mini"
                logger.info("Using gpt-4o-mini model for this query")
            
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
        """Check if admin session is valid"""
        if not session_id:
            return False
        return rag_system.admin_config.is_valid_session(session_id)
    
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
    
    # Admin interface content (no wrapping Blocks)
    # Session state
    session_state = gr.State("")
    
    # Authentication modal
    with gr.Group() as auth_group:
        gr.Markdown("## 🔐 Admin Authentication")
        admin_password = gr.Textbox(label="Admin Password", type="password")
        auth_btn = gr.Button("Login", variant="primary")
        auth_status = gr.Textbox(label="Status", interactive=False)
    
    # Admin content (hidden until authenticated)
    with gr.Group(visible=False) as admin_content:
        gr.Markdown("## Admin Dashboard")
        
        with gr.Tabs():
            # System Configuration
            with gr.Tab("System Config"):
                gr.Markdown("### System Prompt Configuration")
                gr.Markdown("*System prompt defines the AI's role, expertise, and instructions. Use {context} placeholder for knowledge base content.*")
                system_prompt_text = gr.Textbox(
                    label="System Prompt Template",
                    lines=15,
                    placeholder="Enter the system prompt template (use {context} for knowledge base content)..."
                )
                with gr.Row():
                    load_system_btn = gr.Button("Load Current")
                    save_system_btn = gr.Button("Save", variant="primary")
                system_status = gr.Textbox(label="Status", interactive=False)
                
                gr.Markdown("### User Prompt Template")
                gr.Markdown("*User prompt template formats the user's question. Use {question} placeholder for the user's input.*")
                user_prompt_text = gr.Textbox(
                    label="User Prompt Template",
                    lines=5,
                    placeholder="Enter user prompt template (use {question} for the user's input)..."
                )
                with gr.Row():
                    load_user_btn = gr.Button("Load Current")
                    save_user_btn = gr.Button("Save", variant="primary")
                user_status = gr.Textbox(label="Status", interactive=False)
            
            # Document Management
            with gr.Tab("Documents"):
                gr.Markdown("### Upload Documents")
                file_upload = gr.File(
                    label="Upload Documents (PDF, TXT, DOC, DOCX)",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".doc", ".docx"]
                )
                upload_btn = gr.Button("Upload", variant="primary")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
                
                gr.Markdown("### Current Documents")
                refresh_docs_btn = gr.Button("Refresh List")
                document_list = gr.Textbox(
                    label="Documents",
                    lines=10,
                    interactive=False
                )
                
                gr.Markdown("### Vector Database")
                regenerate_btn = gr.Button("Regenerate Vector Database", variant="secondary")
                vectorstore_status = gr.Textbox(label="Status", interactive=False)
            
            # Usage Statistics
            with gr.Tab("Usage Stats"):
                gr.Markdown("### API Usage Statistics")
                refresh_stats_btn = gr.Button("Refresh Stats")
                usage_display = gr.Textbox(
                    label="Usage Statistics",
                    lines=15,
                    interactive=False
                )
                
                gr.Markdown("### Embedding Configuration")
                refresh_embedding_btn = gr.Button("Check Embedding Status")
                embedding_status = gr.Textbox(
                    label="Embedding Status",
                    lines=10,
                    interactive=False
                )
            
            # Admin Settings
            with gr.Tab("Admin Settings"):
                gr.Markdown("### Change Admin Password")
                current_pwd = gr.Textbox(label="Current Password", type="password")
                new_pwd = gr.Textbox(label="New Password", type="password")
                confirm_pwd = gr.Textbox(label="Confirm New Password", type="password")
                change_pwd_btn = gr.Button("Change Password", variant="primary")
                pwd_status = gr.Textbox(label="Status", interactive=False)
    
    # Authentication handler
    def handle_auth(password):
        success, session_id, message = authenticate_admin(password)
        if success:
            return (
                gr.update(visible=False),  # Hide auth
                gr.update(visible=True),   # Show admin content
                session_id,
                message
            )
        else:
            return (
                gr.update(visible=True),   # Keep auth visible
                gr.update(visible=False),  # Hide admin content
                "",
                message
            )
    
    # Event handlers
    auth_btn.click(
        handle_auth,
        inputs=[admin_password],
        outputs=[auth_group, admin_content, session_state, auth_status]
    )
    
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

def create_main_interface_content():
    """Create main user interface content"""
    
    def negotiate_advisor(question, partner_context="", use_premium=False, use_preprocessing=True):
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
            return "Please enter a negotiation question.", "Ready: Please enter a question", ""
        
        # Update model status
        model_name = "o3-mini" if use_premium else "gpt-4o-mini"
        status = f"Processing with {model_name}..."
        
        # Get advice with preprocessing option
        advice = rag_system.get_advice(enhanced_question, use_premium_model=use_premium, use_preprocessing=use_preprocessing)
        
        # Generate preprocessing stats if preprocessing was used
        preprocessing_stats = ""
        if use_preprocessing and len(enhanced_question.strip()) > 100:
            try:
                preprocessing_result = rag_system.text_preprocessor.preprocess(enhanced_question)
                preprocessing_stats = f"""📊 **Text Optimization Results:**
- Original tokens: {preprocessing_result['original_tokens']:,}
- Optimized tokens: {preprocessing_result['processed_tokens']:,}
- Tokens saved: {preprocessing_result['tokens_saved']:,} ({preprocessing_result['reduction_percentage']:.1f}% reduction)
- Estimated cost savings: ${preprocessing_result['estimated_cost_savings']:.4f}
- Characters removed: {preprocessing_result['character_reduction']:,}"""
            except Exception as e:
                preprocessing_stats = f"Preprocessing stats unavailable: {str(e)}"
        
        # Update final status
        final_status = f"Completed with {model_name}"
        
        return advice, final_status, preprocessing_stats
    
    # Main interface content (no wrapping Blocks)
    gr.Markdown("Get expert negotiation guidance based on proven strategies from leading negotiation books including 'Getting to Yes', 'Never Split the Difference', and more.")
    
    with gr.Row():
        with gr.Column(scale=1):
            partner_info = gr.Textbox(
                label="About Your Negotiation Partner (Optional)",
                placeholder="e.g., Experienced executive, tends to be aggressive, budget-conscious, deadline pressure...",
                lines=3
            )
            question = gr.Textbox(
                label="Your Negotiation Question",
                placeholder="e.g., How should I handle a lowball offer? What's the best way to anchor the price?",
                lines=4
            )
            
            with gr.Row():
                use_premium_model = gr.Checkbox(
                    label="🚀 Use o3-mini (Premium Model)",
                    value=False,
                    info="Default: gpt-4o-mini (faster, cost-effective) | Premium: o3-mini (more advanced reasoning, no temperature control)"
                )
                
            with gr.Row():
                use_preprocessing = gr.Checkbox(
                    label="⚡ Smart Text Optimization",
                    value=True,
                    info="Remove email signatures, footers, and fluff to reduce token usage and costs"
                )
            
            model_status = gr.Textbox(
                label="Model Status",
                value="Ready: gpt-4o-mini (default)",
                interactive=False,
                max_lines=1
            )
            
            submit_btn = gr.Button("Get Negotiation Advice", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            advice_output = gr.Textbox(
                label="Negotiation Advice",
                lines=15,
                interactive=False
            )
            preprocessing_stats = gr.Markdown(
                label="Optimization Statistics",
                value="",
                visible=True
            )
    
    submit_btn.click(
        fn=negotiate_advisor,
        inputs=[question, partner_info, use_premium_model, use_preprocessing],
        outputs=[advice_output, model_status, preprocessing_stats]
    )
    
    # Example questions section
    gr.Markdown("### Example Questions You Can Ask:")
    
    example_questions = [
        "How do I respond to 'That's my final offer'?",
        "What's the best way to make the first offer?",
        "How can I build rapport with a difficult negotiator?",
        "They're using high-pressure tactics. What should I do?",
        "How do I negotiate when I have less leverage?",
        "What questions should I ask to understand their interests?",
        "How do I handle emotional manipulation in negotiations?",
        "What's the best way to counter anchoring tactics?",
        "How do I create win-win solutions?",
        "What should I do when they walk away from the table?"
    ]
    
    with gr.Row():
        for i in range(0, len(example_questions), 2):
            with gr.Column():
                if i < len(example_questions):
                    def create_example_handler(example_text):
                        def handler():
                            return example_text, "", False, True, "Ready: gpt-4o-mini (default)"
                        return handler
                    
                    gr.Button(example_questions[i], size="sm").click(
                        create_example_handler(example_questions[i]),
                        outputs=[question, partner_info, use_premium_model, use_preprocessing, model_status]
                    )
                if i+1 < len(example_questions):
                    def create_example_handler2(example_text):
                        def handler():
                            return example_text, "", False, True, "Ready: gpt-4o-mini (default)"
                        return handler
                    
                    gr.Button(example_questions[i+1], size="sm").click(
                        create_example_handler2(example_questions[i+1]),
                        outputs=[question, partner_info, use_premium_model, use_preprocessing, model_status]
                    )
    
    gr.Markdown("---")
    gr.Markdown("*This advisor draws from negotiation expertise in your PDF library. Always adapt advice to your specific situation and legal/ethical constraints.*")

if __name__ == "__main__":
    logger.info("=== Starting Enhanced Negotiation RAG System ===")
    
    try:
        logger.info("Initializing RAG system...")
        rag_system.setup_system()
        
        logger.info("Creating combined interface...")
        
        # Custom CSS to handle font fallbacks and reduce console errors
        custom_css = """
        * {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        }
        
        /* Hide manifest and font loading errors */
        .gradio-container {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        """
        
        # Create single interface with both main and admin functionality
        with gr.Blocks(
            title="Negotiation Advisor with Admin Panel", 
            theme=gr.themes.Soft(),
            css=custom_css
        ) as combined_demo:
            gr.Markdown("# 🤝 Negotiation Advisor with Admin Panel")
            
            with gr.Tabs():
                with gr.Tab("🤝 Negotiation Advisor"):
                    # Embed main interface content directly
                    create_main_interface_content()
                    
                with gr.Tab("🔧 Admin Panel"):
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