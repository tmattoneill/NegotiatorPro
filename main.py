import os
import gradio as gr
from dotenv import load_dotenv
import logging
import uuid
from datetime import datetime

# Import backend components
from backend import (
    AdminConfig,
    DocumentManager,
    EmbeddingConfig,
    TextPreprocessor,
    PromptManager,
    LLMBackendManager,
    backend_manager,
    EnhancedNegotiationRAG,
    ModelConfig
)

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

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
            # For Ollama backends, get actual available models
            if backend.provider == "ollama":
                base_url = None
                if provider == "ollama-cloud":
                    base_url = os.getenv("OLLAMA_CLOUD_URL", "https://ollama.com")
                else:
                    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

                available_models = rag_system.backend_manager.get_ollama_available_models(base_url)
                return [(model.name, model.id) for model in available_models]
            else:
                # For OpenAI and Anthropic, use predefined models
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