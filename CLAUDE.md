# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an enhanced RAG (Retrieval-Augmented Generation) system that provides expert negotiation guidance by analyzing PDF sources of negotiation books. The system features a comprehensive Gradio web interface with admin capabilities, **multi-backend LLM support (OpenAI, Anthropic Claude, Ollama)**, intelligent model selection between default and premium models, and robust document management.

## Core Architecture

**Main Application**: `main.py` contains the complete RAG pipeline with integrated admin interface:

- **LLMBackendManager**: Centralized management of multiple LLM backends (OpenAI, Anthropic, Ollama)
- **ModelConfig**: Middleware class that handles model-specific parameters and creates LLM instances
- **EnhancedNegotiationRAG**: Core RAG system with admin integration, processing PDFs/DOCX/TXT, creating embeddings, and managing QA chains
- **AdminConfig**: Manages admin authentication, sessions, system prompts, and usage statistics
- **DocumentManager**: Handles file uploads, validation, and source document management
- **EmbeddingConfig**: Manages embedding model configuration and vectorstore compatibility
- **Dual Gradio Interface**: Combined user and admin interface with secure authentication

**Document Processing Flow**:
1. Multiple file formats (PDF, TXT, DOCX, DOC) in `sources/` directory are loaded via appropriate loaders
2. Documents are chunked using RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
3. FAISS vectorstore is created with configurable OpenAI embeddings (text-embedding-3-large default)
4. Vectorstore is persisted to `vectorstore/` with metadata for model compatibility
5. Admin can upload new documents and regenerate vectorstore via web interface

**Multi-Backend LLM Architecture**:
- **Supported Backends**: OpenAI, Anthropic Claude, Ollama (local and cloud)
- **OpenAI Models**: GPT-4o, GPT-4o Mini, O3 Mini, GPT-4 Turbo, GPT-4, GPT-3.5 Turbo
- **Anthropic Models**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku
- **Ollama Models**: Llama 3.1 (70B/8B), Llama 3 (70B/8B), Mistral 7B, Mixtral 8x7B, Qwen 2.5 (72B/14B), Phi-3 14B, Gemma 2 (27B/9B)
- **Model Configuration**: Separate default and premium model selection from any backend
- **Backend Manager**: LLMBackendManager handles model initialization, API keys, and backend-specific parameters
- **Intelligent Switching**: Dynamic model selection based on user preferences (default vs premium)
- **Parameter Handling**: Automatic filtering of unsupported parameters (e.g., temperature for o3-mini)
- **EmbeddingConfig**: Ensures compatibility between embedding models and vectorstore

## Development Commands

**Environment Setup**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Required Environment Variables**:
Create `.env` file based on `.env.example`:
```bash
# OpenAI (Required for default setup)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Claude (Optional - for Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Ollama Local (Optional - default: http://localhost:11434)
# OLLAMA_BASE_URL=http://localhost:11434

# Ollama Cloud (Optional - for hosted Ollama)
# OLLAMA_CLOUD_URL=https://your-ollama-instance.com
# OLLAMA_API_KEY=your_ollama_api_key_here
```

**Note**: Only OpenAI API key is required by default. Add other API keys to enable additional backends.

**Run Application**:
```bash
source .venv/bin/activate
python main.py

# Or use the startup script:
./run.sh
```

**Test PDF Loading**:
```bash
python test_rag.py
```

**Admin Access**:
- Default admin password: `admin123`
- Access via "Admin Panel" tab in web interface
- Change password in Admin Settings after first login
- Session-based authentication with configurable timeout

## Docker Deployment

**Docker Support**: The application includes full Docker and Docker Compose support for easy deployment on Ubuntu and other Linux systems.

**Quick Start with Docker**:
```bash
# Copy environment file and add API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Build and run with Docker Compose
docker compose up -d

# View logs
docker compose logs -f

# Access at http://localhost:7860
```

**Docker Files**:
- `Dockerfile` - Multi-stage build for optimized image size with non-root user
- `docker-compose.yml` - Complete orchestration with persistent volumes, health checks, and resource limits
- `.dockerignore` - Excludes unnecessary files from Docker build context
- `DEPLOYMENT.md` - Comprehensive deployment guide for production systems

**Docker Features**:
- Multi-stage build for minimal image size
- Non-root user execution for security
- Persistent volumes for vectorstore, uploads, sources, and config files
- Health checks for container monitoring
- Resource limits (2GB memory, 2 CPU cores by default)
- Automatic restart on failure
- JSON logging with rotation

**Production Deployment**: See `DEPLOYMENT.md` for complete guide including:
- Ubuntu system setup
- Security best practices
- Reverse proxy configuration (Nginx)
- HTTPS/SSL setup with Let's Encrypt
- Firewall configuration
- Backup and restore procedures
- Monitoring and maintenance tasks

## Key Implementation Details

**PLEASE Framework**: The system prompt implements a structured response format requiring:
- Detailed negotiation breakdown
- Calibrated questions
- Draft responses
- Scenario planning
- Self-assessment scoring (Polite, Logical, Empathetic, Assertive, Strategic, Engaging)

**Admin Features**:
- **System Prompt Management**: Customize the AI's behavior and response format
- **LLM Backend Configuration**: Select and configure multiple LLM backends (OpenAI, Anthropic, Ollama)
  - Choose default and premium models from any available backend
  - View backend status and API key configuration
  - Support for 20+ models across different providers
- **Document Upload**: Web-based file upload with validation for PDF, TXT, DOCX, DOC
- **Vectorstore Management**: Regenerate embeddings when documents change
- **Usage Statistics**: Track API usage, tokens, and costs by model
- **Embedding Configuration**: Monitor and manage embedding model compatibility
- **Session Management**: Secure admin authentication with expiring sessions

**Vectorstore Persistence**: The system checks for existing vectorstore on startup to avoid reprocessing documents. Admin can regenerate vectorstore via web interface when documents change.

**Model Selection**: Users can toggle between models via checkbox. The ModelConfig class ensures proper parameter handling for each model type.

**Document Sources**: Place negotiation books in `sources/` directory or upload via admin interface. Supports PDF, TXT, DOCX, and DOC formats. Current sources include "Getting to Yes", "Never Split the Difference", etc.

**Embedding Intelligence**: The EmbeddingConfig class automatically detects which embedding model was used to build the current vectorstore and ensures compatibility.

## File Structure

**Application Code**:
- `main.py` - Main application with enhanced RAG system, dual Gradio UI, and integrated admin panel
- `llm_backend_config.py` - Multi-backend LLM configuration and management (OpenAI, Anthropic, Ollama)
- `admin_config.py` - Admin authentication, sessions, prompts, and usage tracking
- `document_manager.py` - File upload handling, validation, and document management
- `embedding_config.py` - Embedding model configuration and vectorstore compatibility
- `text_preprocessor.py` - Intelligent text preprocessing for token optimization
- `prompt_manager.py` - System and user prompt template management
- `test_rag.py` - Testing utilities for document loading and embeddings

**Data Directories**:
- `sources/` - Source documents for RAG knowledge base (PDF, TXT, DOCX, DOC)
- `uploads/` - Temporary storage for uploaded files
- `vectorstore/` - Persisted FAISS embeddings with metadata (auto-generated)
- `utils/` - Utility scripts for vectorstore rebuilding

**Configuration Files**:
- `requirements.txt` - Python dependencies
- `.env` - API keys for LLM backends (create manually from .env.example)
- `.env.example` - Template for environment variables with all supported backends
- `llm_backend_config.json` - LLM backend and model selection configuration (auto-generated)
- `admin_config.json` - Admin configuration and settings (auto-generated)
- `admin_sessions.json` - Active admin sessions (auto-generated)
- `usage_stats.json` - API usage statistics (auto-generated)
- `embedding_config.json` - Embedding model configuration (auto-generated)
- `prompt_config.json` - Stored prompt templates (auto-generated)

**Docker Deployment**:
- `Dockerfile` - Multi-stage Docker build configuration
- `docker-compose.yml` - Docker Compose orchestration file
- `.dockerignore` - Files excluded from Docker build context
- `DEPLOYMENT.md` - Comprehensive deployment guide for production

## LLM Backend Configuration Guide

### Overview

NegotiatorPro supports multiple LLM backends, allowing you to choose the best model for your needs. You can mix and match backends - for example, use OpenAI's GPT-4o Mini as your default model and Claude 3.5 Sonnet as your premium model.

### Supported Backends

1. **OpenAI** (Default)
   - Models: GPT-4o, GPT-4o Mini, O3 Mini, GPT-4 Turbo, GPT-4, GPT-3.5 Turbo
   - Requires: `OPENAI_API_KEY`
   - Get key from: https://platform.openai.com/api-keys

2. **Anthropic Claude**
   - Models: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku
   - Requires: `ANTHROPIC_API_KEY`
   - Get key from: https://console.anthropic.com/

3. **Ollama (Local)**
   - Models: Llama 3.1, Llama 3, Mistral, Mixtral, Qwen 2.5, Phi-3, Gemma 2
   - Requires: Ollama installed locally
   - Install: https://ollama.ai/
   - Default URL: http://localhost:11434
   - No API key needed

4. **Ollama (Cloud)**
   - Same models as local Ollama
   - Requires: `OLLAMA_API_KEY` and `OLLAMA_CLOUD_URL`
   - For hosted Ollama instances

### Setup Instructions

#### 1. Configure Environment Variables

Edit your `.env` file to add API keys for the backends you want to use:

```bash
# Required for default setup
OPENAI_API_KEY=sk-...

# Optional: Add Claude support
ANTHROPIC_API_KEY=sk-ant-...

# Optional: Configure Ollama
OLLAMA_BASE_URL=http://localhost:11434
```

#### 2. Install Dependencies

If you want to use Claude models, ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

This will install `langchain-anthropic` and `anthropic` packages.

#### 3. Configure via Admin Panel

1. Navigate to the **Admin Panel** tab
2. Log in with your admin password
3. Go to the **🤖 LLM Backends** tab
4. View backend status and available models
5. Configure your **Default Model** (used for regular queries)
6. Configure your **Premium Model** (used when "Premium Model" is selected)
7. Click **Set Default Model** or **Set Premium Model** to save

#### 4. Using Ollama (Local)

To use local Ollama models:

1. Install Ollama: `curl https://ollama.ai/install.sh | sh`
2. Pull desired models: `ollama pull llama3.1:70b`
3. Verify Ollama is running: `ollama list`
4. In Admin Panel → LLM Backends, select "Ollama (Local)" as backend
5. Choose your model (e.g., llama3.1:70b)

### Model Selection Strategy

**Default Model**: Fast, cost-effective model for most queries
- Recommended: OpenAI GPT-4o Mini, Claude 3 Haiku, or Llama 3.1 8B

**Premium Model**: Advanced reasoning for complex negotiations
- Recommended: OpenAI O3 Mini, Claude 3.5 Sonnet, or Llama 3.1 70B

### Backend Configuration File

The system stores backend preferences in `llm_backend_config.json`:

```json
{
  "active_backend": "openai",
  "active_models": {
    "default": {
      "backend": "openai",
      "model": "gpt-4o-mini"
    },
    "premium": {
      "backend": "anthropic",
      "model": "claude-3-5-sonnet-20241022"
    }
  },
  "backend_settings": {
    "openai": {"enabled": true},
    "anthropic": {"enabled": true},
    "ollama": {"enabled": false}
  }
}
```

This file is auto-generated and managed through the admin interface.

### Troubleshooting

**Issue**: "API key not found" error
- **Solution**: Ensure API key is set in `.env` file and restart the application

**Issue**: Claude models not showing up
- **Solution**: Install required packages: `pip install langchain-anthropic anthropic`

**Issue**: Ollama connection failed
- **Solution**: Verify Ollama is running: `ollama serve` or check `OLLAMA_BASE_URL`

**Issue**: Model initialization failed
- **Solution**: Check logs for specific error. System will fallback to OpenAI models

### Cost Optimization

Each model has different pricing. View approximate costs in the backend configuration:

- **GPT-4o Mini**: $0.15/$0.60 per 1M input/output tokens (lowest OpenAI cost)
- **Claude 3 Haiku**: $0.25/$1.25 per 1M tokens (fastest Claude)
- **Ollama (Local)**: Free (self-hosted)

Use the cheaper default model for most queries and premium model only when needed.

## API Format and Compatibility

### Message Format

All LLM backends use a unified chat message format:

```python
messages = [
    {"role": "system", "content": "You are a negotiation expert..."},
    {"role": "user", "content": "How do I negotiate salary?"}
]
response = llm.invoke(messages)
```

### Backend-Specific API Details

**OpenAI (ChatOpenAI)**:
- API: OpenAI Chat Completion API (`/v1/chat/completions`)
- Native support for chat messages with roles
- Request format: `{model, messages: [{role, content}], temperature, ...}`

**Anthropic (ChatAnthropic)**:
- API: Anthropic Messages API (`/v1/messages`)
- Native support for chat messages
- Request format: `{model, messages: [{role, content}], system, ...}`
- Note: System message handled separately by Anthropic API
- Parameter mapping: `api_key` → `anthropic_api_key`

**Ollama (ChatOllama)**:
- API: Ollama Chat API (`/api/chat`)
- OpenAI-compatible chat format
- Request format: `{model, messages: [{role, content}], ...}`
- **Important**: Uses `ChatOllama` class (not base `Ollama` class)

### LangChain Abstraction

The system uses LangChain's chat model abstractions which automatically handle:
- Provider-specific parameter mapping
- API endpoint routing
- Request/response formatting
- Error handling and retries

This allows the application to use the same code regardless of which backend is active.