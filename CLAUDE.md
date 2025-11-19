# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an enhanced RAG (Retrieval-Augmented Generation) system that provides expert negotiation guidance by analyzing PDF sources of negotiation books. The system features a **modern React frontend** with **FastAPI backend**, **multi-backend LLM support (OpenAI, Anthropic Claude, Ollama)**, intelligent model selection between default and premium models, and robust document management.

## Core Architecture

**Frontend**: React 18 + TypeScript application in `frontend/` directory:
- **Components**: Sidebar, ChatContainer, ChatMessage, ChatInput
- **State Management**: Zustand for conversation and UI state
- **API Client**: Axios for backend communication
- **Routing**: Multi-session chat with conversation history
- **Markdown Support**: Rich formatting for AI responses

**API Layer**: FastAPI application in `backend/api/`:
- **Routes**: `/chat`, `/auth`, `/health` endpoints
- **Models**: Pydantic request/response validation
- **Middleware**: JWT authentication and CORS
- **Async**: Non-blocking request handling

**Backend Core**: Shared RAG system in `backend/`:
- **LLMBackendManager**: Centralized management of multiple LLM backends (OpenAI, Anthropic, Ollama)
- **ModelConfig**: Middleware class that handles model-specific parameters and creates LLM instances
- **EnhancedNegotiationRAG**: Core RAG system with admin integration, processing PDFs/DOCX/TXT, creating embeddings, and managing QA chains
- **AdminConfig**: Manages admin authentication, sessions, system prompts, and usage statistics
- **DocumentManager**: Handles file uploads, validation, and source document management
- **EmbeddingConfig**: Manages embedding model configuration and vectorstore compatibility

**Document Processing Flow**:
1. Multiple file formats (PDF, TXT, DOCX, DOC) in `sources/` directory are loaded via appropriate loaders
2. Documents are chunked using RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
3. FAISS vectorstore is created with configurable OpenAI embeddings (text-embedding-3-large default)
4. Vectorstore is persisted to `vectorstore/` with metadata for model compatibility
5. Admin can upload new documents via FastAPI endpoints (React admin UI coming soon)

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

**Component Interaction Patterns**:
- **Frontend ↔ API**: React frontend (port 5173) communicates with FastAPI backend (port 8000) via REST endpoints
- **Singleton Backend Manager**: `backend_manager` is a global singleton instance of `LLMBackendManager` used throughout the application
- **Model Creation Flow**: React UI → FastAPI `/chat` → ModelConfig.create_llm() → LLMBackendManager.create_llm_instance() → LangChain ChatModel
- **Configuration Persistence**: All settings auto-save to JSON files in root directory; no database required
- **Session-Based Auth**: JWT tokens for React frontend, UUID tokens for admin sessions (AdminConfig)
- **Vectorstore Lazy Loading**: EnhancedNegotiationRAG loads existing vectorstore on startup; regenerates only when explicitly requested
- **Usage Tracking**: All LLM calls logged to usage_stats.json with token counts and costs
- **Error Handling**: Backend failures trigger fallback to OpenAI default model with user notification

## Quick Reference

**Common Tasks**:
```bash
# Start development servers
./run-api.sh               # Start FastAPI backend (port 8000)
./run-frontend.sh          # Start React frontend (port 5173)

# Or manually:
# Backend: uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
# Frontend: cd frontend && npm run dev

# Run all tests
pytest

# Run tests with coverage
pytest --cov=. --cov-report=html

# Run specific test category
pytest -m unit             # Unit tests
pytest -m integration      # Integration tests

# Docker deployment (both services)
docker compose up -d       # Start backend + frontend
docker compose logs -f     # View logs (all services)
docker compose logs -f backend   # Backend logs only
docker compose logs -f frontend  # Frontend logs only
docker compose stop        # Stop all services

# Rebuild vectorstore
python scripts/rebuild_vectordb.py

# Test LLM backends
python test_llm_backends.py
```

## Development Commands

**Environment Setup**:
```bash
# Python backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# React frontend
cd frontend
npm install
cd ..
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

# Ollama Cloud (Optional - for cloud models like gpt-oss:120b-cloud)
# Get your API key from: https://ollama.com/settings/keys
# OLLAMA_CLOUD_URL=https://ollama.com
# OLLAMA_API_KEY=your_ollama_api_key_here
```

**Note**: Only OpenAI API key is required by default. Add other API keys to enable additional backends.

**Run Application**:
```bash
# Terminal 1 - Start FastAPI backend
source .venv/bin/activate
./run-api.sh
# Or: uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Start React frontend
./run-frontend.sh
# Or: cd frontend && npm run dev
```

**Testing**:
```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m docker        # Docker infrastructure tests

# Run specific test file
pytest tests/test_admin_config.py

# Legacy RAG testing
python test_rag.py
```

**Admin Access**:
- Default admin password: `admin123`
- Authentication via FastAPI `/auth` endpoints
- React admin UI under development
- Session-based authentication with JWT tokens

**Utility Scripts**:
```bash
# Rebuild vectorstore from command line
python scripts/rebuild_vectordb.py

# Test LLM backend connections
python test_llm_backends.py

# Debug ChatOllama integration
python debug_chatollama.py
```

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

# Access at:
# Frontend: http://localhost:5173
# Backend: http://localhost:8000
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

**Text Preprocessing**: The TextPreprocessor class provides intelligent token optimization:
- Removes email signatures, footers, and forwarding headers
- Strips legal boilerplate and confidentiality notices
- Context-aware stop word removal that preserves negotiation-critical content
- Preserves: emotions, numbers, prices, commitments, deadlines, names
- Can reduce token usage by up to 68% while maintaining context quality
- Optional preprocessing available via admin interface toggle

## File Structure

**Frontend Application**:
- `frontend/` - React 18 + TypeScript application
  - `src/`
    - `components/` - React components (Sidebar, ChatContainer, ChatMessage, ChatInput)
    - `store/` - Zustand state management (chatStore.ts)
    - `services/` - API client (api.ts with Axios)
    - `types/` - TypeScript type definitions
    - `App.tsx` - Main application component
    - `App.css` - Styles with Markdown support
    - `main.tsx` - React entry point
  - `package.json` - NPM dependencies
  - `vite.config.ts` - Vite build configuration
  - `index.html` - HTML entry point

**Backend Application**:
- `backend/` - All backend logic (organized modules)
  - `api/` - FastAPI application layer
    - `main.py` - FastAPI entry point
    - `routes/` - API endpoints (chat.py, auth.py, health.py)
    - `models/` - Pydantic request/response models
    - `middleware/` - Authentication middleware
  - `rag_engine.py` - Core RAG system with EnhancedNegotiationRAG and ModelConfig classes
  - `llm_backend_config.py` - Multi-backend LLM configuration (OpenAI, Anthropic, Ollama)
  - `admin_config.py` - Admin authentication, sessions, and usage tracking
  - `document_manager.py` - File upload handling and validation
  - `embedding_config.py` - Embedding model configuration and vectorstore compatibility
  - `text_preprocessor.py` - Intelligent text preprocessing for token optimization
  - `prompt_manager.py` - System and user prompt template management

**Data Directories**:
- `sources/` - Source documents for RAG knowledge base (PDF, TXT, DOCX, DOC)
- `uploads/` - Temporary storage for uploaded files
- `vectorstore/` - Persisted FAISS embeddings with metadata (auto-generated)
- `data/` - Runtime data (gitignored)
  - `db/` - Database files (PostgreSQL/SQLite)
  - `backups/` - Database backups

**Scripts**:
- `scripts/` - Utility scripts
  - `rebuild_vectordb.py` - Vectorstore regeneration utility
- `run-api.sh` - Start FastAPI backend (port 8000)
- `run-frontend.sh` - Start React frontend (port 5173)

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
- `Dockerfile` - Multi-stage Docker build (React + FastAPI)
- `docker-compose.yml` - Docker Compose with backend and frontend services
- `.dockerignore` - Files excluded from Docker build context

**Documentation**:
- `README.md` - Project overview and quick start (keep in root)
- `CLAUDE.md` - AI development guide (keep in root)
- `docs/` - All documentation
  - `deployment/` - DEPLOYMENT.md, DOCKER-DEPLOY.md
  - `features/` - ADMIN_FEATURES.md, OLLAMA_CLOUD_SETUP.md, QUICKSTART.md
  - `archive/` - Historical documentation (Gradio UI docs, POC migration files)
  - `TESTING.md` - Testing guide

**Database**:
- `migrations/` - PostgreSQL schema migrations
  - `001_initial_schema.sql` - Initial database schema
  - `README.md` - Migration documentation

**Testing**:
- `tests/` - Comprehensive test suite (100+ tests)
  - `test_docker.py` - Docker infrastructure and deployment tests
  - `test_admin_config.py` - Admin authentication and configuration tests
  - `test_document_manager.py` - Document upload and validation tests
  - `test_model_config.py` - LLM backend and model configuration tests
  - `test_modules.py` - Text preprocessor, prompt manager, embedding tests
  - `test_integration.py` - End-to-end RAG pipeline tests
  - `conftest.py` - Shared fixtures and test configuration
- `pytest.ini` - Pytest configuration with markers and coverage settings
- `.coveragerc` - Coverage reporting configuration
- `requirements-test.txt` - Test dependencies (pytest, coverage, etc.)

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

**Issue**: Ollama 404 error for cloud models (e.g., gpt-oss:120b-cloud)
- **Solution**: Use the "ollama-cloud" backend instead of "ollama" backend in Admin Panel → LLM Backends
- Set `OLLAMA_API_KEY` in `.env` (get from https://ollama.com/settings/keys)
- Set `OLLAMA_CLOUD_URL=https://ollama.com` in `.env` (or leave unset to use default)

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