# 🤝 NegotiatorPro

An AI-powered negotiation advisor that leverages expert knowledge from leading negotiation books to provide personalized guidance. Built with advanced RAG (Retrieval-Augmented Generation) technology, multi-backend LLM support, and intelligent text preprocessing for optimal performance.

## ✨ Features

### 🎯 **Core Functionality**
- **Expert Negotiation Guidance**: Draws from "Getting to Yes", "Never Split the Difference", and other proven negotiation frameworks
- **Multi-Backend LLM Support**: Choose from 20+ models across OpenAI, Anthropic Claude, and Ollama
- **Dual AI Models**: Configure separate default (fast, cost-effective) and premium (advanced reasoning) models from any backend
- **Smart Text Optimization**: Reduces token usage by up to 68% while preserving negotiation context
- **Comprehensive Analysis**: Provides detailed breakdowns, draft responses, calibrated questions, and scenario planning

### 🤖 **Supported LLM Backends**

**OpenAI**
- GPT-4o, GPT-4o Mini, O3 Mini, GPT-4 Turbo, GPT-4, GPT-3.5 Turbo
- Industry-leading models with proven performance

**Anthropic Claude**
- Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku
- Advanced reasoning and natural conversation

**Ollama (Local & Cloud)**
- Llama 3.1 (70B/8B), Llama 3 (70B/8B), Mistral 7B, Mixtral 8x7B
- Qwen 2.5 (72B/14B), Phi-3 14B, Gemma 2 (27B/9B)
- Self-hosted or cloud deployment options

**Mix and Match**: Use OpenAI GPT-4o Mini as your default and Claude 3.5 Sonnet as premium!

### 🛠️ **Advanced Features**
- **Intelligent Preprocessing**: Removes email signatures, footers, legal disclaimers, and fluff
- **Context-Aware**: Preserves critical negotiation elements (emotions, numbers, commitments, deadlines)
- **Real-time Statistics**: Track token usage, cost savings, and optimization results
- **Admin Dashboard**: Full system management, LLM backend configuration, and prompt customization
- **Document Management**: Web-based upload for PDF, TXT, DOCX, DOC files
- **Vectorstore Intelligence**: Automatic embedding model compatibility detection
- **Usage Analytics**: Track API usage, tokens, and costs by model and backend

### 📊 **Text Optimization Engine**
- **Email Content Cleaning**: Removes signatures, footers, forwarding headers
- **Legal Boilerplate Removal**: Strips confidentiality notices and disclaimers
- **Smart Stop Word Removal**: Context-aware filtering that preserves meaning
- **Negotiation-Critical Preservation**: Never removes prices, emotions, deadlines, or commitments

## 🚀 Quick Start

### Prerequisites

- **Docker Deployment** (Recommended): Docker and Docker Compose
- **Local Development**: Python 3.8+ and at least one LLM API key (OpenAI recommended)

### Option 1: Docker Deployment (Recommended) 🐳

Perfect for production deployments on Ubuntu/Linux systems.

1. **Clone and configure**
   ```bash
   git clone <repository-url>
   cd NegotiatorPro
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY (required)
   # Optionally add ANTHROPIC_API_KEY or OLLAMA_API_KEY
   ```

2. **Start with Docker**
   ```bash
   docker compose up -d
   ```

3. **Access the application**
   - Open your browser to `http://localhost:7860`
   - Admin Panel (default password: `admin123`)

📖 **See [docs/deployment/DOCKER-DEPLOY.md](docs/deployment/DOCKER-DEPLOY.md) for the quick start guide**
📖 **See [docs/deployment/DEPLOYMENT.md](docs/deployment/DEPLOYMENT.md) for production deployment**

### Option 2: Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd NegotiatorPro
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   Create a `.env` file in the root directory:
   ```bash
   # Required: At least one LLM backend API key
   OPENAI_API_KEY=sk-your_openai_api_key_here

   # Optional: Add for multi-backend support
   ANTHROPIC_API_KEY=sk-ant-your_anthropic_api_key_here

   # Optional: Ollama (local installation)
   OLLAMA_BASE_URL=http://localhost:11434

   # Optional: Ollama Cloud
   OLLAMA_CLOUD_URL=https://ollama.com
   OLLAMA_API_KEY=your_ollama_api_key_here
   ```

5. **Run the application**
   ```bash
   python main.py
   # Or use the startup script:
   ./scripts/run.sh
   ```

6. **Access the interface**
   - Open your browser to the URL shown in the terminal (typically `http://localhost:7860`)
   - Use the "Negotiation Advisor" tab for asking questions
   - Use the "Admin Panel" tab for administration (default password: `admin123`)

## 💡 How to Use

### For Users

1. **Ask Negotiation Questions**: Enter your specific negotiation scenario or question
2. **Provide Context** (Optional): Add information about your negotiation partner
3. **Choose Model**: Toggle "Use Premium Model" to switch between default and premium models
4. **Get Expert Advice**: Receive structured guidance following the PLEASE framework

### Example Questions

- "How do I respond to 'That's my final offer'?"
- "What's the best way to make the first offer?"
- "How can I build rapport with a difficult negotiator?"
- "They're using high-pressure tactics. What should I do?"

### For Administrators

Access the Admin Panel to:

- **🤖 LLM Backends**: Configure OpenAI, Anthropic Claude, or Ollama models
  - View backend status and available models
  - Set default model (fast, cost-effective)
  - Set premium model (advanced reasoning)
  - Mix and match backends (e.g., GPT-4o Mini + Claude 3.5 Sonnet)
- **📝 System Prompts**: Customize how the AI responds
- **📄 Documents**: Upload new negotiation resources to the knowledge base
- **📊 Usage Stats**: Track API costs, usage patterns, and token consumption
- **⚙️ Admin Settings**: Update admin password and session configuration

## 🏗️ System Architecture

### Core Components

**Backend Management**
- **LLMBackendManager**: Centralized management of multiple LLM backends (OpenAI, Anthropic, Ollama)
- **ModelConfig**: Middleware for model-specific parameters and LLM instance creation
- **EnhancedNegotiationRAG**: Core RAG system with admin integration
- **AdminConfig**: Authentication, sessions, and system settings management
- **DocumentManager**: File upload handling and document processing
- **EmbeddingConfig**: Embedding model configuration and vectorstore compatibility
- **TextPreprocessor**: Intelligent text optimization for token reduction
- **PromptManager**: System and user prompt template management

### Data Flow

1. **Document Processing**: Files (PDF/TXT/DOCX/DOC) are loaded, chunked, and converted to embeddings
2. **Vector Storage**: FAISS vectorstore persists embeddings with metadata
3. **Query Processing**: User questions are enhanced with retrieved context from knowledge base
4. **Backend Selection**: ModelConfig selects appropriate LLM backend based on user preference
5. **AI Response**: Selected model generates structured negotiation advice using PLEASE framework
6. **Usage Tracking**: Admin system logs usage statistics with token counts and costs

### Component Interaction

- **Singleton Backend Manager**: Global `backend_manager` instance manages all LLM backends
- **Model Creation Flow**: UI selection → ModelConfig.create_llm() → LLMBackendManager.create_llm_instance() → LangChain ChatModel
- **Configuration Persistence**: All settings auto-save to JSON files (no database required)
- **Session-Based Auth**: UUID tokens stored in admin_sessions.json
- **Vectorstore Lazy Loading**: Loads existing vectorstore on startup; regenerates only when requested
- **Error Handling**: Backend failures trigger fallback to OpenAI with user notification

## 📁 Project Structure

```
NegotiatorPro/
├── main.py                      # Gradio UI entry point
├── backend/                     # Backend modules (organized)
│   ├── rag_engine.py           # Core RAG system and ModelConfig
│   ├── llm_backend_config.py   # Multi-backend LLM management
│   ├── admin_config.py         # Admin auth and sessions
│   ├── document_manager.py     # File upload handling
│   ├── embedding_config.py     # Embedding configuration
│   ├── text_preprocessor.py    # Text optimization
│   └── prompt_manager.py       # Prompt templates
├── scripts/                     # Utility scripts
│   ├── run.sh                  # Application startup
│   └── rebuild_vectordb.py     # Vectorstore regeneration
├── sources/                     # Source documents (PDF, TXT, DOCX, DOC)
├── uploads/                     # Temporary upload storage
├── vectorstore/                 # Generated FAISS embeddings
├── tests/                       # Test suite (100+ tests)
│   ├── test_docker.py          # Docker infrastructure
│   ├── test_admin_config.py    # Admin system
│   ├── test_document_manager.py # Document handling
│   ├── test_model_config.py    # LLM backends
│   ├── test_modules.py         # Supporting modules
│   ├── test_integration.py     # Integration tests
│   └── conftest.py             # Test fixtures
├── docs/                        # Documentation
│   ├── deployment/             # DEPLOYMENT.md, DOCKER-DEPLOY.md
│   ├── features/               # ADMIN_FEATURES.md, OLLAMA_CLOUD_SETUP.md
│   └── TESTING.md              # Testing guide
├── migrations/                  # Database migrations
│   └── 001_initial_schema.sql
├── .github/workflows/
│   └── test.yml                # CI/CD pipeline
├── requirements.txt            # Python dependencies
├── requirements-test.txt       # Test dependencies
├── .env                        # Environment variables (create from .env.example)
├── .env.example                # Environment template
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Docker orchestration
├── pytest.ini                  # Test configuration
├── .coveragerc                 # Coverage configuration
├── README.md                   # This file
└── CLAUDE.md                   # AI development guide

Auto-generated configuration files:
├── llm_backend_config.json     # LLM backend settings
├── admin_config.json           # Admin configuration
├── admin_sessions.json         # Active sessions
├── usage_stats.json            # Usage statistics
├── embedding_config.json       # Embedding config
└── prompt_config.json          # Prompt templates
```

## ⚙️ Configuration

### Environment Variables

**Required** (at least one):
```bash
# OpenAI (Recommended for default setup)
OPENAI_API_KEY=sk-your_openai_api_key_here
```

**Optional** (for multi-backend support):
```bash
# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-your_anthropic_api_key_here

# Ollama Local (default: http://localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434

# Ollama Cloud
OLLAMA_CLOUD_URL=https://ollama.com
OLLAMA_API_KEY=your_ollama_api_key_here
```

### LLM Backend Configuration

Configure via **Admin Panel → 🤖 LLM Backends**:

**Default Model** (fast, cost-effective):
- Recommended: OpenAI GPT-4o Mini, Claude 3 Haiku, Llama 3.1 8B
- Used for standard negotiation queries

**Premium Model** (advanced reasoning):
- Recommended: OpenAI O3 Mini, Claude 3.5 Sonnet, Llama 3.1 70B
- Used when "Use Premium Model" is toggled

**Cost Optimization**:
- GPT-4o Mini: $0.15/$0.60 per 1M input/output tokens
- Claude 3 Haiku: $0.25/$1.25 per 1M tokens
- Ollama (Local): Free (self-hosted)

### Admin Settings

Access via **Admin Panel → ⚙️ Admin Settings**:
- Default admin password: `admin123` (change immediately!)
- Session duration: 24 hours (configurable)
- Upload limits: 50MB maximum file size

### Embedding Models

Supported OpenAI embedding models:
- `text-embedding-3-large` (default): 3072 dimensions, highest quality
- `text-embedding-3-small`: 1536 dimensions, good balance
- `text-embedding-ada-002`: 1536 dimensions, legacy

## 🛠️ Advanced Usage

### PLEASE Framework

The system prompt implements a structured response format:

- **P**olite: Maintain professional, respectful tone
- **L**ogical: Provide structured reasoning and analysis
- **E**mpathetic: Understand all parties' positions and interests
- **A**ssertive: Advocate for favorable outcomes
- **S**trategic: Think several moves ahead, plan scenarios
- **E**ngaging: Keep interactions productive and actionable

Each response includes:
- Detailed negotiation breakdown
- Calibrated questions to ask
- Draft response suggestions
- Scenario planning (best/worst/likely outcomes)
- Self-assessment scoring across PLEASE dimensions

### Document Management

**Supported Formats**: PDF, DOCX, DOC, TXT

**Upload Process**:
1. Navigate to Admin Panel → 📄 Documents
2. Click "Upload Document" and select file(s)
3. System validates format and size
4. Click "Regenerate Vector Database" to process new documents

**Auto-Processing**:
- Documents are chunked (1000 chars, 200 overlap)
- Embeddings are generated using configured model
- FAISS vectorstore is updated with new content
- Metadata tracks document information

### Text Preprocessing

Enable via Admin Panel to optimize token usage:

**Features**:
- Removes email signatures and footers
- Strips forwarding headers (FW:, RE:, etc.)
- Eliminates legal boilerplate
- Context-aware stop word removal

**Preservation**:
- Prices and numbers
- Emotions and sentiment
- Commitments and deadlines
- Names and entities

**Results**: Up to 68% token reduction while maintaining context quality

## 🔒 Security Features

- **Session-based Authentication**: Secure admin access with UUID tokens
- **Password Protection**: Hashed password storage
- **File Validation**: Type, size, and content validation on upload
- **Session Management**: Automatic cleanup of expired sessions
- **Non-root Docker**: Container runs as non-root user
- **Environment Isolation**: API keys stored in .env (gitignored)

## 🧪 Testing

NegotiatorPro includes a comprehensive test suite with 100+ tests.

### Running Tests

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
pytest -m docker        # Docker tests only

# Run specific test file
pytest tests/test_admin_config.py

# Legacy RAG testing
python test_rag.py

# Test LLM backends
python test_llm_backends.py
```

### Test Coverage

- **Docker Infrastructure**: Dockerfile, docker-compose, deployment validation
- **Unit Tests**: Admin, documents, models, text preprocessing, prompts, embeddings
- **Integration Tests**: End-to-end RAG pipeline, model switching
- **Security**: Automated scanning with Bandit and Safety
- **CI/CD**: Automated testing on GitHub Actions (Python 3.9, 3.10, 3.11)

**Coverage**: 90%+ across core modules

📖 **See [docs/TESTING.md](docs/TESTING.md) for complete testing guide**

## 🐳 Docker Deployment

### Features

- **Multi-stage build**: Optimized image size (~800MB)
- **Non-root user**: Security best practices
- **Health checks**: Container monitoring
- **Persistent volumes**: Data survives restarts (vectorstore, uploads, sources, config)
- **Resource limits**: Configurable CPU/memory (default: 2GB/2 cores)
- **Auto-restart**: Production reliability
- **JSON logging**: Structured logs with rotation

### Quick Commands

```bash
# Start application
docker compose up -d

# Stop application
docker compose stop

# Rebuild containers
docker compose build --no-cache

# View logs
docker compose logs -f

# Check status
docker compose ps

# Access shell in container
docker compose exec negotiator bash

# Remove everything
docker compose down -v
```

📖 **See [docs/deployment/DOCKER-DEPLOY.md](docs/deployment/DOCKER-DEPLOY.md) for quick start**
📖 **See [docs/deployment/DEPLOYMENT.md](docs/deployment/DEPLOYMENT.md) for production deployment**

## 📊 Usage Analytics

Access via **Admin Panel → 📊 Usage Stats**:

**Metrics Tracked**:
- Daily request counts
- Token usage (input/output)
- API costs by model and backend
- Model performance comparison
- Embedding model status
- Vectorstore compatibility

**Cost Monitoring**:
- Real-time cost calculations
- Per-model breakdown
- Historical trends
- Budget alerts (configurable)

## 🔧 Troubleshooting

### Common Issues

**Issue**: API key not found error
**Solution**: Ensure API key is set in `.env` file and restart application

**Issue**: Vectorstore compatibility warning
**Solution**: Check Admin Panel → Usage Stats → Embedding Status; regenerate if needed

**Issue**: File upload errors
**Solution**: Ensure files are under 50MB and in supported formats (PDF, TXT, DOCX, DOC)

**Issue**: Session timeout
**Solution**: Admin sessions expire after 24 hours; re-login required

**Issue**: Claude models not showing up
**Solution**: Install required packages: `pip install langchain-anthropic anthropic`

**Issue**: Ollama connection failed
**Solution**: Verify Ollama is running (`ollama serve`) or check `OLLAMA_BASE_URL` in `.env`

**Issue**: Model initialization failed
**Solution**: Check logs for specific error; system will fallback to OpenAI default

### Rebuilding Vectorstore

From command line:
```bash
python scripts/rebuild_vectordb.py
```

From web interface:
1. Go to Admin Panel → 📄 Documents
2. Click "Regenerate Vector Database"
3. Wait for processing to complete

## 🔄 CI/CD Pipeline

Automated testing runs on every push and pull request:

- **Multi-version testing**: Python 3.9, 3.10, 3.11
- **Linting**: flake8 code quality checks
- **Formatting**: black code style validation
- **Unit tests**: Full test suite with coverage reporting (90%+)
- **Docker tests**: Build validation and smoke tests
- **Security scans**: Bandit and Safety vulnerability checks

**Status**: ![Tests](https://github.com/<your-repo>/NegotiatorPro/workflows/test/badge.svg)

View the workflow: [.github/workflows/test.yml](.github/workflows/test.yml)

## 📝 Documentation

- **[README.md](README.md)** - This file (overview and quick start)
- **[CLAUDE.md](CLAUDE.md)** - Developer guide for Claude Code
- **[docs/TESTING.md](docs/TESTING.md)** - Comprehensive testing guide
- **[docs/deployment/DEPLOYMENT.md](docs/deployment/DEPLOYMENT.md)** - Production deployment (600+ lines)
- **[docs/deployment/DOCKER-DEPLOY.md](docs/deployment/DOCKER-DEPLOY.md)** - Docker quick start
- **[docs/features/ADMIN_FEATURES.md](docs/features/ADMIN_FEATURES.md)** - Admin panel guide
- **[docs/features/OLLAMA_CLOUD_SETUP.md](docs/features/OLLAMA_CLOUD_SETUP.md)** - Ollama cloud setup
- **[docs/features/QUICKSTART.md](docs/features/QUICKSTART.md)** - Quick start guide

## 🤝 Contributing

This system is designed to be extensible. Key areas for enhancement:

- Additional LLM backends (Google Gemini, Cohere, etc.)
- More document formats (CSV, JSON, etc.)
- Enhanced analytics and visualization
- Custom prompt templates and frameworks
- Integration with CRM systems
- Multi-language support
- Voice input/output capabilities

## 📜 License

See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with:
- **LangChain** - RAG framework and LLM orchestration
- **OpenAI** - GPT models and embeddings
- **Anthropic** - Claude models
- **Ollama** - Local and cloud LLM deployment
- **Gradio** - Web interface
- **FAISS** - Vector similarity search
- **Docker** - Containerization

Negotiation knowledge from:
- "Getting to Yes" by Roger Fisher and William Ury
- "Never Split the Difference" by Chris Voss
- And other expert negotiation resources

---

**Built with ❤️ for better negotiations**

Ready to negotiate like a pro? Get started in 5 minutes with Docker or dive into local development!
