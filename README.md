# 🤝 NegotiatorPro

An AI-powered negotiation advisor that leverages expert knowledge from leading negotiation books to provide personalized guidance. Built with advanced RAG (Retrieval-Augmented Generation) technology and intelligent text preprocessing for optimal performance.

## ✨ Features

### 🎯 **Core Functionality**
- **Expert Negotiation Guidance**: Draws from "Getting to Yes", "Never Split the Difference", and other proven negotiation frameworks
- **Dual AI Models**: Choose between gpt-4o-mini (fast, cost-effective) and o3-mini (advanced reasoning)
- **Smart Text Optimization**: Reduces token usage by up to 68% while preserving negotiation context
- **Comprehensive Analysis**: Provides detailed breakdowns, draft responses, calibrated questions, and scenario planning

### 🛠️ **Advanced Features**
- **Intelligent Preprocessing**: Removes email signatures, footers, legal disclaimers, and fluff
- **Context-Aware**: Preserves critical negotiation elements (emotions, numbers, commitments, deadlines)
- **Real-time Statistics**: Track token usage, cost savings, and optimization results
- **Admin Dashboard**: Full system management and prompt customization

### 📊 **Text Optimization Engine**
- **Email Content Cleaning**: Removes signatures, footers, forwarding headers
- **Legal Boilerplate Removal**: Strips confidentiality notices and disclaimers
- **Smart Stop Word Removal**: Context-aware filtering that preserves meaning
- **Negotiation-Critical Preservation**: Never removes prices, emotions, deadlines, or commitments

## 🚀 Quick Start

### Prerequisites

- **Docker Deployment** (Recommended): Docker and Docker Compose
- **Local Development**: Python 3.8+ and OpenAI API key

### Option 1: Docker Deployment (Recommended) 🐳

Perfect for production deployments on Ubuntu/Linux systems.

1. **Clone and configure**
   ```bash
   git clone <repository-url>
   cd NegotiatorPro
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

2. **Start with Docker**
   ```bash
   docker compose up -d
   ```

3. **Access the application**
   - Open your browser to `http://localhost:7860`
   - Admin Panel (default password: `admin123`)

📖 **See [DOCKER-DEPLOY.md](DOCKER-DEPLOY.md) for the quick start guide**
📖 **See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment**

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
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

5. **Run the application**
   ```bash
   python main.py
   # Or use the startup script:
   ./run.sh
   ```

6. **Access the interface**
   - Open your browser to the URL shown in the terminal (typically `http://localhost:7860`)
   - Use the "Negotiation Advisor" tab for asking questions
   - Use the "Admin Panel" tab for administration (default password: `admin123`)

## 💡 How to Use

### For Users

1. **Ask Negotiation Questions**: Enter your specific negotiation scenario or question
2. **Provide Context** (Optional): Add information about your negotiation partner
3. **Choose Model**: Select between gpt-4o-mini (default) or o3-mini (premium)
4. **Get Expert Advice**: Receive structured guidance following the PLEASE framework

### Example Questions

- "How do I respond to 'That's my final offer'?"
- "What's the best way to make the first offer?"
- "How can I build rapport with a difficult negotiator?"
- "They're using high-pressure tactics. What should I do?"

### For Administrators

Access the Admin Panel to:

- **Manage System Prompts**: Customize how the AI responds
- **Upload Documents**: Add new negotiation resources to the knowledge base
- **Monitor Usage**: Track API costs and usage patterns
- **Manage Embeddings**: Ensure vectorstore compatibility
- **Change Settings**: Update admin password and configurations

## 🏗️ System Architecture

### Core Components

- **EnhancedNegotiationRAG**: Main RAG system with admin integration
- **ModelConfig**: Handles model-specific parameters and compatibility
- **AdminConfig**: Manages authentication, sessions, and system settings
- **DocumentManager**: Handles file uploads and document processing
- **EmbeddingConfig**: Manages embedding models and vectorstore compatibility

### Data Flow

1. **Document Processing**: Files are loaded, chunked, and converted to embeddings
2. **Vector Storage**: FAISS vectorstore persists embeddings with metadata
3. **Query Processing**: User questions are enhanced with retrieved context
4. **AI Response**: Model generates structured negotiation advice
5. **Usage Tracking**: Admin system logs usage statistics

## 📁 File Structure

```
├── main.py                    # Main application entry point
├── admin_config.py           # Admin authentication and configuration
├── document_manager.py       # File upload and document management
├── embedding_config.py       # Embedding model configuration
├── text_preprocessor.py      # Text optimization engine
├── prompt_manager.py         # Prompt template management
├── test_rag.py              # Testing utilities
├── requirements.txt         # Python dependencies
├── requirements-test.txt    # Test dependencies
├── run.sh                   # Startup script
├── .env                     # Environment variables (create manually)
├── .env.example             # Environment template
├── Dockerfile               # Docker image definition
├── docker-compose.yml       # Docker orchestration
├── .dockerignore           # Docker build exclusions
├── pytest.ini              # Test configuration
├── .coveragerc             # Coverage configuration
├── README.md               # This file
├── TESTING.md              # Testing guide
├── DEPLOYMENT.md           # Production deployment guide
├── DOCKER-DEPLOY.md        # Docker quick start
├── sources/                # Source documents directory
│   └── *.pdf, *.docx, etc. # Negotiation books and resources
├── uploads/                # Temporary upload storage
├── vectorstore/            # Generated FAISS embeddings
├── tests/                  # Test suite (100+ tests)
│   ├── test_docker.py      # Docker infrastructure tests
│   ├── test_admin_config.py # Admin system tests
│   ├── test_document_manager.py # Document tests
│   ├── test_model_config.py # Model configuration tests
│   ├── test_modules.py     # Supporting module tests
│   ├── test_integration.py # Integration tests
│   └── conftest.py         # Test fixtures
├── .github/
│   └── workflows/
│       └── test.yml        # CI/CD pipeline
├── utils/                  # Utility scripts
└── Auto-generated files:
    ├── admin_config.json    # Admin settings
    ├── admin_sessions.json  # Active sessions
    ├── usage_stats.json     # Usage statistics
    ├── embedding_config.json # Embedding configuration
    └── prompt_config.json   # Prompt templates
```

## ⚙️ Configuration Options

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Admin Configuration

Access via Admin Panel → Admin Settings:

- **Admin Password**: Change from default `admin123`
- **Session Duration**: Configure session timeout
- **Upload Limits**: Set maximum file size for uploads

### Embedding Models

The system supports multiple OpenAI embedding models:

- `text-embedding-3-large` (default): Highest quality, 3072 dimensions
- `text-embedding-3-small`: Good balance, 1536 dimensions  
- `text-embedding-ada-002`: Legacy model, 1536 dimensions

## 🔒 Security Features

- **Session-based Authentication**: Secure admin access with configurable timeouts
- **Password Protection**: Hashed password storage
- **File Validation**: Secure file upload with type and size validation
- **Session Management**: Automatic cleanup of expired sessions

## 📊 Usage Analytics

The admin panel provides detailed analytics:

- **Daily Usage**: Track requests, tokens, and costs by day
- **Model Breakdown**: Compare usage between different AI models
- **Cost Monitoring**: Monitor OpenAI API expenses
- **Performance Metrics**: Track system usage patterns

## 🛠️ Advanced Usage

### Custom System Prompts

Administrators can customize the AI's behavior by modifying the system prompt in the Admin Panel. The default implements the PLEASE framework:

- **P**olite: Maintain professional tone
- **L**ogical: Provide structured reasoning
- **E**mpathetic: Understand all parties' positions
- **A**ssertive: Advocate for favorable outcomes
- **S**trategic: Think several moves ahead
- **E**ngaging: Keep interactions productive

### Document Management

- **Supported Formats**: PDF, DOCX, TXT, DOC
- **Auto-Processing**: Documents are automatically chunked and embedded
- **Metadata Tracking**: System tracks document information and processing status
- **Regeneration**: Vectorstore can be rebuilt when document library changes

### Model Selection

- **gpt-4o-mini**: Fast responses, cost-effective, supports temperature control
- **o3-mini**: Advanced reasoning capabilities, higher cost, no temperature control

## 🔧 Troubleshooting

### Common Issues

1. **Vectorstore Compatibility**: If embeddings seem inconsistent, check Admin Panel → Usage Stats → Embedding Status
2. **File Upload Errors**: Ensure files are under 50MB and in supported formats
3. **API Errors**: Verify your OpenAI API key in the `.env` file
4. **Session Timeout**: Admin sessions expire after 24 hours by default

### Rebuilding Vectorstore

If you need to rebuild the vectorstore:
1. Go to Admin Panel → Documents
2. Click "Regenerate Vector Database"
3. Wait for processing to complete

## 🧪 Testing

NegotiatorPro includes a comprehensive test suite with 100+ tests covering all components.

### Running Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m docker        # Docker tests only
```

### Test Coverage

- **Docker Infrastructure**: Dockerfile, docker-compose, deployment validation
- **Unit Tests**: Admin, documents, models, text preprocessing, prompts
- **Integration Tests**: End-to-end RAG pipeline, model switching
- **Security**: Automated security scanning with Bandit and Safety
- **CI/CD**: Automated testing on GitHub Actions

📖 **See [TESTING.md](TESTING.md) for complete testing guide**

### Legacy Testing

Run the legacy test script to verify document loading:
```bash
python test_rag.py
```

## 🐳 Docker Deployment

### Features

- **Multi-stage build**: Optimized image size
- **Non-root user**: Security best practices
- **Health checks**: Container monitoring
- **Persistent volumes**: Data survives restarts
- **Resource limits**: Configurable CPU/memory
- **Auto-restart**: Production reliability

### Quick Commands

```bash
# Start
docker compose up -d

# Stop
docker compose stop

# Rebuild
docker compose build --no-cache

# View logs
docker compose logs -f

# Check status
docker compose ps
```

📖 **See [DOCKER-DEPLOY.md](DOCKER-DEPLOY.md) for quick start**
📖 **See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment**

## 🔄 CI/CD Pipeline

Automated testing runs on every push and pull request:

- **Multi-version testing**: Python 3.9, 3.10, 3.11
- **Linting**: flake8 code quality checks
- **Formatting**: black code style validation
- **Unit tests**: Full test suite with coverage reporting
- **Docker tests**: Build validation and smoke tests
- **Security scans**: Bandit and Safety vulnerability checks

View the workflow: [.github/workflows/test.yml](.github/workflows/test.yml)

## 📝 Documentation

- **[README.md](README.md)** - This file (overview and quick start)
- **[TESTING.md](TESTING.md)** - Comprehensive testing guide
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment guide (600+ lines)
- **[DOCKER-DEPLOY.md](DOCKER-DEPLOY.md)** - Docker quick start (5-minute setup)
- **[CLAUDE.md](CLAUDE.md)** - Developer guide for Claude Code

## 📝 Support

- Check the logs in the terminal for detailed error information
- Verify your `.env` file contains a valid OpenAI API key
- Ensure all dependencies are installed correctly
- For document processing issues, check Admin Panel → Documents
- For Docker issues, see [DEPLOYMENT.md](DEPLOYMENT.md) troubleshooting section

## 🤝 Contributing

This system is designed to be extensible. Key areas for enhancement:

- Additional document formats
- New embedding models
- Enhanced analytics
- Custom prompt templates
- Integration with other AI providers
- Additional deployment options

## 📜 License

See [LICENSE](LICENSE) file for details.

---

Built with ❤️ using LangChain, OpenAI, Gradio, FAISS, and Docker.