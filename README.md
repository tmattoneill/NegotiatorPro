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

- Python 3.8+
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-sources
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
├── test_rag.py              # Testing utilities
├── requirements.txt         # Python dependencies
├── run.sh                   # Startup script
├── .env                     # Environment variables (create manually)
├── sources/                 # Source documents directory
│   └── *.pdf, *.docx, etc. # Negotiation books and resources
├── uploads/                 # Temporary upload storage
├── vectorstore/             # Generated FAISS embeddings
├── utils/                   # Utility scripts
└── Auto-generated files:
    ├── admin_config.json    # Admin settings
    ├── admin_sessions.json  # Active sessions
    ├── usage_stats.json     # Usage statistics
    └── embedding_config.json # Embedding configuration
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

### Testing

Run the test script to verify document loading:
```bash
python test_rag.py
```

## 📝 Support

- Check the logs in the terminal for detailed error information
- Verify your `.env` file contains a valid OpenAI API key
- Ensure all dependencies are installed correctly
- For document processing issues, check Admin Panel → Documents

## 🤝 Contributing

This system is designed to be extensible. Key areas for enhancement:

- Additional document formats
- New embedding models
- Enhanced analytics
- Custom prompt templates
- Integration with other AI providers

---

Built with ❤️ using LangChain, OpenAI, Gradio, and FAISS.