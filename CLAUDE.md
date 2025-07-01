# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an enhanced RAG (Retrieval-Augmented Generation) system that provides expert negotiation guidance by analyzing PDF sources of negotiation books. The system features a comprehensive Gradio web interface with admin capabilities, intelligent model selection between gpt-4o-mini (default) and o3-mini (premium), and robust document management.

## Core Architecture

**Main Application**: `main.py` contains the complete RAG pipeline with integrated admin interface:

- **ModelConfig**: Middleware class that handles model-specific parameters (e.g., o3-mini doesn't support temperature)
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

**Model Architecture**:
- Separate QA chains for each model (default_qa_chain, premium_qa_chain)
- ModelConfig.get_model_kwargs() filters parameters based on model capabilities
- Intelligent switching between gpt-4o-mini and o3-mini based on user selection
- EmbeddingConfig ensures compatibility between embedding models and vectorstore

## Development Commands

**Environment Setup**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Required Environment Variables**:
Create `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

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

## Key Implementation Details

**PLEASE Framework**: The system prompt implements a structured response format requiring:
- Detailed negotiation breakdown
- Calibrated questions
- Draft responses
- Scenario planning
- Self-assessment scoring (Polite, Logical, Empathetic, Assertive, Strategic, Engaging)

**Admin Features**:
- **System Prompt Management**: Customize the AI's behavior and response format
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

- `main.py` - Main application with enhanced RAG system, dual Gradio UI, and integrated admin panel
- `admin_config.py` - Admin authentication, sessions, prompts, and usage tracking
- `document_manager.py` - File upload handling, validation, and document management
- `embedding_config.py` - Embedding model configuration and vectorstore compatibility
- `sources/` - Source documents for RAG knowledge base (PDF, TXT, DOCX, DOC)
- `uploads/` - Temporary storage for uploaded files
- `vectorstore/` - Persisted FAISS embeddings with metadata (auto-generated)
- `utils/` - Utility scripts for vectorstore rebuilding
- `requirements.txt` - Python dependencies
- `.env` - OpenAI API key (create manually)
- `test_rag.py` - Testing utilities for document loading and embeddings
- `admin_config.json` - Admin configuration and settings (auto-generated)
- `admin_sessions.json` - Active admin sessions (auto-generated)
- `usage_stats.json` - API usage statistics (auto-generated)
- `embedding_config.json` - Embedding model configuration (auto-generated)