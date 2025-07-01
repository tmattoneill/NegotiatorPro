# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) system that provides expert negotiation guidance by analyzing PDF sources of negotiation books. The system uses a Gradio web interface and leverages OpenAI models with intelligent model selection between gpt-4o-mini (default) and o3-mini (premium).

## Core Architecture

**Main Application**: `main.py` contains the complete RAG pipeline with admin interface:

- **ModelConfig**: Middleware class that handles model-specific parameters (e.g., o3-mini doesn't support temperature)
- **NegotiationRAG**: Core RAG system that processes PDFs, creates embeddings, and manages QA chains
- **Gradio Interface**: Web UI with model selection, partner context input, and example questions

**Document Processing Flow**:
1. PDFs in `sources/` directory are loaded via PyPDFLoader
2. Documents are chunked using RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
3. FAISS vectorstore is created with OpenAI embeddings
4. Vectorstore is persisted to `vectorstore/` directory for reuse

**Model Architecture**:
- Separate QA chains for each model (default_qa_chain, premium_qa_chain)
- ModelConfig.get_model_kwargs() filters parameters based on model capabilities
- Intelligent switching between gpt-4o-mini and o3-mini based on user selection

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

## Key Implementation Details

**PLEASE Framework**: The system prompt implements a structured response format requiring:
- Detailed negotiation breakdown
- Calibrated questions
- Draft responses
- Scenario planning
- Self-assessment scoring (Polite, Logical, Empathetic, Assertive, Strategic, Engaging)

**Vectorstore Persistence**: The system checks for existing vectorstore on startup to avoid reprocessing PDFs. Delete `vectorstore/` directory to force rebuild.

**Model Selection**: Users can toggle between models via checkbox. The ModelConfig class ensures proper parameter handling for each model type.

**PDF Sources**: Place negotiation books in `sources/` directory. Current sources include "Getting to Yes", "Never Split the Difference", etc.

## File Structure

- `main.py` - Main application with RAG system, Gradio UI, and admin panel
- `sources/` - PDF documents for RAG knowledge base
- `vectorstore/` - Persisted FAISS embeddings (auto-generated)
- `requirements.txt` - Python dependencies
- `.env` - OpenAI API key (create manually)
- `test_rag.py` - Testing utilities for PDF loading and embeddings