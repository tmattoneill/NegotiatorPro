# NegotiatorPro

An AI-powered negotiation advisor that uses RAG (Retrieval-Augmented Generation) to provide expert guidance based on negotiation literature. Features a React frontend, FastAPI backend, and multi-LLM support (OpenAI, Anthropic Claude, Ollama).

## Prerequisites

- **Docker** and **Docker Compose** (v2+)
- At least one LLM API key (OpenAI, Anthropic, or local Ollama)

Install Docker:
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker compose version
```

## Quick Start

### 1. Run the Setup Script

The easiest way to get started is using the interactive setup script:

```bash
./scripts/setup.sh
```

This will:
- Check Docker prerequisites
- Prompt for environment configuration (development/production)
- Generate secure passwords and keys
- Configure LLM API keys (OpenAI, Anthropic, Ollama)
- Create the `.env` file
- Create required data directories
- Build and start all containers

### 2. Access the Application

Once started, access:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Manual Setup

If you prefer manual configuration:

### 1. Create Environment File

Copy the example and edit:
```bash
cp .env.example .env
```

Required variables:
```bash
# At least one LLM provider
OPENAI_API_KEY=sk-your-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Database (required)
POSTGRES_PASSWORD=your-secure-password

# Security (auto-generated if using setup.sh)
JWT_SECRET_KEY=your-jwt-secret
ENCRYPTION_KEY=your-encryption-key
```

Generate security keys:
```bash
# JWT secret
openssl rand -hex 32

# Encryption key
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### 2. Create Data Directories

```bash
mkdir -p data/db data/vectorstore data/uploads data/sources data/config
```

### 3. Add Source Documents

Place your negotiation books (PDF, TXT, DOCX) in the `data/sources/` directory:
```bash
cp /path/to/your/books/*.pdf data/sources/
```

### 4. Build and Start

```bash
docker compose up -d --build
```

## Adding New Sources & Rebuilding the RAG

When you add new negotiation books or documents, you need to rebuild the vector database.

### 1. Add Documents

Place new documents in the sources directory:
```bash
cp /path/to/new-book.pdf data/sources/
```

Supported formats: PDF, TXT, DOCX

### 2. Rebuild the Vector Database

Run the rebuild script inside the backend container:

```bash
docker exec -it negotiator-pro-backend python scripts/rebuild_vectordb.py
```

This interactive script will:
1. Scan all documents in `data/sources/`
2. Let you select an embedding model
3. Show cost estimation
4. Ask for confirmation before rebuilding
5. Create backups of existing vectorstore
6. Generate new embeddings and save the vectorstore

### 3. Restart to Apply Changes

```bash
docker compose restart backend
```

### Non-Interactive Rebuild (Advanced)

For automated rebuilds, you can modify the script or use:
```bash
docker exec -it negotiator-pro-backend python -c "
from scripts.rebuild_vectordb import VectorDBRebuilder
rebuilder = VectorDBRebuilder(sources_dir='/app/sources', vectorstore_dir='/app/vectorstore')
# Custom automation here
"
```

## Docker Commands Reference

### Start/Stop Services

```bash
# Start all services
docker compose up -d

# Start with rebuild
docker compose up -d --build

# Stop all services
docker compose down

# Restart specific service
docker compose restart backend
docker compose restart frontend
```

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f postgres
```

### Execute Commands in Containers

```bash
# Backend shell
docker exec -it negotiator-pro-backend bash

# Run Python script
docker exec -it negotiator-pro-backend python scripts/rebuild_vectordb.py

# Initialize user profiles
docker exec -it negotiator-pro-backend python scripts/init_user_profile.py
```

### Database Operations

```bash
# Access PostgreSQL
docker exec -it negotiator-pro-postgres psql -U negotiatorpro -d negotiatorpro

# Backup database
docker exec negotiator-pro-postgres pg_dump -U negotiatorpro negotiatorpro > backup.sql

# Restore database
docker exec -i negotiator-pro-postgres psql -U negotiatorpro negotiatorpro < backup.sql
```

### Cleanup

```bash
# Stop and remove containers
docker compose down

# Remove containers and volumes (WARNING: deletes data)
docker compose down -v

# Remove unused images
docker image prune -f
```

## Data Persistence

All persistent data is stored in the `data/` directory:

| Directory | Purpose |
|-----------|---------|
| `data/db/` | PostgreSQL database files |
| `data/vectorstore/` | FAISS vector embeddings |
| `data/uploads/` | User uploaded documents |
| `data/sources/` | Source negotiation books |
| `data/config/` | Configuration files |

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `POSTGRES_PASSWORD` | PostgreSQL password |
| `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` | At least one LLM provider |

### Security (Required for Production)

| Variable | Description |
|----------|-------------|
| `JWT_SECRET_KEY` | JWT signing secret |
| `ENCRYPTION_KEY` | API key encryption |
| `ADMIN_PASSWORD` | Admin user password |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | development | Environment mode |
| `POSTGRES_DB` | negotiatorpro | Database name |
| `POSTGRES_USER` | negotiatorpro | Database user |
| `OLLAMA_BASE_URL` | http://host.docker.internal:11434 | Local Ollama URL |
| `CORS_ALLOWED_ORIGINS` | localhost:5173,3000 | Allowed CORS origins |

## LLM Configuration

NegotiatorPro supports multiple LLM backends:

### OpenAI
- Models: GPT-4o, GPT-4o Mini, O3 Mini, GPT-4 Turbo
- Set `OPENAI_API_KEY` in `.env`

### Anthropic Claude
- Models: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- Set `ANTHROPIC_API_KEY` in `.env`

### Ollama (Local)
- Models: Llama 3.1, Mistral, Mixtral, Qwen 2.5
- Install Ollama on host: https://ollama.com
- Default URL: `http://host.docker.internal:11434`

Configure active models via the Admin Panel in the web UI.

## Troubleshooting

### Container won't start

Check logs:
```bash
docker compose logs backend
```

Common issues:
- Missing `POSTGRES_PASSWORD` in `.env`
- Port 8000 or 5173 already in use
- Insufficient memory (requires 2GB+)

### Vector database errors

Rebuild the vectorstore:
```bash
docker exec -it negotiator-pro-backend python scripts/rebuild_vectordb.py
```

### Database connection issues

Ensure PostgreSQL is healthy:
```bash
docker compose ps
docker exec -it negotiator-pro-postgres pg_isready
```

### Ollama not connecting

For local Ollama on the host machine:
```bash
# Ensure Ollama is running
ollama serve

# Test connectivity from container
docker exec -it negotiator-pro-backend curl http://host.docker.internal:11434/api/version
```

## Development

For development with hot reload, the docker-compose mounts local directories:
- `./backend` -> `/app/backend`
- `./frontend` -> `/app/frontend`

Changes to backend Python files will auto-reload. Frontend changes require the dev server to detect them.

## License

See LICENSE file for details.
