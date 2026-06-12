# NegotiatorPro

An AI-powered negotiation advisor that uses RAG (Retrieval-Augmented Generation) to provide expert guidance based on negotiation literature. Features a React frontend, FastAPI backend, and multi-LLM support (OpenAI, Anthropic Claude, Ollama, DeepSeek). The database is PostgreSQL on Neon (external); the Docker stack runs the backend and frontend only.

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

# Database (required): a PostgreSQL connection string. In production and the dev
# deploy this is Neon. The POSTGRES_* individual vars are an optional fallback
# for a local Postgres and are ignored when DATABASE_URL is set.
DATABASE_URL=postgresql://user:pass@host/dbname

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
mkdir -p data/vectorstore data/uploads data/sources data/config
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

### 5. Build the Vectorstore (first run)

The FAISS index is a build artifact and is **not** stored in the repo, so a
fresh checkout or deploy starts with no index. Build it once from your sources
before the RAG can answer anything:

- Preferred: open the admin **📚 Sources & RAG** tab and click **Rebuild
  Vectorstore** (see below).
- Or from the CLI: `docker compose exec backend python scripts/rebuild_vectordb.py`

After the first build the index persists in the `data/vectorstore/` volume, so
this step is only needed on a fresh environment or when you change the corpus.

## Adding New Sources & Rebuilding the RAG

The preferred way is through the admin panel — no `docker exec` needed.

### Via the Admin Panel (recommended)

1. Log in as admin and open the **📚 Sources & RAG** tab
2. Drag-and-drop or browse to upload PDF, TXT, or DOCX files
3. Edit tags (`sales` / `negotiation`) and other metadata per source
4. Click **Rebuild Vectorstore** — set chunk size, overlap, embedding model, and optional tag filter
5. Watch the progress bar; use **Test Query** to verify results before the new index goes live
6. The rebuild swaps atomically — the live index is only replaced after a successful build

Uploads are deduplicated by SHA-256 hash, so re-uploading the same file is a no-op.

### One-Time Migration (first run with expanded corpus)

To populate `sources/` with the full sales + negotiation corpus and backfill metadata:

```bash
# Preview what would happen
docker compose exec backend python scripts/migrate_expanded_corpus.py --dry-run

# Apply
docker compose exec backend python scripts/migrate_expanded_corpus.py
```

Then trigger a rebuild from the admin panel.

### Via CLI (advanced / scripting)

```bash
# Apply a DB migration (first time only). The DB is Neon; migrations are run
# against it directly, not from a local container. See docs/deployment/DEPLOY.md.
psql "$DATABASE_URL" -f migrations/004_source_documents.sql

# Programmatic rebuild (skips interactive prompts)
docker compose exec backend python scripts/rebuild_vectordb.py
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
```

### Execute Commands in Containers

```bash
# Backend shell
docker compose exec backend bash

# Run Python script
docker compose exec backend python scripts/rebuild_vectordb.py

# Initialize user profiles
docker compose exec backend python scripts/init_user_profile.py
```

### Database Operations

The database is Neon (external), so connect with `DATABASE_URL` from `.env`, not
through a container. Neon provides backups and point-in-time restore from its
dashboard.

```bash
# Access the database
psql "$DATABASE_URL"

# Ad-hoc dump (Neon also has managed backups / PITR in the dashboard)
pg_dump "$DATABASE_URL" > backup.sql
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
| `data/vectorstore/` | FAISS vector embeddings |
| `data/uploads/` | User uploaded documents |
| `data/sources/` | Source negotiation books |
| `data/config/` | Configuration files |

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string (Neon in prod/dev; or a local Postgres) |
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

## Adaptive Response System

NegotiatorPro classifies each query by intent and selects a response format to match. A live tactical question gets a decisive move and word-for-word language; a conceptual question gets a principle with a cited source; a pasted email gets a full structured analysis. The format is shown as a badge on each message.

| Mode | Triggered by | Response shape |
|------|-------------|----------------|
| **Tactical** | "they countered", "should I accept", real-time situations | Move + exact language + one contingency |
| **Analysis** | Pasted transcript/email, "analyse this" | Full structured breakdown with scenario planning |
| **Q&A** | "what is X", "when should I", conceptual questions | Principle + why it matters + one applied step |
| **Advisory** | Open-ended, exploratory | Direct teaching, no rigid sections |

Classification is pure regex — no extra LLM call. Format instructions are injected per-request into the non-cached context block, so the static system prompt stays byte-identical for prompt caching across all four modes.

See `docs/features/INTENT_AWARE_PROMPTING.md` for the full architecture and extension guide.

## Troubleshooting

### Container won't start

Check logs:
```bash
docker compose logs backend
```

Common issues:
- Missing or wrong `DATABASE_URL` in `.env`
- Port 8000 or 5173 already in use
- Insufficient memory (requires 2GB+)

### Vector database errors

Rebuild the vectorstore:
```bash
docker compose exec backend python scripts/rebuild_vectordb.py
```

### Database connection issues

The DB is Neon (external). Check it is reachable:
```bash
psql "$DATABASE_URL" -c "select 1"
```
Confirm `DATABASE_URL` is set and the Neon branch is not paused (free tier auto-suspends).

### Ollama not connecting

For local Ollama on the host machine:
```bash
# Ensure Ollama is running
ollama serve

# Test connectivity from container
docker compose exec backend curl http://host.docker.internal:11434/api/version
```

## Development

For development with hot reload, the docker-compose mounts local directories:
- `./backend` -> `/app/backend`
- `./frontend` -> `/app/frontend`

Changes to backend Python files will auto-reload. Frontend changes require the dev server to detect them.

## License

See LICENSE file for details.
