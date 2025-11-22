# Database Migrations

This directory contains the PostgreSQL database schema for NegotiatorPro.

## Migration File

- `001_full_schema.sql` - Complete production database schema

## Schema Overview

The database includes the following tables:

**Core User Management:**
- **users** - User accounts with profile and encrypted API key storage
- **sessions** - Session management with expiration tracking

**System Configuration:**
- **system_config** - System-wide configuration key-value store
- **llm_config** - LLM backend and model configurations
- **embedding_config** - Vector embedding model configuration
- **prompts** - Prompt templates with versioning

**Negotiation System:**
- **user_personas** - User's negotiation identities/roles
- **partner_personas** - Negotiation counterpart profiles (shareable)
- **negotiations** - Core negotiation tracking with status
- **negotiation_partners** - Links negotiations to partner personas
- **conversations** - Chat sessions within negotiations
- **chat_messages** - Conversation history

**Document & Usage:**
- **documents** - Uploaded source documents for RAG
- **negotiation_documents** - Documents attached to negotiations
- **usage_logs** - API usage statistics and token tracking

## Running Migrations

### Docker (Automatic)

Migrations run automatically when the PostgreSQL container starts for the first time.
The `migrations/` directory is mounted to `/docker-entrypoint-initdb.d/`.

```bash
docker compose up -d
```

### Manual Execution

```bash
# Using psql directly
psql -U negotiatorpro -d negotiatorpro -f migrations/001_full_schema.sql

# Using Docker
docker exec -i negotiator-pro-postgres psql -U negotiatorpro -d negotiatorpro < migrations/001_full_schema.sql
```

## Fresh Database Setup

For a completely fresh database:

```bash
# Stop containers and remove database volume
docker compose down -v

# Remove local data
rm -rf data/db

# Start fresh
docker compose up -d
```

## Environment Setup

Required environment variables in `.env`:

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=negotiatorpro
POSTGRES_USER=negotiatorpro
POSTGRES_PASSWORD=your_secure_password
```

## Default Users

The migration creates a default admin user:
- **Username:** admin
- **Password:** admin123
- **IMPORTANT:** Change this password immediately after first login!
