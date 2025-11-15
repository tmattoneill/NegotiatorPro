# Database Migrations

This directory contains PostgreSQL database migration scripts for NegotiatorPro.

## Migration Files

- `001_initial_schema.sql` - Initial database schema with users, sessions, configuration, and chat history

## Running Migrations

### Using psql

```bash
# Connect to your PostgreSQL database
psql -U your_username -d negotiatorpro

# Run a migration
\i migrations/001_initial_schema.sql
```

### Using Docker Compose

```bash
# Copy SQL file to running Postgres container
docker cp migrations/001_initial_schema.sql postgres_container:/001_initial_schema.sql

# Execute migration
docker exec -it postgres_container psql -U negotiatorpro -d negotiatorpro -f /001_initial_schema.sql
```

## Creating New Migrations

When creating new migrations:

1. Use sequential numbering: `002_description.sql`, `003_description.sql`, etc.
2. Include rollback logic where appropriate
3. Test migrations on a development database first
4. Document schema changes in comments

## Schema Overview

The database includes the following main tables:

- **users** - User accounts and authentication
- **sessions** - Session management with expiration
- **system_config** - System-wide configuration
- **llm_config** - LLM backend and model settings
- **usage_logs** - API usage statistics and tracking
- **documents** - Uploaded source documents
- **prompts** - Prompt templates with versioning
- **chat_messages** - Chat conversation history
- **embedding_config** - Vector embedding configuration

## Environment Setup

Create a `.env` file with PostgreSQL connection details:

```bash
# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=negotiatorpro
POSTGRES_USER=negotiatorpro
POSTGRES_PASSWORD=your_secure_password
```

## Future: Migration Tool

Consider using a migration tool like:
- **Alembic** (Python)
- **Flyway** (Java/Docker)
- **migrate** (Go)
- **node-pg-migrate** (Node.js)

For now, migrations are manual SQL scripts.
