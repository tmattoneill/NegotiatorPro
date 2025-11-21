# User Profile System - Docker Architecture Summary

## What Changed

I've refactored the entire system to be **Docker-first** with all persistent data in a `/data` mount. This is a cleaner, production-ready architecture that avoids hybrid host/Docker setups.

## Architecture Overview

### Before (Hybrid)
```
❌ PostgreSQL on host
❌ Python env on host
❌ Config files scattered
❌ Manual setup needed
```

### After (Docker-First)
```
✅ All services in Docker
✅ All data in /data mount
✅ No host dependencies
✅ One command to start
```

## Directory Structure

```
NegotiatorPro/
├── docker-compose.yml       # Orchestrates: postgres, backend, frontend
├── Dockerfile               # Backend container
├── .env                     # Environment variables
│
├── data/                    # ALL PERSISTENT DATA (gitignored)
│   ├── db/                 # PostgreSQL database files
│   ├── vectorstore/        # FAISS embeddings
│   ├── sources/            # Source documents (PDFs, DOCX)
│   ├── uploads/            # Temporary uploads
│   ├── config/             # JSON config files
│   └── backups/            # Backup storage
│
├── backend/                 # Python code (mounted for hot reload)
│   ├── database.py         # PostgreSQL connection (asyncpg)
│   ├── user_profile.py     # User CRUD + encryption
│   ├── config_paths.py     # Docker-aware path management
│   └── api/routes/users.py # User REST endpoints
│
├── migrations/              # Database migrations (auto-run on startup)
│   ├── 001_initial_schema.sql
│   └── 002_add_user_profile_fields.sql
│
└── scripts/
    └── init_user_profile.py # Initialize default users (runs in container)
```

## What Runs Where

| Component | Where | How |
|-----------|-------|-----|
| PostgreSQL 15 | Docker container | `postgres:15-alpine` image |
| Python/FastAPI | Docker container | Custom Dockerfile |
| Node/React | Docker container | `node:20-alpine` image |
| Database data | `./data/db` mount | Persists across restarts |
| Vectorstore | `./data/vectorstore` mount | Persists across restarts |
| Config files | `./data/config` mount | Persists across restarts |

## Key Files Created

### 1. Database Layer
- **`backend/database.py`** - Async PostgreSQL connection pooling
  - Uses asyncpg for high performance
  - Connection pool management
  - Health checks

### 2. User Profile System
- **`backend/user_profile.py`** - User management
  - UserProfileCreate/Update/Response models
  - EncryptionManager for API keys (Fernet)
  - UserProfileManager for CRUD operations

### 3. API Routes
- **`backend/api/routes/users.py`** - REST endpoints
  - POST `/api/users/` - Create user
  - GET `/api/users/{id}` - Get user
  - PATCH `/api/users/{id}` - Update user
  - DELETE `/api/users/{id}` - Delete user
  - GET `/api/users/{id}/api-keys` - Get encrypted keys

### 4. Docker Configuration
- **`docker-compose.yml`** - Updated with:
  - PostgreSQL service with health checks
  - Backend depends on healthy postgres
  - All data in `/data` mounts
  - Auto-migrations on first startup

### 5. Database Migrations
- **`migrations/001_initial_schema.sql`** - Complete schema
  - Users, sessions, LLM config, documents, chat history, etc.
- **`migrations/002_add_user_profile_fields.sql`** - User profile extensions
  - first_name, last_name, API keys, profile timestamps

### 6. Utilities
- **`backend/config_paths.py`** - Docker-aware paths
  - Detects if running in Docker
  - Maps to `/app/config` in container
  - Falls back to local paths for development

- **`scripts/init_user_profile.py`** - Initialize users
  - Waits for PostgreSQL to be ready
  - Creates default admin user
  - Creates test user
  - Runs inside container

### 7. Documentation
- **`DOCKER_SETUP.md`** - Complete Docker guide
- **`DOCKER_QUICK_START.md`** - 5-minute quick start
- **`data/README.md`** - Data directory documentation
- **`USER_PROFILE_DOCKER_SUMMARY.md`** - This file

## User Profile Features

### Stored Fields
- **Username** (unique) - Login identifier
- **Email** (unique) - User email address
- **First/Last Name** - Optional display name
- **OpenAI API Key** - Optional, encrypted with Fernet
- **Anthropic API Key** - Optional, encrypted with Fernet
- **Password** - Hashed with bcrypt
- **Role** - admin/user/viewer
- **Timestamps** - created_at, last_login, profile_updated_at

### Security
- **Password Hashing**: Bcrypt with automatic salting
- **API Key Encryption**: Fernet symmetric encryption
- **SQL Injection**: Prevented with asyncpg parameterized queries
- **Environment Secrets**: Encryption key from .env

## Quick Start Commands

### Initial Setup
```bash
# 1. Configure
cp .env.example .env
# Edit .env with API keys and database password

# 2. Start services
docker compose up -d

# 3. Initialize users
docker exec -it negotiator-pro-backend python scripts/init_user_profile.py
```

### Daily Usage
```bash
# View logs
docker compose logs -f backend

# Restart after code changes
docker compose restart backend

# Access database
docker exec -it negotiator-pro-postgres psql -U negotiatorpro -d negotiatorpro

# Backup data
docker exec negotiator-pro-postgres pg_dump -U negotiatorpro negotiatorpro > backup.sql

# Stop everything
docker compose down
```

## Environment Variables

Required in `.env`:

```bash
# LLM API Keys
OPENAI_API_KEY=sk-...

# Database (defaults shown)
POSTGRES_DB=negotiatorpro
POSTGRES_USER=negotiatorpro
POSTGRES_PASSWORD=changeme

# Encryption Key
# Generate: docker run --rm python:3.11 python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
ENCRYPTION_KEY=...
```

## Default Users

Created by `init_user_profile.py`:

1. **Admin User**
   - Username: `admin`
   - Email: `admin@negotiatorpro.local`
   - Password: `admin123` (change after first login!)
   - Role: `admin`

2. **Test User**
   - Username: `testuser`
   - Email: `test@example.com`
   - Password: `testpass123`
   - Role: `user`

## API Testing

### Using Swagger UI
1. Open http://localhost:8000/api/docs
2. Expand `/api/users/` endpoints
3. Click "Try it out"
4. Test user creation, retrieval, updates

### Using curl
```bash
# Create user
curl -X POST http://localhost:8000/api/users/ \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "email": "john@example.com",
    "password": "SecurePass123!",
    "first_name": "John",
    "last_name": "Doe",
    "openai_api_key": "sk-...",
    "role": "user"
  }'

# Get user
curl http://localhost:8000/api/users/username/johndoe

# Update user
curl -X PATCH http://localhost:8000/api/users/{user_id} \
  -H "Content-Type: application/json" \
  -d '{"first_name": "Jonathan"}'
```

## Data Persistence

All data persists in the `/data` directory:

| Directory | Purpose | Size |
|-----------|---------|------|
| `data/db/` | PostgreSQL files | ~100MB per 10k chats |
| `data/vectorstore/` | FAISS embeddings | ~10MB per 100 pages |
| `data/sources/` | Source documents | Your PDFs/DOCX |
| `data/uploads/` | Temp uploads | Cleaned automatically |
| `data/config/` | JSON configs | ~10KB |
| `data/backups/` | Backups | As needed |

## Backup & Restore

### Full Backup
```bash
# Database
docker exec negotiator-pro-postgres pg_dump -U negotiatorpro negotiatorpro > data/backups/db.sql

# Config files
cp -r data/config data/backups/config_$(date +%Y%m%d)/

# Everything
tar -czf backup_$(date +%Y%m%d).tar.gz data/
```

### Restore
```bash
# Database
cat data/backups/db.sql | docker exec -i negotiator-pro-postgres psql -U negotiatorpro -d negotiatorpro

# Config
cp -r data/backups/config_YYYYMMDD/* data/config/

# Everything
tar -xzf backup_YYYYMMDD.tar.gz
docker compose restart
```

## Next Steps

### 1. Integration with Chat
Modify chat endpoint to use user-specific API keys:

```python
# In chat route
user_keys = await UserProfileManager.get_user_api_keys(user_id)
openai_key = user_keys["openai_api_key"] or os.getenv("OPENAI_API_KEY")

# Use user's key for this request
```

### 2. Add Authentication
- Implement JWT token authentication
- Protect user endpoints
- Add login/logout routes
- Session management

### 3. Build UI
- User registration form
- Profile settings page
- API key management
- Account deletion

### 4. Multi-tenancy
- Link users to chat sessions
- Track conversation history per user
- User-specific usage analytics
- Per-user rate limiting

## Benefits of This Architecture

✅ **Clean Separation**: Code vs. data clearly separated
✅ **No Host Dependencies**: Everything in Docker
✅ **Easy Backups**: Just backup `/data` directory
✅ **Production Ready**: Same setup for dev/staging/prod
✅ **Portable**: Deploy anywhere Docker runs
✅ **Scalable**: Easy to add more services
✅ **Maintainable**: Clear structure and documentation

## Troubleshooting

See [DOCKER_SETUP.md](DOCKER_SETUP.md) for detailed troubleshooting, including:
- PostgreSQL connection issues
- Migration failures
- Encryption key errors
- Port conflicts
- Permission problems

## Summary

You now have a **fully functional, Docker-first user profile system** with:
- ✅ PostgreSQL database in Docker
- ✅ User CRUD operations
- ✅ Encrypted API key storage
- ✅ REST API endpoints
- ✅ Automatic migrations
- ✅ Default users created
- ✅ All data in `/data` mount
- ✅ Production-ready architecture
- ✅ Comprehensive documentation

**No host dependencies. Everything in Docker. All data in one place.**
