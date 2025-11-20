# Docker-Based Setup Guide

This guide covers the **Docker-first architecture** for NegotiatorPro where all services (PostgreSQL, Python, Node) run in containers and persistent data lives in the `/data` mount.

## Architecture Overview

```
NegotiatorPro/
├── docker-compose.yml          # Orchestrates all services
├── Dockerfile                  # Backend container build
├── data/                       # ALL persistent data (gitignored)
│   ├── db/                    # PostgreSQL data
│   ├── vectorstore/           # FAISS embeddings
│   ├── uploads/               # Temporary uploads
│   ├── sources/               # Source documents
│   ├── config/                # JSON config files
│   └── backups/               # Backup storage
├── migrations/                 # Database migrations
├── backend/                    # Python backend code
└── frontend/                   # React frontend code
```

## Quick Start

### 1. Prerequisites

- Docker 24.0+ and Docker Compose 2.0+
- No need for local Python, Node, or PostgreSQL!

### 2. Environment Configuration

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and configure:

```bash
# LLM API Keys
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here  # Optional

# Database Configuration
POSTGRES_DB=negotiatorpro
POSTGRES_USER=negotiatorpro
POSTGRES_PASSWORD=your_secure_password_here

# Encryption Key (for API keys in user profiles)
# Generate with: docker run --rm python:3.11 python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
ENCRYPTION_KEY=your_generated_key_here
```

### 3. Generate Encryption Key

```bash
docker run --rm python:3.11 python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Copy the output and add to `.env` as `ENCRYPTION_KEY=...`

### 4. Start All Services

```bash
docker compose up -d
```

This will:
1. Start PostgreSQL database
2. Run migrations automatically (from `/migrations` directory)
3. Start FastAPI backend
4. Start React frontend

### 5. Initialize User Profiles

Run the initialization script inside the backend container:

```bash
docker exec -it negotiator-pro-backend python scripts/init_user_profile.py
```

This creates:
- **Admin user**: `admin` / `admin123`
- **Test user**: `testuser` / `testpass123`

### 6. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/api/docs
- **PostgreSQL**: localhost:5432 (if you need direct access)

## Service Management

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend
docker compose logs -f postgres
docker compose logs -f frontend
```

### Restart Services

```bash
# All services
docker compose restart

# Specific service
docker compose restart backend
```

### Stop Services

```bash
docker compose stop
```

### Rebuild After Code Changes

```bash
# Rebuild backend
docker compose up -d --build backend

# Rebuild all
docker compose up -d --build
```

### Clean Start (Remove All Data)

```bash
# Stop services
docker compose down

# Remove data (WARNING: This deletes everything!)
rm -rf data/db data/vectorstore data/config

# Start fresh
docker compose up -d
docker exec -it negotiator-pro-backend python scripts/init_user_profile.py
```

## Database Operations

### Access PostgreSQL CLI

```bash
docker exec -it negotiator-pro-postgres psql -U negotiatorpro -d negotiatorpro
```

Common queries:

```sql
-- List all users
SELECT username, email, first_name, last_name, role, created_at FROM users;

-- Count users
SELECT COUNT(*) FROM users;

-- View tables
\dt

-- Exit
\q
```

### Run Migrations Manually

Migrations run automatically on first startup, but you can re-run them:

```bash
# Connect to database
docker exec -it negotiator-pro-postgres psql -U negotiatorpro -d negotiatorpro

# In psql, run migration
\i /docker-entrypoint-initdb.d/001_initial_schema.sql
\i /docker-entrypoint-initdb.d/002_add_user_profile_fields.sql
```

### Create Database Backup

```bash
docker exec negotiator-pro-postgres pg_dump -U negotiatorpro negotiatorpro > data/backups/backup_$(date +%Y%m%d_%H%M%S).sql
```

### Restore Database

```bash
cat data/backups/backup_YYYYMMDD_HHMMSS.sql | docker exec -i negotiator-pro-postgres psql -U negotiatorpro -d negotiatorpro
```

## User Profile Management

### Create User via API

```bash
curl -X POST http://localhost:8000/api/users/ \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "email": "john@example.com",
    "password": "SecurePass123!",
    "first_name": "John",
    "last_name": "Doe",
    "openai_api_key": "sk-...",
    "anthropic_api_key": "sk-ant-...",
    "role": "user"
  }'
```

### Get User

```bash
curl http://localhost:8000/api/users/username/johndoe
```

### Update User Profile

```bash
curl -X PATCH http://localhost:8000/api/users/{user_id} \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "Jonathan",
    "openai_api_key": "sk-new-key"
  }'
```

### Python Script (Inside Container)

```bash
docker exec -it negotiator-pro-backend python -c "
import asyncio
from backend.database import db
from backend.user_profile import UserProfileManager

async def main():
    await db.connect()
    users = await db.fetch('SELECT username, email FROM users')
    for user in users:
        print(f'{user[\"username\"]}: {user[\"email\"]}')
    await db.disconnect()

asyncio.run(main())
"
```

## Adding Source Documents

### Copy Documents to Container

```bash
# Copy PDF to sources directory
cp path/to/book.pdf data/sources/

# Or use docker cp
docker cp path/to/book.pdf negotiator-pro-backend:/app/sources/
```

### Rebuild Vectorstore

After adding new documents, rebuild the vectorstore:

```bash
docker exec -it negotiator-pro-backend python scripts/rebuild_vectordb.py
```

## Development Workflow

### Hot Reload

The backend code is mounted as a volume, so changes are reflected immediately:

```yaml
volumes:
  - ./backend:/app/backend  # Hot reload enabled
```

Just edit files in `backend/` and the FastAPI server will auto-reload.

### Frontend Development

Frontend runs in dev mode with hot reload:

```bash
# Watch logs
docker compose logs -f frontend

# Changes to frontend/ files will auto-reload
```

### Installing Python Packages

```bash
# Add package to requirements.txt
echo "new-package==1.0.0" >> requirements.txt

# Rebuild backend
docker compose up -d --build backend
```

### Installing Node Packages

```bash
# Add package
docker exec negotiator-pro-frontend npm install --save new-package

# Or edit package.json and restart
docker compose restart frontend
```

## Troubleshooting

### PostgreSQL Not Starting

Check logs:
```bash
docker compose logs postgres
```

Common issues:
- Permissions on `data/db/` directory
- Port 5432 already in use
- Incorrect password in `.env`

Solution:
```bash
# Stop everything
docker compose down

# Fix permissions
sudo chown -R $(whoami) data/

# Start again
docker compose up -d
```

### Backend Can't Connect to Database

Check:
1. PostgreSQL is healthy: `docker ps` (should show "healthy")
2. Environment variables: `docker exec negotiator-pro-backend env | grep POSTGRES`
3. Network: `docker network inspect negotiator-network`

### Encryption Key Error

If you see "No ENCRYPTION_KEY found":

```bash
# Generate key
docker run --rm python:3.11 python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Add to .env
echo "ENCRYPTION_KEY=<generated-key>" >> .env

# Restart
docker compose restart backend
```

### Migrations Not Applied

Migrations only run on fresh database. If you need to re-run:

```bash
# Drop database
docker exec -it negotiator-pro-postgres psql -U negotiatorpro -c "DROP DATABASE negotiatorpro;"
docker exec -it negotiator-pro-postgres psql -U negotiatorpro -c "CREATE DATABASE negotiatorpro;"

# Restart postgres (migrations will run)
docker compose restart postgres
```

### Can't Access Frontend

Check:
```bash
# Frontend logs
docker compose logs -f frontend

# Is it running?
docker ps | grep frontend

# Check port mapping
curl http://localhost:5173
```

## Data Persistence

All data persists in the `/data` directory:

```bash
# Check sizes
du -sh data/*

# Example output:
# 150M    data/db
# 25M     data/vectorstore
# 5M      data/sources
# 100K    data/config
```

### Backup Everything

```bash
# Create backup directory
mkdir -p data/backups/full_backup_$(date +%Y%m%d)

# Backup database
docker exec negotiator-pro-postgres pg_dump -U negotiatorpro negotiatorpro \
  > data/backups/full_backup_$(date +%Y%m%d)/database.sql

# Backup config
cp -r data/config data/backups/full_backup_$(date +%Y%m%d)/

# Backup vectorstore
cp -r data/vectorstore data/backups/full_backup_$(date +%Y%m%d)/

# Create archive
tar -czf backup_$(date +%Y%m%d).tar.gz data/backups/full_backup_$(date +%Y%m%d)/
```

### Restore from Backup

```bash
# Extract backup
tar -xzf backup_YYYYMMDD.tar.gz

# Stop services
docker compose down

# Restore data
cp -r data/backups/full_backup_YYYYMMDD/config/* data/config/
cp -r data/backups/full_backup_YYYYMMDD/vectorstore/* data/vectorstore/

# Start services
docker compose up -d

# Restore database
cat data/backups/full_backup_YYYYMMDD/database.sql | \
  docker exec -i negotiator-pro-postgres psql -U negotiatorpro -d negotiatorpro
```

## Production Deployment

For production deployments:

1. **Change default passwords**:
   - PostgreSQL password in `.env`
   - Admin user password (first login)
   - Generate new encryption key

2. **Use secrets management**:
   - Docker secrets for sensitive data
   - Environment variable injection
   - Key management service (AWS KMS, etc.)

3. **Set up reverse proxy**:
   - Nginx or Traefik in front
   - HTTPS with Let's Encrypt
   - Rate limiting

4. **Configure backups**:
   - Automated daily backups
   - Off-site backup storage
   - Test restore procedures

5. **Monitoring**:
   - Docker health checks
   - Log aggregation (ELK stack)
   - Metrics (Prometheus/Grafana)

See `docs/deployment/DEPLOYMENT.md` for full production setup guide.

## Summary

With this Docker-first architecture:

- ✅ No local Python/Node/PostgreSQL installation needed
- ✅ All data in `/data` directory (easy to backup/restore)
- ✅ Clean separation of code and data
- ✅ Production-ready containerization
- ✅ Easy to deploy to any Docker host
- ✅ Consistent environment across dev/staging/prod

All functionality is self-contained in Docker containers with persistent data properly managed!
