# Docker Quick Start - User Profile System

## Prerequisites
- Docker & Docker Compose installed
- That's it! No Python, Node, or PostgreSQL needed on host

## Setup (5 minutes)

### 1. Configure Environment
```bash
cp .env.example .env
```

Edit `.env`:
```bash
OPENAI_API_KEY=sk-your-key-here
POSTGRES_PASSWORD=your_secure_password_here
ENCRYPTION_KEY=$(docker run --rm python:3.11 python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
```

### 2. Start Services
```bash
docker compose up -d
```

Wait ~30 seconds for PostgreSQL to initialize.

### 3. Initialize User Profiles
```bash
docker exec -it negotiator-pro-backend python scripts/init_user_profile.py
```

## What You Get

### Services Running
- **PostgreSQL**: localhost:5432
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/api/docs
- **Frontend**: http://localhost:5173

### Default Users
- **Admin**: `admin` / `admin123`
- **Test User**: `testuser` / `testpass123`

### Data Locations
All persistent data in `/data` directory:
```
data/
├── db/           # PostgreSQL data
├── vectorstore/  # FAISS embeddings
├── sources/      # Source documents
├── uploads/      # Temporary uploads
└── config/       # JSON config files
```

## Common Commands

### View Logs
```bash
docker compose logs -f backend
docker compose logs -f postgres
```

### Create User
```bash
curl -X POST http://localhost:8000/api/users/ \
  -H "Content-Type: application/json" \
  -d '{
    "username": "newuser",
    "email": "user@example.com",
    "password": "SecurePass123!",
    "first_name": "New",
    "last_name": "User"
  }'
```

### Access Database
```bash
docker exec -it negotiator-pro-postgres psql -U negotiatorpro -d negotiatorpro
```

### Restart Services
```bash
docker compose restart
```

### Stop Everything
```bash
docker compose down
```

### Fresh Start (Deletes All Data)
```bash
docker compose down
rm -rf data/
docker compose up -d
docker exec -it negotiator-pro-backend python scripts/init_user_profile.py
```

## API Endpoints

### User Management
- `POST /api/users/` - Create user
- `GET /api/users/{user_id}` - Get user by ID
- `GET /api/users/username/{username}` - Get user by username
- `PATCH /api/users/{user_id}` - Update user
- `DELETE /api/users/{user_id}` - Delete user
- `GET /api/users/{user_id}/api-keys` - Get encrypted keys

### Test in Browser
Open http://localhost:8000/api/docs and try the interactive API!

## Backup & Restore

### Backup
```bash
# Database
docker exec negotiator-pro-postgres pg_dump -U negotiatorpro negotiatorpro > backup.sql

# All data
tar -czf backup.tar.gz data/
```

### Restore
```bash
# Database
cat backup.sql | docker exec -i negotiator-pro-postgres psql -U negotiatorpro -d negotiatorpro

# All data
tar -xzf backup.tar.gz
docker compose restart
```

## Troubleshooting

### Can't connect to database?
```bash
# Check health
docker ps

# Should show postgres as "healthy"
# If not, check logs:
docker compose logs postgres
```

### Migrations not applied?
Migrations run automatically on first startup. To re-run:
```bash
docker compose down
rm -rf data/db
docker compose up -d
```

### Port already in use?
Change ports in `docker-compose.yml`:
```yaml
ports:
  - "5433:5432"  # PostgreSQL
  - "8001:8000"  # Backend
  - "5174:5173"  # Frontend
```

## Next Steps

1. ✅ Test API at http://localhost:8000/api/docs
2. ✅ Create users via API or UI
3. ✅ Add source documents to `data/sources/`
4. ✅ Configure LLM backends in admin panel
5. ✅ Start chatting!

## Full Documentation

- **Complete Docker Guide**: [DOCKER_SETUP.md](DOCKER_SETUP.md)
- **User Profile Details**: [docs/USER_PROFILE_SETUP.md](docs/USER_PROFILE_SETUP.md)
- **Production Deployment**: [docs/deployment/DEPLOYMENT.md](docs/deployment/DEPLOYMENT.md)

---

**That's it!** You now have a fully functional multi-user NegotiatorPro instance with PostgreSQL, user profiles, and encrypted API key storage - all running in Docker.
