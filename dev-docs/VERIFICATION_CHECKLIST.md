# User Profile System - Verification Checklist

Use this checklist to verify your Docker-based user profile system is properly set up.

## ✅ Pre-Flight Checks

### Files Created
- [ ] `backend/database.py` - Database connection manager
- [ ] `backend/user_profile.py` - User CRUD operations
- [ ] `backend/config_paths.py` - Docker-aware path management
- [ ] `backend/api/routes/users.py` - User REST endpoints
- [ ] `migrations/002_add_user_profile_fields.sql` - User profile migration
- [ ] `scripts/init_user_profile.py` - User initialization script
- [ ] `data/` directory structure created

### Configuration
- [ ] `.env` file exists and configured
- [ ] `OPENAI_API_KEY` set in `.env`
- [ ] `POSTGRES_PASSWORD` set in `.env`
- [ ] `ENCRYPTION_KEY` generated and set in `.env`

### Docker Configuration
- [ ] `docker-compose.yml` includes PostgreSQL service
- [ ] PostgreSQL uses `./data/db` mount
- [ ] Backend uses `./data/config` mount
- [ ] Backend depends on healthy PostgreSQL
- [ ] Migrations mounted to `/docker-entrypoint-initdb.d`

### Dependencies
- [ ] `asyncpg>=0.29.0` in `requirements.txt`
- [ ] `cryptography>=41.0.0` in `requirements.txt`

## 🚀 Startup Verification

### 1. Start Services
```bash
docker compose up -d
```

Expected output:
```
✓ Network negotiator-network Created
✓ Container negotiator-pro-postgres Started
✓ Container negotiator-pro-backend Started
✓ Container negotiator-pro-frontend Started
```

### 2. Check Container Status
```bash
docker ps
```

Verify:
- [ ] `negotiator-pro-postgres` is running (healthy)
- [ ] `negotiator-pro-backend` is running
- [ ] `negotiator-pro-frontend` is running

### 3. Check PostgreSQL Logs
```bash
docker compose logs postgres | tail -20
```

Look for:
- [ ] "database system is ready to accept connections"
- [ ] Migration files executed (001, 002)
- [ ] No errors

### 4. Check Backend Logs
```bash
docker compose logs backend | tail -20
```

Look for:
- [ ] "Database connection established"
- [ ] "Application startup complete"
- [ ] No connection errors

### 5. Initialize Users
```bash
docker exec -it negotiator-pro-backend python scripts/init_user_profile.py
```

Expected output:
```
✓ Database connection established
✓ Created default admin user
  Username: admin
  Email: admin@negotiatorpro.local
✓ Created test user
  Username: testuser
  Email: test@example.com
```

## 🧪 API Testing

### 1. Health Check
```bash
curl http://localhost:8000/api/health
```

Expected: `{"status": "healthy"}`

- [ ] Health endpoint responds

### 2. API Documentation
```bash
curl http://localhost:8000/api/docs
```

- [ ] Swagger UI loads at http://localhost:8000/api/docs
- [ ] `/api/users/` endpoints visible

### 3. Get User by Username
```bash
curl http://localhost:8000/api/users/username/admin
```

Expected: JSON with admin user data

- [ ] Returns user object
- [ ] Contains `username`, `email`, `role`
- [ ] `has_openai_key` and `has_anthropic_key` fields present
- [ ] Password NOT included in response

### 4. Create New User
```bash
curl -X POST http://localhost:8000/api/users/ \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser2",
    "email": "test2@example.com",
    "password": "SecurePass123!",
    "first_name": "Test",
    "last_name": "User2",
    "role": "user"
  }'
```

- [ ] Returns 201 Created
- [ ] User object returned with ID
- [ ] Can retrieve user: `curl http://localhost:8000/api/users/username/testuser2`

### 5. Update User
```bash
# Get user ID from previous step
USER_ID="<user-id-from-response>"

curl -X PATCH http://localhost:8000/api/users/$USER_ID \
  -H "Content-Type: application/json" \
  -d '{"first_name": "Updated"}'
```

- [ ] Returns 200 OK
- [ ] `first_name` changed to "Updated"

### 6. Get API Keys (if user has keys)
```bash
curl http://localhost:8000/api/users/$USER_ID/api-keys
```

- [ ] Returns JSON with `openai_api_key` and `anthropic_api_key`
- [ ] Keys are null if not set

## 🗄️ Database Verification

### 1. Access PostgreSQL
```bash
docker exec -it negotiator-pro-postgres psql -U negotiatorpro -d negotiatorpro
```

### 2. List Tables
```sql
\dt
```

Expected tables:
- [ ] users
- [ ] sessions
- [ ] chat_messages
- [ ] usage_logs
- [ ] documents
- [ ] llm_config
- [ ] embedding_config
- [ ] prompts

### 3. Check Users Table
```sql
SELECT username, email, first_name, last_name, role, created_at 
FROM users;
```

- [ ] At least 2 users (admin, testuser)
- [ ] All fields populated correctly

### 4. Verify Profile Fields
```sql
\d users
```

Look for:
- [ ] `first_name` column
- [ ] `last_name` column
- [ ] `openai_api_key` column
- [ ] `anthropic_api_key` column
- [ ] `profile_updated_at` column

### 5. Check Encryption
```sql
SELECT username, 
       openai_api_key IS NOT NULL as has_openai,
       LENGTH(openai_api_key) as key_length
FROM users 
WHERE openai_api_key IS NOT NULL;
```

If any users have keys:
- [ ] Keys are encrypted (long base64 strings)
- [ ] NOT plaintext

## 📁 Data Persistence

### 1. Check Data Directory
```bash
ls -la data/
```

Expected directories:
- [ ] `data/db/` exists and contains PostgreSQL files
- [ ] `data/config/` exists
- [ ] `data/vectorstore/` exists
- [ ] `data/sources/` exists
- [ ] `data/uploads/` exists
- [ ] `data/backups/` exists

### 2. Check Config Files
```bash
ls -la data/config/
```

Expected files (may be created on first use):
- [ ] Some JSON config files may exist
- [ ] Directory is writable

### 3. Test Persistence
```bash
# Stop containers
docker compose down

# Check data still exists
ls -la data/db/

# Start again
docker compose up -d

# Verify users still exist
sleep 10
curl http://localhost:8000/api/users/username/admin
```

- [ ] Data persists across container restarts

## 🔒 Security Verification

### 1. Password Hashing
```sql
-- In psql
SELECT username, password_hash 
FROM users 
LIMIT 1;
```

- [ ] Password is bcrypt hash (starts with `$2b$`)
- [ ] NOT plaintext

### 2. API Key Encryption
```sql
SELECT username, openai_api_key 
FROM users 
WHERE openai_api_key IS NOT NULL 
LIMIT 1;
```

- [ ] API key is encrypted (long base64 string)
- [ ] Starts with `gAAAAA` (Fernet encryption)

### 3. Environment Secrets
```bash
docker exec negotiator-pro-backend env | grep ENCRYPTION_KEY
```

- [ ] ENCRYPTION_KEY is set
- [ ] Not visible in logs or API responses

## 📊 Frontend Integration (Future)

- [ ] Frontend can display user list
- [ ] Frontend can create users
- [ ] Frontend can update profiles
- [ ] Frontend can manage API keys

## 🐛 Troubleshooting Checks

If something fails, verify:

### PostgreSQL Issues
```bash
# Is it running?
docker ps | grep postgres

# Logs
docker compose logs postgres

# Can backend connect?
docker exec negotiator-pro-backend python -c "
from backend.database import db
import asyncio
asyncio.run(db.health_check())
"
```

### Backend Issues
```bash
# Logs
docker compose logs backend

# Is Python environment ok?
docker exec negotiator-pro-backend python --version
docker exec negotiator-pro-backend pip list | grep asyncpg
```

### Migration Issues
```bash
# Check if migrations ran
docker exec negotiator-pro-postgres psql -U negotiatorpro -d negotiatorpro -c "\dt"

# Re-run migrations manually if needed
docker exec negotiator-pro-postgres psql -U negotiatorpro -d negotiatorpro < migrations/001_initial_schema.sql
docker exec negotiator-pro-postgres psql -U negotiatorpro -d negotiatorpro < migrations/002_add_user_profile_fields.sql
```

## ✅ Final Checklist

- [ ] All services running (postgres, backend, frontend)
- [ ] Database initialized with migrations
- [ ] Default users created (admin, testuser)
- [ ] Can create users via API
- [ ] Can retrieve users via API
- [ ] Can update users via API
- [ ] Passwords are hashed (bcrypt)
- [ ] API keys are encrypted (Fernet)
- [ ] Data persists in `/data` directory
- [ ] Data survives container restarts
- [ ] No errors in logs
- [ ] API documentation accessible

## 🎉 Success!

If all checks pass, you have a fully functional Docker-based user profile system!

Next steps:
1. Integrate user authentication into chat endpoints
2. Use user-specific API keys for LLM calls
3. Build frontend UI for user management
4. Add session management
5. Implement usage quotas per user

See:
- [DOCKER_SETUP.md](DOCKER_SETUP.md) - Complete Docker guide
- [USER_PROFILE_DOCKER_SUMMARY.md](USER_PROFILE_DOCKER_SUMMARY.md) - Implementation summary
- [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture diagrams
