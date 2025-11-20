# User Profile System - Implementation Summary

## What Was Built

I've successfully implemented a complete user profile system for NegotiatorPro with the following features:

### 1. Database Schema
- Extended PostgreSQL database with user profile fields
- Migration scripts for database setup
- Secure storage of API keys with encryption

### 2. User Profile Fields
- **Username** (unique)
- **Email** (unique)
- **First Name** (optional)
- **Last Name** (optional)
- **OpenAI API Key** (optional, encrypted)
- **Anthropic API Key** (optional, encrypted)
- **Role** (admin/user/viewer)
- **Password** (bcrypt hashed)

### 3. Core Components

#### backend/database.py
- Async PostgreSQL connection pooling using `asyncpg`
- Connection management with context managers
- Health check functionality
- Support for DATABASE_URL or individual connection parameters

#### backend/user_profile.py
- **UserProfileCreate** - Model for creating users
- **UserProfileUpdate** - Model for updating profiles
- **UserProfile** - Response model (excludes sensitive data)
- **EncryptionManager** - Handles API key encryption/decryption
- **UserProfileManager** - CRUD operations for user profiles

#### backend/api/routes/users.py
- RESTful API endpoints for user management
- Full CRUD operations
- Secure API key retrieval
- Input validation with Pydantic

### 4. Database Migrations

#### migrations/001_initial_schema.sql
- Complete database schema with tables for:
  - Users
  - Sessions
  - LLM Configuration
  - Usage Logs
  - Documents
  - Chat Messages
  - And more...

#### migrations/002_add_user_profile_fields.sql
- Adds user profile fields to existing users table
- Encryption support for API keys
- Automatic timestamp triggers

### 5. API Endpoints

All endpoints available at `/api/users/`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/users/` | Create new user |
| GET | `/api/users/{user_id}` | Get user by ID |
| GET | `/api/users/username/{username}` | Get user by username |
| GET | `/api/users/email/{email}` | Get user by email |
| PATCH | `/api/users/{user_id}` | Update user profile |
| DELETE | `/api/users/{user_id}` | Delete user |
| GET | `/api/users/{user_id}/api-keys` | Get decrypted API keys |

### 6. Docker Integration

Updated `docker-compose.yml` to include:
- PostgreSQL 15 container
- Automatic database initialization from migration scripts
- Persistent volume for database data
- Health checks for database availability
- Environment variable configuration

### 7. Security Features

- **Password Hashing**: Bcrypt with automatic salting
- **API Key Encryption**: Fernet symmetric encryption
- **SQL Injection Protection**: Parameterized queries with asyncpg
- **Connection Pooling**: Prevents resource exhaustion
- **Environment-based Secrets**: Encryption key from .env file

## Quick Start

### 1. Start Docker Services

```bash
docker compose up -d
```

This will:
- Start PostgreSQL database
- Run migrations automatically
- Start FastAPI backend
- Start React frontend

### 2. Verify Database

Check that PostgreSQL is running:
```bash
docker compose logs postgres
```

### 3. Test API Endpoints

Open Swagger UI:
```
http://localhost:8000/api/docs
```

Try creating a user:
```json
POST /api/users/
{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "SecurePass123!",
  "first_name": "John",
  "last_name": "Doe",
  "openai_api_key": "sk-...",
  "role": "user"
}
```

### 4. Access Database (Optional)

Connect to PostgreSQL container:
```bash
docker exec -it negotiator-pro-postgres psql -U negotiatorpro -d negotiatorpro
```

Query users:
```sql
SELECT username, email, first_name, last_name, role, created_at FROM users;
```

## Environment Variables

The following variables are configured in `.env`:

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=negotiatorpro
POSTGRES_USER=negotiatorpro
POSTGRES_PASSWORD=negotiatorpro_secure_2024

# Encryption
ENCRYPTION_KEY=MRZ2XWtdXLHHp6rhxfzwblbQY2rraT18o13IbMHgddg=
```

## File Structure

```
backend/
├── database.py              # Database connection management
├── user_profile.py          # User profile models and operations
└── api/
    └── routes/
        └── users.py         # User API endpoints

migrations/
├── 001_initial_schema.sql   # Initial database schema
└── 002_add_user_profile_fields.sql  # Profile extensions

scripts/
└── setup_database.py        # Database setup script

docs/
└── USER_PROFILE_SETUP.md    # Detailed setup guide

test_user_profile.py         # Test script for user operations
USER_PROFILE_SUMMARY.md      # This file
```

## Next Steps

### 1. Integration with Chat System
You can now integrate user-specific API keys into the chat flow:

```python
# In chat endpoint
user_keys = await UserProfileManager.get_user_api_keys(user_id)

# Use user's keys if available, otherwise fall back to env vars
openai_key = user_keys["openai_api_key"] or os.getenv("OPENAI_API_KEY")
anthropic_key = user_keys["anthropic_api_key"] or os.getenv("ANTHROPIC_API_KEY")
```

### 2. Add Authentication
Protect user endpoints with JWT authentication:
- Only allow users to access/modify their own profiles
- Admin users can manage all profiles
- Implement session management

### 3. Build User Management UI
Create React components for:
- User registration form
- Profile settings page
- API key management
- Account deletion

### 4. Session Tracking
Link user profiles with chat sessions:
- Track conversation history per user
- Store preferences and settings
- Usage analytics per user

### 5. Multi-tenancy
If you want to support multiple users:
- Add user authentication to chat endpoints
- Isolate vectorstores per user or organization
- Implement usage quotas and rate limiting

## Testing

### Manual Testing with curl

Create user:
```bash
curl -X POST http://localhost:8000/api/users/ \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "testpass123",
    "first_name": "Test",
    "last_name": "User"
  }'
```

Get user:
```bash
curl http://localhost:8000/api/users/username/testuser
```

Update user:
```bash
curl -X PATCH http://localhost:8000/api/users/{user_id} \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "Updated",
    "openai_api_key": "sk-new-key"
  }'
```

### Automated Testing

Run the test script:
```bash
python test_user_profile.py
```

## Troubleshooting

### Database connection failed
```
Failed to connect to database: could not connect to server
```

**Solution**: Ensure PostgreSQL container is running:
```bash
docker compose up -d postgres
docker compose logs postgres
```

### Migrations not applied
The migrations are automatically run when the PostgreSQL container starts. Check:
```bash
docker compose logs postgres | grep -i "migrat"
```

### Encryption key error
If you see "No ENCRYPTION_KEY found", add it to `.env`:
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Add output to .env as ENCRYPTION_KEY=...
```

## Security Notes

1. **Change Default Password**: Update `POSTGRES_PASSWORD` in `.env` for production
2. **Secure Encryption Key**: Generate a new `ENCRYPTION_KEY` and store it securely
3. **HTTPS Only**: In production, use HTTPS for all API calls
4. **Rate Limiting**: Implement rate limiting on user creation endpoints
5. **Input Validation**: All inputs are validated by Pydantic models
6. **SQL Injection**: Using parameterized queries with asyncpg prevents SQL injection

## Dependencies Added

Updated `requirements.txt`:
```
asyncpg>=0.29.0          # PostgreSQL async driver
cryptography>=41.0.0     # Encryption library
```

Install with:
```bash
pip install -r requirements.txt
```

## API Documentation

Full API documentation available at:
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc
- OpenAPI JSON: http://localhost:8000/api/openapi.json

## Summary

You now have a complete user profile system with:
- ✅ PostgreSQL database integration
- ✅ User registration and management
- ✅ Secure API key storage
- ✅ RESTful API endpoints
- ✅ Docker deployment
- ✅ Comprehensive documentation
- ✅ Test scripts and utilities

The system is ready to be integrated with authentication and the main chat application!
