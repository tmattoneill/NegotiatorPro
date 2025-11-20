# User Profile Setup Guide

This guide explains how to set up and use the user profile system in NegotiatorPro.

## Overview

The user profile system provides:
- User registration and authentication
- Personal information storage (name, email, username)
- Secure API key management (OpenAI, Anthropic)
- PostgreSQL database backend with asyncpg
- Encrypted storage of sensitive data

## Prerequisites

### 1. PostgreSQL Installation

**macOS (using Homebrew):**
```bash
brew install postgresql@15
brew services start postgresql@15
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### 2. Create Database User

```bash
# Connect to PostgreSQL
sudo -u postgres psql

# Create user and database
CREATE USER negotiatorpro WITH PASSWORD 'your_secure_password';
CREATE DATABASE negotiatorpro OWNER negotiatorpro;
GRANT ALL PRIVILEGES ON DATABASE negotiatorpro TO negotiatorpro;

# Exit
\q
```

## Setup Instructions

### Step 1: Install Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate

# Install new dependencies
pip install -r requirements.txt
```

New dependencies added:
- `asyncpg>=0.29.0` - PostgreSQL async driver
- `cryptography>=41.0.0` - Encryption for API keys

### Step 2: Configure Environment Variables

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and configure:

```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=negotiatorpro
POSTGRES_USER=negotiatorpro
POSTGRES_PASSWORD=your_secure_password_here

# Generate encryption key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Add the generated key to .env
ENCRYPTION_KEY=your_generated_encryption_key_here
```

### Step 3: Run Database Setup

The setup script will:
1. Create the database (if needed)
2. Run all migrations
3. Create a test user

```bash
python scripts/setup_database.py
```

This creates a test user:
- **Username**: `testuser`
- **Email**: `test@example.com`
- **Password**: `testpass123`

### Step 4: Test the Implementation

Run the test script:
```bash
python test_user_profile.py
```

This will test:
- User creation
- User retrieval (by ID, username, email)
- Profile updates
- API key encryption/decryption

## Database Schema

### Users Table

The `users` table includes these fields:

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| username | VARCHAR(255) | Unique username |
| email | VARCHAR(255) | Unique email address |
| password_hash | VARCHAR(255) | Bcrypt hashed password |
| first_name | VARCHAR(100) | First name (optional) |
| last_name | VARCHAR(100) | Last name (optional) |
| openai_api_key | TEXT | Encrypted OpenAI API key (optional) |
| anthropic_api_key | TEXT | Encrypted Anthropic API key (optional) |
| role | VARCHAR(50) | User role (admin/user/viewer) |
| created_at | TIMESTAMP | Account creation timestamp |
| last_login | TIMESTAMP | Last login timestamp |
| is_active | BOOLEAN | Account active status |
| profile_updated_at | TIMESTAMP | Profile last updated timestamp |

### Migrations

Two migration files:
1. `001_initial_schema.sql` - Complete database schema
2. `002_add_user_profile_fields.sql` - User profile extensions

## API Endpoints

The user profile system provides these REST endpoints:

### Create User
```http
POST /api/users/
Content-Type: application/json

{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "SecurePass123!",
  "first_name": "John",
  "last_name": "Doe",
  "openai_api_key": "sk-...",  // optional
  "anthropic_api_key": "sk-ant-...",  // optional
  "role": "user"
}
```

### Get User by ID
```http
GET /api/users/{user_id}
```

### Get User by Username
```http
GET /api/users/username/{username}
```

### Get User by Email
```http
GET /api/users/email/{email}
```

### Update User Profile
```http
PATCH /api/users/{user_id}
Content-Type: application/json

{
  "first_name": "Jonathan",
  "last_name": "Doe Jr.",
  "openai_api_key": "sk-new-key"  // optional
}
```

### Get User API Keys
```http
GET /api/users/{user_id}/api-keys
```

Returns decrypted API keys (should be protected with authentication).

### Delete User
```http
DELETE /api/users/{user_id}
```

## Security Features

### Password Hashing
- Uses bcrypt for password hashing
- Configurable cost factor
- Automatic salt generation

### API Key Encryption
- Uses Fernet (symmetric encryption)
- Keys encrypted at rest
- Automatic decryption on retrieval
- Environment-based encryption key

### Database Security
- Prepared statements (SQL injection protection)
- Connection pooling with limits
- Automatic connection cleanup

## Testing with Swagger UI

1. Start the API:
```bash
./run-api.sh
```

2. Open Swagger UI:
```
http://localhost:8000/api/docs
```

3. Try the endpoints:
   - Expand `/api/users/` endpoints
   - Click "Try it out"
   - Enter test data
   - Click "Execute"

## Next Steps

### 1. Add Authentication Middleware
Protect user endpoints with JWT authentication so users can only access/modify their own profiles.

### 2. Integrate with Chat System
Use user-specific API keys when making LLM calls instead of global environment variables.

### 3. Add User Management UI
Create React components for:
- User registration
- Profile editing
- API key management

### 4. Add Session Management
Link user profiles with chat sessions for conversation history.

## Troubleshooting

### Database Connection Failed
```
Failed to connect to database: could not connect to server
```

**Solutions:**
- Check PostgreSQL is running: `sudo systemctl status postgresql`
- Verify credentials in `.env`
- Check host and port settings

### Encryption Key Error
```
No ENCRYPTION_KEY found in environment
```

**Solution:**
Generate and add to `.env`:
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### Migration Failed
```
Migration failed: relation "users" already exists
```

**Solution:**
The migration is idempotent. If the table exists, you can skip migration 001 or drop the database and recreate:
```bash
sudo -u postgres psql
DROP DATABASE negotiatorpro;
CREATE DATABASE negotiatorpro OWNER negotiatorpro;
\q

# Re-run setup
python scripts/setup_database.py
```

### User Already Exists
```
ValueError: Username or email already exists
```

**Solution:**
This is expected if you run the setup script multiple times. The test user already exists.

## File Structure

```
backend/
  database.py              # Database connection management
  user_profile.py          # User profile models and operations
  api/
    routes/
      users.py             # User profile API endpoints

migrations/
  001_initial_schema.sql   # Initial database schema
  002_add_user_profile_fields.sql  # User profile extensions

scripts/
  setup_database.py        # Database setup and migration script

docs/
  USER_PROFILE_SETUP.md    # This file

test_user_profile.py       # Test script
```

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [asyncpg Documentation](https://magicstack.github.io/asyncpg/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Cryptography Library](https://cryptography.io/)
