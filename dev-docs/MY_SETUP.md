# My NegotiatorPro Setup

## Architecture Overview

Your NegotiatorPro uses a **hybrid LLM model**:

```
┌─────────────────────────────────────────────────┐
│            LLM Model Selection                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  Base Models (Free/Cheap)                       │
│  ├─ Ollama Cloud                                │
│  ├─ System-level API key (in .env)             │
│  └─ Used for: Default queries, testing          │
│                                                 │
│  Premium Models (Pay-per-use)                   │
│  ├─ OpenAI (GPT-4, GPT-4o, etc.)              │
│  ├─ Anthropic Claude                            │
│  ├─ User-level API keys (in PostgreSQL)        │
│  └─ Used for: Complex negotiations, premium    │
│                                                 │
└─────────────────────────────────────────────────┘
```

## API Key Storage

### System-Level Keys (`.env`)
```bash
# Base service - Ollama Cloud
OLLAMA_CLOUD_URL=https://ollama.com
OLLAMA_API_KEY=your_ollama_cloud_key

# Database
POSTGRES_PASSWORD=your_db_password
ENCRYPTION_KEY=your_encryption_key
```

**Used for**: Ollama Cloud models (Llama, Mistral, etc.)

### User-Level Keys (PostgreSQL - Encrypted)
- **OpenAI API Key** - For GPT-4, GPT-4o, etc.
- **Anthropic API Key** - For Claude models

**Used for**: Premium models when user selects them

## Quick Setup

### 1. Configure Environment
```bash
# Edit .env
OLLAMA_CLOUD_URL=https://ollama.com
OLLAMA_API_KEY=<your-ollama-key>
POSTGRES_PASSWORD=<choose-secure-password>
ENCRYPTION_KEY=<generated-key>
```

### 2. Start Services
```bash
docker compose up -d
```

### 3. Setup Your User Profile
```bash
docker exec -it negotiator-pro-backend python scripts/setup_my_user.py
```

This will prompt you for:
- Username (default: admin)
- Email
- Password
- **OpenAI API Key** (your personal key for GPT-4/GPT-4o)
- **Anthropic API Key** (your personal key for Claude)

Your premium keys are **encrypted** before storage using Fernet.

## How It Works

### When You Make a Request

```python
# User selects model in UI
selected_model = "gpt-4o"  # or "llama3.1", "claude-3.5-sonnet", etc.

if selected_model.startswith("gpt-"):
    # OpenAI model - use YOUR API key from database
    api_key = user.openai_api_key  # Decrypted from PostgreSQL

elif selected_model.startswith("claude-"):
    # Anthropic model - use YOUR API key from database
    api_key = user.anthropic_api_key  # Decrypted from PostgreSQL

else:
    # Ollama model - use system key from .env
    api_key = os.getenv("OLLAMA_API_KEY")
```

### Billing

- **Ollama Cloud**: Billed to your Ollama account (from `.env`)
- **OpenAI**: Billed to YOUR OpenAI account (from user profile)
- **Anthropic**: Billed to YOUR Anthropic account (from user profile)

## Model Selection Strategy

### Default/Fast Models (Ollama Cloud)
- **llama3.1:8b** - Fast, cheap, good for simple questions
- **mistral:7b** - Alternative, decent reasoning
- **qwen2.5:14b** - Better reasoning, still cheap

**Cost**: ~$0 (or very low with Ollama Cloud)

### Premium Models (User Keys)
- **gpt-4o-mini** - OpenAI's cheapest smart model
- **gpt-4o** - Best OpenAI model, expensive
- **claude-3.5-sonnet** - Anthropic's best, mid-price
- **claude-3-haiku** - Anthropic's cheapest

**Cost**: Whatever OpenAI/Anthropic charges YOU

## Adding/Updating Your API Keys

### Option 1: Setup Script (Recommended)
```bash
docker exec -it negotiator-pro-backend python scripts/setup_my_user.py
```

### Option 2: Via API
```bash
# Get your user ID
USER_ID=$(curl -s http://localhost:8000/api/users/username/admin | jq -r '.id')

# Update keys
curl -X PATCH http://localhost:8000/api/users/$USER_ID \
  -H "Content-Type: application/json" \
  -d '{
    "openai_api_key": "sk-proj-...",
    "anthropic_api_key": "sk-ant-..."
  }'
```

### Option 3: Via Database (Advanced)
```bash
docker exec -it negotiator-pro-postgres psql -U negotiatorpro -d negotiatorpro

-- View current keys (encrypted)
SELECT username,
       openai_api_key IS NOT NULL as has_openai,
       anthropic_api_key IS NOT NULL as has_anthropic
FROM users;
```

## Viewing Your Keys

```bash
# Get your user ID
USER_ID=$(curl -s http://localhost:8000/api/users/username/admin | jq -r '.id')

# Get decrypted keys
curl http://localhost:8000/api/users/$USER_ID/api-keys
```

**Response**:
```json
{
  "openai_api_key": "sk-proj-...",
  "anthropic_api_key": "sk-ant-...",
  "has_openai_key": true,
  "has_anthropic_key": true
}
```

## Security Notes

### Encryption
Your premium API keys are encrypted using **Fernet** (symmetric encryption):
- Encrypted before storage in PostgreSQL
- Encryption key stored in `.env` (`ENCRYPTION_KEY`)
- Only your app can decrypt them
- Keys are decrypted on-the-fly when needed

### Password
Your user password is hashed using **bcrypt**:
- Never stored in plaintext
- Automatic salting
- Can't be reversed

## Multi-User Future

Currently: Just you (single user)

Future: Multiple users can each have their own premium API keys
```
User A (you):
  ├─ OpenAI: Your key
  └─ Anthropic: Your key

User B (teammate):
  ├─ OpenAI: Their key
  └─ Anthropic: Their key

System:
  └─ Ollama Cloud: Shared for everyone
```

Each user gets billed for their own premium usage.

## Configuration Files

```
.env                    # System config (Ollama, DB, encryption)
data/config/            # Runtime config (LLM backend settings)
data/db/                # PostgreSQL data (includes encrypted user keys)
```

## Backup Your Keys

### Environment Backup
```bash
# Backup .env (contains OLLAMA_API_KEY and ENCRYPTION_KEY)
cp .env .env.backup
```

⚠️ **Important**: If you lose `ENCRYPTION_KEY`, you can't decrypt stored API keys!

### Database Backup
```bash
# Backup database (includes encrypted user API keys)
docker exec negotiator-pro-postgres pg_dump -U negotiatorpro negotiatorpro > backup.sql
```

## Troubleshooting

### "No API key found" error
Check your user profile has keys:
```bash
curl http://localhost:8000/api/users/username/admin
```

Should show `has_openai_key: true` if configured.

### Keys not decrypting
Verify `ENCRYPTION_KEY` in `.env` matches what was used to encrypt:
```bash
docker exec negotiator-pro-backend env | grep ENCRYPTION_KEY
```

### Want to reset keys
```bash
docker exec -it negotiator-pro-backend python scripts/setup_my_user.py
# Choose "update API keys"
```

## Summary

✅ **Base models**: Ollama Cloud (system-level, shared)
✅ **Premium models**: OpenAI/Anthropic (your personal keys, encrypted)
✅ **Billing**: Premium usage billed to YOUR accounts
✅ **Security**: Fernet encryption for API keys, bcrypt for passwords
✅ **Single user**: Just you for now, multi-user ready

Your premium API keys are safely encrypted in the database and only decrypted when needed for LLM calls.
