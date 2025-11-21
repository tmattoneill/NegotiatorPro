# Quick Start - Your Setup

Get NegotiatorPro running in 5 minutes with your hybrid LLM architecture.

## Your Configuration

- **Base Models**: Ollama Cloud (system-level API key)
- **Premium Models**: OpenAI & Anthropic (your personal API keys, encrypted in PostgreSQL)

## Step 1: Configure Environment

```bash
# Copy template
cp .env.example .env
```

Edit `.env` and set:

```bash
# REQUIRED: Ollama Cloud for base models
OLLAMA_CLOUD_URL=https://ollama.com
OLLAMA_API_KEY=<your-ollama-cloud-key>

# REQUIRED: Database password
POSTGRES_PASSWORD=<choose-a-secure-password>

# REQUIRED: Encryption key for user API keys
# Generate with: docker run --rm python:3.11 python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
ENCRYPTION_KEY=<generated-encryption-key>
```

## Step 2: Start Docker Services

```bash
docker compose up -d
```

Wait ~30 seconds for PostgreSQL to initialize.

Check status:
```bash
docker ps
```

Should see 3 running containers:
- `negotiator-pro-postgres` (healthy)
- `negotiator-pro-backend`
- `negotiator-pro-frontend`

## Step 3: Setup Your User Profile

```bash
docker exec -it negotiator-pro-backend python scripts/setup_my_user.py
```

This will prompt you for:
- Username (default: `admin`)
- Email
- Password
- **OpenAI API Key** (for GPT-4, GPT-4o - your personal key)
- **Anthropic API Key** (for Claude models - your personal key)

Your premium keys are encrypted before storage.

## Step 4: Access the Application

- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/api/docs
- **Backend**: http://localhost:8000

## How Model Selection Works

When you chat:

```
Select "llama3.1:8b"     → Uses Ollama Cloud (system key)
Select "gpt-4o"          → Uses YOUR OpenAI key (decrypted)
Select "claude-3-sonnet" → Uses YOUR Anthropic key (decrypted)
```

Billing:
- Ollama Cloud → Your Ollama account
- OpenAI → YOUR OpenAI account
- Anthropic → YOUR Anthropic account

## Common Commands

### View Logs
```bash
docker compose logs -f backend
docker compose logs -f postgres
```

### Check Your User Profile
```bash
curl http://localhost:8000/api/users/username/admin | jq
```

### View Your API Keys
```bash
# Get user ID
USER_ID=$(curl -s http://localhost:8000/api/users/username/admin | jq -r '.id')

# View encrypted keys status
curl http://localhost:8000/api/users/$USER_ID/api-keys | jq
```

### Update Your API Keys
```bash
docker exec -it negotiator-pro-backend python scripts/setup_my_user.py
# Choose "update API keys"
```

### Restart Services
```bash
docker compose restart
```

### Stop Everything
```bash
docker compose down
```

## Data Location

All persistent data in `/data`:
- `data/db/` - PostgreSQL (includes your encrypted keys)
- `data/vectorstore/` - FAISS embeddings
- `data/sources/` - Source documents
- `data/config/` - JSON config files

## Backup

```bash
# Backup database (includes encrypted user keys)
docker exec negotiator-pro-postgres pg_dump -U negotiatorpro negotiatorpro > backup.sql

# Backup .env (contains ENCRYPTION_KEY - don't lose this!)
cp .env .env.backup
```

⚠️ **Important**: If you lose `ENCRYPTION_KEY` in `.env`, you can't decrypt your stored API keys!

## Troubleshooting

### PostgreSQL not starting
```bash
docker compose logs postgres
```

Common issues:
- Port 5432 already in use
- Permissions on `data/db/` directory

### Can't connect to database
```bash
# Check backend can reach postgres
docker exec negotiator-pro-backend python -c "
from backend.database import db
import asyncio
print('Testing connection...')
result = asyncio.run(db.health_check())
print(f'Connected: {result}')
"
```

### API keys not working
```bash
# Verify keys are stored
curl http://localhost:8000/api/users/username/admin | jq '.has_openai_key, .has_anthropic_key'

# Should return: true, true (if you entered keys)
```

## What's Next?

1. ✅ Add source documents to `data/sources/`
2. ✅ Configure LLM backends in admin panel
3. ✅ Start chatting with negotiation questions
4. ✅ Switch between base (Ollama) and premium (GPT/Claude) models

## Full Documentation

- **Your Setup**: [MY_SETUP.md](MY_SETUP.md) - Your specific configuration
- **Docker Guide**: [DOCKER_SETUP.md](DOCKER_SETUP.md) - Complete Docker reference
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture diagrams
- **Verification**: [VERIFICATION_CHECKLIST.md](VERIFICATION_CHECKLIST.md) - Test everything works

---

**You're all set!** Ollama Cloud for base models, your premium keys for OpenAI/Claude. 🚀
