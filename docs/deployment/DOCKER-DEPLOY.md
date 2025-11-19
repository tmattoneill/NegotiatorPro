# Docker Quick Start Guide

Get NegotiatorPro (React + FastAPI) running in Docker in under 5 minutes.

**Note**: This guide covers the React frontend + FastAPI backend architecture. The system runs two Docker services: backend (port 8000) and frontend (port 5173).

## Prerequisites

- Docker installed (if not: `curl -fsSL https://get.docker.com | sh`)
- Docker Compose installed (usually comes with Docker)
- An OpenAI API key

## Step 1: Clone or Pull the Repo

```bash
# If you don't have it yet
git clone https://github.com/tmattoneill/NegotiatorPro.git
cd NegotiatorPro

# If you already have it
cd NegotiatorPro
git pull
```

## Step 2: Set Your API Key

```bash
# Copy the example file
cp .env.example .env

# Edit it and add your OpenAI API key
nano .env
# or
vi .env
# or use any text editor
```

Add your key:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

Save and exit.

## Step 3: Start Docker

```bash
docker compose up -d
```

That's it. Wait 30-60 seconds for it to build and start.

## Step 4: Check It's Running

```bash
# See logs for all services
docker compose logs -f

# Or check specific services
docker compose logs -f backend   # FastAPI backend
docker compose logs -f frontend  # React frontend
```

Look for:
- Backend: "Uvicorn running on http://0.0.0.0:8000"
- Frontend: "Local: http://localhost:5173/"

Press `Ctrl+C` to exit logs.

## Step 5: Open the App

Open your browser to:
```
Frontend: http://localhost:5173
Backend API: http://localhost:8000
API Docs: http://localhost:8000/docs
```

Done! 🎉

## Common Commands

```bash
# Stop the container
docker compose stop

# Start it again
docker compose start

# Stop and remove everything
docker compose down

# Rebuild after code changes
docker compose up -d --build

# See what's running
docker compose ps

# View logs
docker compose logs -f
```

## Troubleshooting

### Ports already in use?
```bash
# Change ports in docker-compose.yml
# Backend (default 8000):
ports:
  - "8001:8000"  # Use 8001 instead

# Frontend (default 5173):
ports:
  - "3000:5173"  # Use 3000 instead
```

### Container won't start?
```bash
# Check logs for errors
docker compose logs

# Check your .env file has the API key
cat .env
```

### Out of memory?
```bash
# Edit docker-compose.yml and increase memory:
# Under deploy.resources.limits:
memory: 4G  # Instead of 2G
```

### Need to rebuild from scratch?
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

## Default Admin Password

- Username: (none needed)
- Password: `admin123`

**⚠️ CHANGE THIS IMMEDIATELY!**

Access via FastAPI `/auth` endpoints. React admin UI is under development.

## That's It!

For detailed production deployment, security, backups, and advanced configuration, see [DEPLOYMENT.md](DEPLOYMENT.md).

---

**Need help?** Check the logs first: `docker compose logs -f`
