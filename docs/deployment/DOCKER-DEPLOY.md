# Docker Quick Start Guide (The Idiot's Guide)

Get NegotiatorPro running in Docker in under 5 minutes.

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
# See logs
docker compose logs -f

# Look for this line:
# "Running on local URL:  http://0.0.0.0:7860"
```

Press `Ctrl+C` to exit logs.

## Step 5: Open the App

Open your browser to:
```
http://localhost:7860
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

### Port 7860 already in use?
```bash
# Change port in docker-compose.yml
# Change this line:
ports:
  - "8080:7860"  # Use 8080 instead
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

Go to Admin Panel → Admin Settings → Change Password

## That's It!

For detailed production deployment, security, backups, and advanced configuration, see [DEPLOYMENT.md](DEPLOYMENT.md).

---

**Need help?** Check the logs first: `docker compose logs -f`
