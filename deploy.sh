#!/usr/bin/env bash
#
# deploy.sh — stand up / update the Amfonica DEV site (dev.amfonica.com).
#
# Run from the repo root:  ./deploy.sh
#
# Syncs the project + corpus + prebuilt vectorstore to the remote host, builds
# and starts the Docker stack (Postgres + FastAPI API), serves the built React
# app through the host nginx, and (once DNS points at the box) issues TLS via
# certbot. Containers use restart: unless-stopped, so they survive reboots.
#
# Prereqs on this machine: ssh access to the host, rsync, and a populated
# .env.dev (git-ignored). DNS: add an A record  dev.amfonica.com -> SERVER_IP.

set -euo pipefail
cd "$(dirname "$0")"

# ---- config -----------------------------------------------------------------
REMOTE="${REMOTE:-webdev@134.209.189.154}"
SERVER_IP="${SERVER_IP:-134.209.189.154}"
DEPLOY_DIR="${DEPLOY_DIR:-/home/webdev/sites/amfonica.com/dev}"
DOMAIN="${DOMAIN:-dev.amfonica.com}"
DEPLOY_ENV="${DEPLOY_ENV:-dev}"                   # used to name backups (amfonica_<env>_<ts>.tar.gz)
APP_PORT="${APP_PORT:-8090}"                      # host loopback port for the API
WEBROOT="${WEBROOT:-$DEPLOY_DIR/public}"          # nginx static root for the SPA (inside the deploy dir)
EMAIL="${CERTBOT_EMAIL:-tmattoneill@gmail.com}"   # Let's Encrypt account email
IMAGE="amfonica-dev-backend"
COMPOSE="docker-compose.deploy.yml"

say() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- preflight --------------------------------------------------------------
say "Preflight"
[ -f .env.dev ]            || die "missing .env.dev (copy .env.example, fill secrets)"
[ -d data/vectorstore ]    || die "missing data/vectorstore (prebuilt index to ship)"
[ -d ../data-sources ]     || die "missing ../data-sources (RAG corpus)"
ssh -o ConnectTimeout=10 "$REMOTE" true || die "cannot ssh to $REMOTE"
echo "ok: local artifacts present, ssh reachable"

# ---- backup the current deployment (before we touch anything) ---------------
# Snapshots the whole deploy dir + a DB dump to ~/backups on the box, so a bad
# deploy can be rolled back. Skips cleanly on a first deploy (nothing there).
say "Backing up current deployment on $REMOTE"
ssh "$REMOTE" "mkdir -p '$DEPLOY_DIR/deploy'"
rsync -az deploy/backup.sh "$REMOTE:$DEPLOY_DIR/deploy/backup.sh"
ssh "$REMOTE" bash "$DEPLOY_DIR/deploy/backup.sh" "$DEPLOY_ENV" "$DEPLOY_DIR" "$COMPOSE"

# ---- build the SPA (in Docker, off-box) -------------------------------------
# Built here and shipped as static files; the frontend source never goes to the
# box. Output lands in ./frontend/dist via the bind mount.
say "Building frontend (dist)"
# Anonymous volume on /app/node_modules shadows the host's, so npm ci installs
# linux deps inside the container and never clobbers the host's node_modules;
# only dist/ is written back through the bind mount.
docker run --rm -v "$PWD/frontend":/app -v /app/node_modules -w /app node:20-alpine \
  sh -lc 'npm ci && npm run build'
[ -d frontend/dist ] || die "frontend build produced no dist/"

# ---- sync -------------------------------------------------------------------
say "Syncing to $REMOTE:$DEPLOY_DIR"
ssh "$REMOTE" "mkdir -p '$DEPLOY_DIR/data/uploads' '$DEPLOY_DIR/data/config' '$DEPLOY_DIR/data/vectorstore' '$DEPLOY_DIR/data-sources' '$DEPLOY_DIR/public'"

# App: allowlist only the backend build context + runtime config. Use --delete
# (NOT --delete-excluded) so rsync only removes stale files INSIDE the synced
# code dirs; excluded top-level paths (data/, data-sources/, public/, .env) are
# left untouched. --delete-excluded would wipe those state/data dirs — the
# earlier `protect` filters did not actually prevent that, so they are gone.
rsync -az --delete \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='.DS_Store' \
  --include='/backend/***' --include='/deploy/***' --include='/scripts/***' \
  --include='/migrations/***' --include='/prompts/***' \
  --include='/Dockerfile' --include='/requirements.txt' \
  --include='/config.json' --include='/llm_backend_config.json' \
  --include='/run-api.sh' --include='/.dockerignore' \
  --include='/docker-compose.deploy.yml' \
  --exclude='*' \
  ./ "$REMOTE:$DEPLOY_DIR/"

# Built SPA -> nginx web root; corpus + prebuilt vectorstore; env file.
rsync -az --delete frontend/dist/      "$REMOTE:$DEPLOY_DIR/public/"
rsync -az --delete --exclude '_archive' --exclude '.DS_Store' ../data-sources/ "$REMOTE:$DEPLOY_DIR/data-sources/"
rsync -az --delete data/vectorstore/   "$REMOTE:$DEPLOY_DIR/data/vectorstore/"
rsync -az .env.dev                      "$REMOTE:$DEPLOY_DIR/.env"
echo "ok: synced"

# ---- remote: build, run, serve, TLS ----------------------------------------
# Run the remote steps from a FILE on the box (rsynced above), not piped via
# stdin. Piping a heredoc to `bash -s` lets stdin-reading commands (docker
# compose exec -T) swallow the rest of the script.
say "Running remote deploy"
ssh "$REMOTE" bash "$DEPLOY_DIR/deploy/remote-deploy.sh" \
  "$DEPLOY_DIR" "$DOMAIN" "$APP_PORT" "$WEBROOT" "$EMAIL" "$SERVER_IP" "$IMAGE" "$COMPOSE"

say "Done. https://$DOMAIN  (or http:// until TLS is issued)"
