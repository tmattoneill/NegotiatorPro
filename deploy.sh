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

# ---- sync -------------------------------------------------------------------
say "Syncing to $REMOTE:$DEPLOY_DIR"
ssh "$REMOTE" "mkdir -p '$DEPLOY_DIR/data/uploads' '$DEPLOY_DIR/data/config' '$DEPLOY_DIR/data/vectorstore' '$DEPLOY_DIR/data-sources'"

# Project tree (build context + runtime config). Exclude heavy/secret/runtime.
rsync -az --delete \
  --exclude '.git' --exclude 'node_modules' --exclude 'frontend/node_modules' \
  --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
  --exclude 'data' --exclude 'public' --exclude '.env' --exclude '.env.dev' --exclude '.env.prod' \
  --exclude '.pytest_cache' --exclude 'htmlcov' --exclude '*.log' \
  ./ "$REMOTE:$DEPLOY_DIR/"

# Corpus (skip _archive — not ingested), prebuilt vectorstore, and the env file.
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
