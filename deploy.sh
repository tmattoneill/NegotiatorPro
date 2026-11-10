#!/usr/bin/env bash
#
# deploy.sh — stand up / update the Amfonica DEV site (dev.amfonica.com).
#
# Run from the repo root:  ./deploy.sh
#
# Syncs the project + corpus + prebuilt vectorstore to the remote host, builds
# the backend image tagged by git sha (amfonica-backend:<sha>, also :dev), runs
# pending DB migrations against Neon dev, starts the container, serves the built
# React app through the host nginx, and (once DNS points at the box) issues TLS
# via certbot. Containers use restart: unless-stopped, so they survive reboots.
#
# Writes $DEPLOY_DIR/RELEASE on success. promote.sh reads it to ship exactly
# this build to www. Only committed code deploys: ALLOW_DIRTY=1 overrides for
# dev, but promote.sh refuses a dirty release.
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
COMPOSE="docker-compose.deploy.yml"

# Paths whose contents end up on the box. Uncommitted changes here would make
# the release untraceable to a git sha.
SHIPPED_PATHS=(backend deploy scripts migrations prompts frontend Dockerfile requirements.txt
               config.json run-api.sh .dockerignore docker-compose.deploy.yml)

say() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- preflight --------------------------------------------------------------
say "Preflight"
[ -f .env.dev ]            || die "missing .env.dev (copy .env.example, fill secrets)"
[ -d data/vectorstore ]    || die "missing data/vectorstore (prebuilt index to ship)"
[ -d ../data-sources ]     || die "missing ../data-sources (RAG corpus)"
ssh -o ConnectTimeout=10 "$REMOTE" true || die "cannot ssh to $REMOTE"
echo "ok: local artifacts present, ssh reachable"

GIT_SHA=$(git rev-parse --short=12 HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
GIT_SUBJECT=$(git log -1 --format=%s)
if [ -n "$(git status --porcelain -- "${SHIPPED_PATHS[@]}")" ]; then
  [ "${ALLOW_DIRTY:-0}" = "1" ] || die "uncommitted changes in shipped paths; commit them, or ALLOW_DIRTY=1 for a dev-only deploy"
  DIRTY=1; IMAGE_TAG="${GIT_SHA}-dirty"
  echo "WARNING: deploying uncommitted changes as $IMAGE_TAG (promote.sh will refuse this release)"
else
  DIRTY=0; IMAGE_TAG="$GIT_SHA"
fi
[ "$GIT_BRANCH" = "main" ] || echo "note: deploying branch $GIT_BRANCH, not main"
echo "release: $IMAGE_TAG ($GIT_BRANCH) $GIT_SUBJECT"

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

# Release manifest, shell-sourceable. remote-deploy.sh promotes it from
# RELEASE.pending to RELEASE only once the new container is up.
RELEASE_FILE=$(mktemp)
trap 'rm -f "$RELEASE_FILE"' EXIT
{
  printf 'GIT_SHA=%q\n' "$GIT_SHA"
  printf 'IMAGE_TAG=%q\n' "$IMAGE_TAG"
  printf 'GIT_BRANCH=%q\n' "$GIT_BRANCH"
  printf 'GIT_SUBJECT=%q\n' "$GIT_SUBJECT"
  printf 'DIRTY=%q\n' "$DIRTY"
  printf 'DEPLOYED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$RELEASE_FILE"
rsync -az "$RELEASE_FILE" "$REMOTE:$DEPLOY_DIR/RELEASE.pending"
echo "ok: synced"

# ---- remote: build, run, serve, TLS ----------------------------------------
# Run the remote steps from a FILE on the box (rsynced above), not piped via
# stdin. Piping a heredoc to `bash -s` lets stdin-reading commands (docker
# compose exec -T) swallow the rest of the script.
say "Running remote deploy"
ssh "$REMOTE" bash "$DEPLOY_DIR/deploy/remote-deploy.sh" \
  "$DEPLOY_DIR" "$DOMAIN" "$APP_PORT" "$WEBROOT" "$EMAIL" "$SERVER_IP" "$IMAGE_TAG" "$COMPOSE"

say "Done. $IMAGE_TAG is live on https://$DOMAIN  (promote with ./promote.sh)"
