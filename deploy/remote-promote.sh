#!/usr/bin/env bash
#
# Remote-side steps for promoting the dev release to www.amfonica.com. Run by
# promote.sh as a FILE on the box (ssh host bash <this>), from the dev deploy
# dir, after promote.sh has rsynced .env.prod to $PROD_DIR/.env.
#
# Nothing is built here. Prod runs the image dev already ran
# (amfonica-backend:<sha>, retagged :prod) and the SPA bundle dev already
# served, so what was verified on dev is byte-for-byte what goes live.
#
# Args: MODE(promote|rollback) DEV_DIR PROD_DIR SERVER_IP EMAIL

set -euo pipefail
MODE="$1"; DEV_DIR="$2"; PROD_DIR="$3"; SERVER_IP="$4"; EMAIL="$5"

IMAGE_REPO="amfonica-backend"
PROJECT="amfonica-prod"
COMPOSE="docker-compose.deploy.yml"
PROD_PORT="8091"
NGINX_TEMPLATES="$DEV_DIR/deploy/nginx"
ACME_WEBROOT="/var/www/letsencrypt"

say() { printf '==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

cd "$PROD_DIR"
[ -f .env ] || die "$PROD_DIR/.env missing (promote.sh should have synced .env.prod)"
dc() { IMAGE_TAG=prod docker compose -p "$PROJECT" -f "$COMPOSE" "$@"; }

wait_healthy() {
  for _ in $(seq 1 45); do
    curl -fsS -m 3 "http://127.0.0.1:$PROD_PORT/api/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

# Swap back to the previous image and SPA. The database is not touched, which
# is why migrations must stay backward-compatible.
rollback() {
  docker image inspect "$IMAGE_REPO:prod-prev" >/dev/null 2>&1 \
    || die "no $IMAGE_REPO:prod-prev image; nothing to roll back to"
  say "Rolling back to $IMAGE_REPO:prod-prev"
  docker tag "$IMAGE_REPO:prod-prev" "$IMAGE_REPO:prod"
  [ -d public.prev ] && rsync -a --delete public.prev/ public/
  [ -f RELEASE.prev ] && cp -f RELEASE.prev RELEASE
  dc up -d --no-build --force-recreate
  wait_healthy || die "rolled-back container is not healthy either; check: docker compose -p $PROJECT logs backend"
  say "Rollback healthy: $(grep -E '^(IMAGE_TAG|GIT_SUBJECT)=' RELEASE 2>/dev/null | tr '\n' ' ')"
}

install_vhost() {  # name -> /etc/nginx/sites-{available,enabled}/<name>
  sudo cp "$NGINX_TEMPLATES/$1" "/etc/nginx/sites-available/$1"
  sudo ln -sf "/etc/nginx/sites-available/$1" "/etc/nginx/sites-enabled/$1"
}

# Issue a cert once, only when every name already resolves to this box.
ensure_cert() {  # cert-name domain...
  local cert_name="$1"; shift
  if sudo test -d "/etc/letsencrypt/live/$cert_name"; then
    echo "    cert $cert_name present"
    return 0
  fi
  local domain resolved
  for domain in "$@"; do
    resolved=$(getent hosts "$domain" | awk '{print $1}' | head -n1 || true)
    if [ "$resolved" != "$SERVER_IP" ]; then
      echo "    $domain -> ${resolved:-<none>}, need $SERVER_IP; skipping cert $cert_name for now"
      return 1
    fi
  done
  local domain_args=()
  for domain in "$@"; do domain_args+=(-d "$domain"); done
  sudo certbot certonly --webroot -w "$ACME_WEBROOT" --cert-name "$cert_name" "${domain_args[@]}" \
    --non-interactive --agree-tos -m "$EMAIL" --deploy-hook "systemctl reload nginx" \
    || { echo "    certbot failed for $cert_name; see /var/log/letsencrypt"; return 1; }
}

configure_nginx() {
  say "nginx + TLS"
  sudo mkdir -p "$ACME_WEBROOT"
  install_vhost amfonica-http.conf
  sudo nginx -t && sudo systemctl reload nginx

  local tls_conf
  ensure_cert www.amfonica.com www.amfonica.com amfonica.com && install_vhost www.amfonica.com.conf || true
  ensure_cert amfonica.ai amfonica.ai www.amfonica.ai && install_vhost amfonica.ai.conf || true

  if ! sudo nginx -t 2>/dev/null; then
    # Never leave nginx unable to reload: back the TLS vhosts out and report.
    for tls_conf in www.amfonica.com.conf amfonica.ai.conf; do
      sudo rm -f "/etc/nginx/sites-enabled/$tls_conf"
    done
    sudo nginx -t
    die "nginx rejected the TLS vhosts; removed them, HTTP-only config still active"
  fi
  sudo systemctl reload nginx
}

if [ "$MODE" = "rollback" ]; then
  rollback
  exit 0
fi
[ "$MODE" = "promote" ] || die "unknown mode $MODE"

# ---- what are we promoting? -----------------------------------------------
[ -f "$DEV_DIR/RELEASE" ] || die "$DEV_DIR/RELEASE missing; run ./deploy.sh first"
# shellcheck disable=SC1091
. "$DEV_DIR/RELEASE"
[ "${DIRTY:-1}" = "0" ] || die "dev release $IMAGE_TAG was built from uncommitted code; commit and redeploy dev first"
docker image inspect "$IMAGE_REPO:$IMAGE_TAG" >/dev/null 2>&1 || die "image $IMAGE_REPO:$IMAGE_TAG not found on the box"
say "Promoting $IMAGE_TAG: $GIT_SUBJECT"

# ---- files ------------------------------------------------------------------
# Only what the container mounts plus the scripts; backend code lives in the
# image. data/config and data/uploads are per-environment state, never copied.
say "Syncing release files from dev"
mkdir -p public data/uploads data/config data/vectorstore data-sources scripts migrations prompts deploy
for dir in scripts migrations prompts deploy; do
  rsync -a --delete --exclude '__pycache__/' "$DEV_DIR/$dir/" "$PROD_DIR/$dir/"
done
for file in config.json llm_backend_config.json docker-compose.deploy.yml; do
  cp -f "$DEV_DIR/$file" "$PROD_DIR/$file"
done
rsync -a --delete "$DEV_DIR/data-sources/" "$PROD_DIR/data-sources/"
rsync -a --delete "$DEV_DIR/data/vectorstore/" "$PROD_DIR/data/vectorstore/"

rsync -a --delete public/ public.prev/
[ -f RELEASE ] && cp -f RELEASE RELEASE.prev
# --delay-updates moves every new file into place at the end, so the swap from
# old bundle to new is close to atomic for anyone loading the page.
rsync -a --delete-after --delay-updates "$DEV_DIR/public/" "$PROD_DIR/public/"
chmod -R a+rX public
chmod 600 .env

# ---- image ------------------------------------------------------------------
had_previous=0
if docker image inspect "$IMAGE_REPO:prod" >/dev/null 2>&1; then
  docker tag "$IMAGE_REPO:prod" "$IMAGE_REPO:prod-prev"
  had_previous=1
fi
docker tag "$IMAGE_REPO:$IMAGE_TAG" "$IMAGE_REPO:prod"

# ---- database ---------------------------------------------------------------
say "Applying pending migrations (Neon prod)"
if ! dc run --rm --no-deps -T backend python3 - --apply < scripts/migrate.py; then
  # Nothing is running the new image yet, so undoing the tag and SPA is enough.
  if [ "$had_previous" = "1" ]; then
    docker tag "$IMAGE_REPO:prod-prev" "$IMAGE_REPO:prod"
    rsync -a --delete public.prev/ public/
  fi
  [ -f RELEASE.prev ] && cp -f RELEASE.prev RELEASE
  die "migration failed; prod left on its previous release"
fi

# ---- start + health gate ------------------------------------------------------
say "Starting prod container"
dc up -d --no-build --force-recreate
if ! wait_healthy; then
  echo "prod container failed its health check:" >&2
  dc logs --tail=40 backend >&2 || true
  if [ "$had_previous" = "1" ]; then
    rollback
    die "promotion of $IMAGE_TAG failed health check; rolled back"
  fi
  die "first promotion failed health check; no previous release to roll back to"
fi
say "Healthy on 127.0.0.1:$PROD_PORT"

say "Syncing admin password"
dc exec -T backend python3 - < scripts/sync_admin_password.py || echo "    admin sync failed (non-fatal)"

say "Source registry backfill (if empty)"
count=$(dc exec -T backend python3 -c "
import asyncio, os, asyncpg
async def main():
    conn = await asyncpg.connect(os.environ['DATABASE_URL'])
    print(await conn.fetchval('SELECT count(*) FROM source_documents'))
    await conn.close()
asyncio.run(main())
" </dev/null 2>/dev/null | tr -d '[:space:]')
if [ "${count:-0}" = "0" ]; then
  dc exec -T backend python3 - --apply < scripts/backfill_source_registry.py \
    && echo "    sources registered" || echo "    backfill failed (non-fatal)"
else
  echo "    ${count} sources already registered"
fi

configure_nginx

{
  cat "$DEV_DIR/RELEASE"
  printf 'PROMOTED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > RELEASE
say "Release recorded: $IMAGE_TAG"
