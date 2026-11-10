#!/usr/bin/env bash
#
# Remote-side deploy steps for the Amfonica dev site (dev.amfonica.com). Rsynced to the box and run
# by deploy.sh as a FILE (ssh host bash <this>), NOT piped via stdin — otherwise
# stdin-reading commands like `docker compose exec -T` swallow the rest of the
# script. stdin is also redirected from /dev/null on those commands as a belt-
# and-braces guard.
#
# Args: DEPLOY_DIR DOMAIN APP_PORT WEBROOT EMAIL SERVER_IP IMAGE_TAG COMPOSE

set -euo pipefail
DEPLOY_DIR="$1"; DOMAIN="$2"; APP_PORT="$3"; WEBROOT="$4"; EMAIL="$5"; SERVER_IP="$6"; IMAGE_TAG="$7"; COMPOSE="$8"
cd "$DEPLOY_DIR"

set -a; . ./.env; set +a
export IMAGE_TAG
IMAGE_REPO="amfonica-backend"
KEEP_IMAGES="${KEEP_IMAGES:-5}"
dc() { docker compose -f "$COMPOSE" "$@"; }

echo "==> Building image $IMAGE_REPO:$IMAGE_TAG"
dc build backend

# Migrate with the NEW image before the new container starts, so code never
# runs against a schema older than it expects. Migrations must stay additive,
# because the old container is still serving while this runs.
echo "==> Applying pending migrations (Neon dev)"
dc run --rm --no-deps -T backend python3 - --apply < scripts/migrate.py

echo "==> Starting containers"
dc up -d
docker tag "$IMAGE_REPO:$IMAGE_TAG" "$IMAGE_REPO:dev"

echo "==> Syncing admin password"
dc exec -T backend python3 - < scripts/sync_admin_password.py || echo "    admin sync failed (non-fatal)"

# Register the corpus in source_documents if the registry is empty.
echo "==> Source registry backfill (if empty)"
count=$(dc exec -T backend python3 -c "
import asyncio, os, asyncpg
async def main():
    conn = await asyncpg.connect(os.environ['DATABASE_URL'])
    n = await conn.fetchval('SELECT count(*) FROM source_documents')
    await conn.close()
    print(n)
asyncio.run(main())
" </dev/null 2>/dev/null | tr -d '[:space:]')
if [ "${count:-0}" = "0" ]; then
  dc exec -T backend python3 - --apply < scripts/backfill_source_registry.py \
    && echo "    sources registered" || echo "    backfill failed (non-fatal)"
else
  echo "    ${count} sources already registered; skipping"
fi

# The SPA is rsynced to $WEBROOT (public/) by deploy.sh; just make sure nginx
# (www-data) can read it.
echo "==> Preparing web root $WEBROOT"
chmod -R a+rX "$WEBROOT" 2>/dev/null || true

echo "==> Installing nginx vhost"
sed -e "s#__WEBROOT__#$WEBROOT#g" -e "s#__APP_PORT__#$APP_PORT#g" -e "s#__DOMAIN__#$DOMAIN#g" \
    "$DEPLOY_DIR/deploy/nginx/dev.amfonica.com.conf" \
  | sudo tee "/etc/nginx/sites-available/$DOMAIN.conf" >/dev/null
sudo ln -sf "/etc/nginx/sites-available/$DOMAIN.conf" "/etc/nginx/sites-enabled/$DOMAIN.conf"
sudo nginx -t && sudo systemctl reload nginx

echo "==> TLS"
RESOLVED=$(getent hosts "$DOMAIN" | awk '{print $1}' | head -n1 || true)
if [ "$RESOLVED" = "$SERVER_IP" ]; then
  sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL" --redirect \
    || echo "    certbot failed — check /var/log/letsencrypt"
else
  echo "    DNS for $DOMAIN -> ${RESOLVED:-<none>}; need $SERVER_IP. Skipping certbot."
  echo "    After DNS resolves, run on the box:"
  echo "      sudo certbot --nginx -d $DOMAIN --agree-tos -m $EMAIL --redirect"
fi

echo "==> Stack status"
dc ps

mv -f RELEASE.pending RELEASE
echo "==> Release recorded: $(grep -E '^(IMAGE_TAG|GIT_SUBJECT)=' RELEASE | tr '\n' ' ')"

# Disk on this shared box is tight. Keep the newest $KEEP_IMAGES sha tags;
# :dev, :prod and :prod-prev stay because they are separate tags, and removing
# a sha tag only untags an image that one of those still points at.
echo "==> Pruning old $IMAGE_REPO tags (keeping $KEEP_IMAGES)"
docker images "$IMAGE_REPO" --format '{{.CreatedAt}}\t{{.Tag}}' \
  | grep -vE $'\t(dev|prod|prod-prev)$' | sort -r | tail -n +$((KEEP_IMAGES + 1)) | cut -f2 \
  | while read -r old_tag; do docker rmi "$IMAGE_REPO:$old_tag" >/dev/null && echo "    removed $old_tag"; done
# One-off: the image name before sha tagging.
docker rmi amfonica-dev-backend:latest >/dev/null 2>&1 && echo "    removed legacy amfonica-dev-backend:latest" || true
