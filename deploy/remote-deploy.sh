#!/usr/bin/env bash
#
# Remote-side deploy steps for the Amfonica dev site. Rsynced to the box and run
# by deploy.sh as a FILE (ssh host bash <this>), NOT piped via stdin — otherwise
# stdin-reading commands like `docker compose exec -T` swallow the rest of the
# script. stdin is also redirected from /dev/null on those commands as a belt-
# and-braces guard.
#
# Args: DEPLOY_DIR DOMAIN APP_PORT WEBROOT EMAIL SERVER_IP IMAGE COMPOSE

set -euo pipefail
DEPLOY_DIR="$1"; DOMAIN="$2"; APP_PORT="$3"; WEBROOT="$4"; EMAIL="$5"; SERVER_IP="$6"; IMAGE="$7"; COMPOSE="$8"
cd "$DEPLOY_DIR"

# Load POSTGRES_USER/DB for the migration step (values come from the .env).
set -a; . ./.env; set +a
PGUSER="${POSTGRES_USER:-negotiatorpro}"; PGDB="${POSTGRES_DB:-negotiatorpro}"
dc() { docker compose -f "$COMPOSE" "$@"; }

echo "==> Building image & starting containers"
dc up -d --build

echo "==> Waiting for Postgres"
for _ in $(seq 1 30); do
  dc exec -T postgres pg_isready -U "$PGUSER" </dev/null >/dev/null 2>&1 && break
  sleep 2
done

# Fresh volumes run every migration via docker-entrypoint-initdb. This applies
# incremental migrations on re-deploys (skips 001 base schema, which initdb owns
# and is not idempotent). Guarded ALTERs (003-007) are safe to re-run.
echo "==> Applying incremental migrations"
for f in migrations/0*.sql; do
  base=$(basename "$f")
  case "$base" in 001_*) continue;; esac
  echo "    - $base"
  dc exec -T postgres psql -v ON_ERROR_STOP=0 -U "$PGUSER" -d "$PGDB" \
     -f "/docker-entrypoint-initdb.d/$base" </dev/null >/dev/null 2>&1 || true
done

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
