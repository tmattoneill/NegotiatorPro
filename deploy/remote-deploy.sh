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

set -a; . ./.env; set +a
dc() { docker compose -f "$COMPOSE" "$@"; }

echo "==> Building image & starting containers"
dc up -d --build

# Database is on Neon (external). No local postgres container to wait for.
# Migrations are applied once directly against Neon; not on every deploy.

# Sync the 'admin' user's password to ADMIN_PASSWORD (idempotent). migration 001
# seeds a placeholder hash that matches no known password; this makes login work
# without a manual reset, and only writes when the hash is out of sync.
echo "==> Syncing admin password"
dc exec -T backend python3 - <<'PY' || echo "    admin sync failed (non-fatal)"
import asyncio, os, bcrypt, asyncpg
async def main():
    pw = os.environ.get("ADMIN_PASSWORD")
    if not pw:
        print("    ADMIN_PASSWORD not set; skipping"); return
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    h = await conn.fetchval("SELECT password_hash FROM users WHERE username='admin'")
    if h and bcrypt.checkpw(pw.encode(), h.encode()):
        print("    admin password already in sync")
    else:
        new = bcrypt.hashpw(pw.encode(), bcrypt.gensalt(12)).decode()
        await conn.execute("UPDATE users SET password_hash=$1, is_active=TRUE WHERE username='admin'", new)
        print("    admin password synced to ADMIN_PASSWORD")
    await conn.close()
asyncio.run(main())
PY

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
