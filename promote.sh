#!/usr/bin/env bash
#
# promote.sh — ship the release currently on dev.amfonica.com to www.amfonica.com.
#
# Run from the repo root:
#   ./promote.sh              show dev vs prod, confirm, promote
#   ./promote.sh --yes        promote without the confirmation prompt
#   ./promote.sh --rollback   put prod back on its previous image and SPA
#
# The workflow is local -> ./deploy.sh (dev) -> verify -> ./promote.sh (www).
# Prod never builds anything: it reruns dev's image (amfonica-backend:<sha>,
# retagged :prod) and dev's SPA bundle, so what you checked on dev is what
# goes live. Pending migrations run against Neon prod first. If the new
# container fails its health check, prod rolls back on its own.
#
# Rollback never reverts the database, so migrations must be additive or
# backward-compatible (expand now, contract in a later release).
#
# Prereqs: ssh to the box, and a filled-in .env.prod (git-ignored).

set -euo pipefail
cd "$(dirname "$0")"

REMOTE="${REMOTE:-webdev@134.209.189.154}"
SERVER_IP="${SERVER_IP:-134.209.189.154}"
SITE_ROOT="${SITE_ROOT:-/home/webdev/sites/amfonica.com}"
DEV_DIR="$SITE_ROOT/dev"
PROD_DIR="$SITE_ROOT/www"
EMAIL="${CERTBOT_EMAIL:-tmattoneill@gmail.com}"
COMPOSE="docker-compose.deploy.yml"

say() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

MODE="promote"; ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) ASSUME_YES=1 ;;
    --rollback) MODE="rollback" ;;
    -h|--help) sed -n '3,20p' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

confirm() {
  [ "$ASSUME_YES" = "1" ] && return 0
  read -r -p "$1 [y/N] " reply
  [ "$reply" = "y" ] || [ "$reply" = "Y" ] || die "aborted"
}

ssh -o ConnectTimeout=10 "$REMOTE" true || die "cannot ssh to $REMOTE"

release_summary() {  # dir -> one line, or a placeholder
  ssh "$REMOTE" "f='$1/RELEASE'; if [ -f \"\$f\" ]; then . \"\$f\"; echo \"\$IMAGE_TAG  \$GIT_SUBJECT  (\${PROMOTED_AT:-\$DEPLOYED_AT})\"; else echo '<none>'; fi"
}

if [ "$MODE" = "rollback" ]; then
  say "Prod now:      $(release_summary "$PROD_DIR")"
  say "Roll back to:  $(ssh "$REMOTE" "f='$PROD_DIR/RELEASE.prev'; if [ -f \"\$f\" ]; then . \"\$f\"; echo \"\$IMAGE_TAG  \$GIT_SUBJECT\"; else echo '<unknown>'; fi")"
  confirm "Roll www.amfonica.com back? The database is not touched."
  ssh "$REMOTE" bash "$DEV_DIR/deploy/remote-promote.sh" rollback "$DEV_DIR" "$PROD_DIR" "$SERVER_IP" "$EMAIL"
  exit 0
fi

# ---- preflight ----------------------------------------------------------------
say "Preflight"
[ -f .env.prod ] || die "missing .env.prod"
env_value() { grep -E "^$1=" .env.prod | tail -n1 | cut -d= -f2-; }
for key in DATABASE_URL JWT_SECRET_KEY ENCRYPTION_KEY ADMIN_PASSWORD DEV_AUTH_PASSWORD DEV_AUTH_SECRET; do
  [ -n "$(env_value "$key")" ] || die ".env.prod: $key is empty"
done
[ "$(env_value DEPLOY_ENV)" = "prod" ]      || die ".env.prod: DEPLOY_ENV must be prod"
[ "$(env_value API_HOST_PORT)" = "8091" ]   || die ".env.prod: API_HOST_PORT must be 8091 (dev owns 8090)"
case "$(env_value DATABASE_URL)" in
  *amfonica_dev*) die ".env.prod: DATABASE_URL points at the dev database" ;;
esac

DEV_RELEASE=$(release_summary "$DEV_DIR")
[ "$DEV_RELEASE" != "<none>" ] || die "dev has no RELEASE; run ./deploy.sh first"
ssh "$REMOTE" ". '$DEV_DIR/RELEASE'; [ \"\$DIRTY\" = 0 ]" \
  || die "dev release was built from uncommitted code; commit and run ./deploy.sh first"
DEV_SHA=$(ssh "$REMOTE" ". '$DEV_DIR/RELEASE'; echo \"\$GIT_SHA\"")

echo "dev  (to ship):  $DEV_RELEASE"
echo "prod (current):  $(release_summary "$PROD_DIR")"
confirm "Promote this release to www.amfonica.com?"

# ---- backup + env ---------------------------------------------------------------
say "Backing up current prod"
ssh "$REMOTE" "mkdir -p '$PROD_DIR'"
ssh "$REMOTE" bash "$DEV_DIR/deploy/backup.sh" prod "$PROD_DIR" "$COMPOSE"

say "Syncing .env.prod"
rsync -az --chmod=F600 .env.prod "$REMOTE:$PROD_DIR/.env"

# ---- promote on the box -------------------------------------------------------
say "Promoting on $REMOTE"
ssh "$REMOTE" bash "$DEV_DIR/deploy/remote-promote.sh" promote "$DEV_DIR" "$PROD_DIR" "$SERVER_IP" "$EMAIL"

# ---- tag the release in git -----------------------------------------------------
TAG="prod-$(date -u +%Y%m%d-%H%M)"
if git cat-file -e "$DEV_SHA^{commit}" 2>/dev/null; then
  git tag -a "$TAG" "$DEV_SHA" -m "Promoted to www.amfonica.com"
  git push origin "$TAG" >/dev/null 2>&1 && echo "tagged $TAG and pushed" || echo "tagged $TAG locally (push failed; run: git push origin $TAG)"
else
  echo "note: $DEV_SHA not in this clone; skipped git tag"
fi

# ---- smoke test from outside ------------------------------------------------------
say "Smoke test"
check() {  # url expected-status
  local code
  code=$(curl -sS -o /dev/null -m 15 -w '%{http_code}' "$1" 2>/dev/null || echo "000")
  printf '  %-36s %s %s\n' "$1" "$code" "$([ "$code" = "$2" ] && echo ok || echo "(expected $2)")"
}
check https://www.amfonica.com/ 302
check https://amfonica.com/ 301
check http://www.amfonica.com/ 301

say "Done. https://www.amfonica.com is on $DEV_RELEASE"
