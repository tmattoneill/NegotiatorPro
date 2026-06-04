#!/usr/bin/env bash
#
# backup.sh — snapshot the CURRENT deployment before a deploy overwrites it.
#
# Runs ON the remote box (invoked by deploy.sh / promote.sh before they sync).
# Writes ~/backups/amfonica_<env>_<timestamp>.tar.gz containing the whole
# deploy dir (backend code, built SPA in public/, data/ with the vectorstore +
# uploads + config, and .env) PLUS a pg_dump of the Postgres database (which
# lives in a Docker volume, not on disk, so a file copy alone would miss it).
#
# The data-sources corpus is deliberately NOT backed up: it is large, static,
# and reproducible from the repo (../data-sources on the dev machine).
#
# Keeps the newest BACKUP_KEEP (default 10) backups per env; older ones pruned.
#
# Args: ENV_NAME DEPLOY_DIR COMPOSE
#
# Restore (manual):
#   mkdir -p /tmp/restore && tar xzf ~/backups/amfonica_dev_<ts>.tar.gz -C /tmp/restore
#   # files: rsync /tmp/restore/ back into the deploy dir
#   # database: cat /tmp/restore/db_backup_<ts>.sql | \
#   #   docker compose -f <compose> exec -T postgres psql -U <user> -d <db>

set -euo pipefail
ENV_NAME="$1"; DEPLOY_DIR="$2"; COMPOSE="$3"
KEEP="${BACKUP_KEEP:-10}"
BACKUP_DIR="$HOME/backups"
mkdir -p "$BACKUP_DIR"

# First deploy: nothing on the box yet, so nothing to back up.
if [ ! -f "$DEPLOY_DIR/.env" ]; then
  echo "    no existing deployment (.env absent); skipping backup (first deploy)"
  exit 0
fi
cd "$DEPLOY_DIR"

TS=$(date +%Y%m%d-%H%M%S)
OUT="$BACKUP_DIR/amfonica_${ENV_NAME}_${TS}.tar.gz"

# DB creds from the box's own .env.
set -a; . ./.env; set +a
PGUSER="${POSTGRES_USER:-negotiatorpro}"; PGDB="${POSTGRES_DB:-negotiatorpro}"

# Dump the database into the deploy dir so the single tarball captures it too.
# stdin from /dev/null so `exec -T` never competes for this script's stdin.
DB_DUMP="db_backup_${TS}.sql"
if docker compose -f "$COMPOSE" exec -T postgres pg_isready -U "$PGUSER" </dev/null >/dev/null 2>&1; then
  if docker compose -f "$COMPOSE" exec -T postgres pg_dump -U "$PGUSER" "$PGDB" </dev/null > "$DB_DUMP"; then
    echo "    db dumped: $(du -h "$DB_DUMP" | cut -f1)"
  else
    echo "    WARNING: db dump failed; backing up files only"; rm -f "$DB_DUMP"
  fi
else
  echo "    postgres not running; backing up files only"
fi

# Tar the deploy dir (+ the db dump if written), minus the static corpus.
if tar czf "$OUT" -C "$DEPLOY_DIR" --exclude='./data-sources' .; then
  echo "    backup: $OUT ($(du -h "$OUT" | cut -f1))"
else
  echo "    ERROR: backup failed"; rm -f "$DB_DUMP" "$OUT"; exit 1
fi
rm -f "$DB_DUMP"

# Retention: keep the newest $KEEP backups for this env, prune the rest.
ls -1t "$BACKUP_DIR"/amfonica_${ENV_NAME}_*.tar.gz 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f
echo "    backups retained: $(ls -1 "$BACKUP_DIR"/amfonica_${ENV_NAME}_*.tar.gz 2>/dev/null | wc -l | tr -d ' ') (keep=$KEEP)"
