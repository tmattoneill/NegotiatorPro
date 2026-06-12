# DEPLOY.md — the Amfonica deploy recipe

This is the authoritative, current deploy workflow for NegotiatorPro/Amfonica. If anything here
disagrees with `DEPLOYMENT.md` in this folder, this file wins. `DEPLOYMENT.md` is a stale generic
Docker guide from the Gradio era and describes a system that no longer exists.

The whole flow runs from one script, `./deploy.sh`, executed from the repo root on the developer
machine. It builds the frontend off-box, ships code plus the corpus plus the prebuilt vectorstore to
the dev host, builds and starts the backend container there, serves the SPA through the host nginx,
and issues TLS. The one thing it does **not** do is run database migrations. That is the main
gotcha, covered below.

## The shape of it

There is no Postgres container in production. The database is **Neon** (managed Postgres, external).
The backend reaches it over `DATABASE_URL` in `.env`. So a deploy ships application code and static
data, but the schema lives on Neon and has its own lifecycle.

```
developer machine                          dev host (134.209.189.154)
─────────────────                          ──────────────────────────
./deploy.sh
  ├─ build frontend (node:20 docker) ──►   rsync dist/  ──►  $DEPLOY_DIR/public/   ──► host nginx
  ├─ rsync backend + corpus + vectorstore ─────────────►   $DEPLOY_DIR/
  └─ ssh: remote-deploy.sh
         ├─ docker compose up -d --build  (backend only)
         ├─ sync admin password
         ├─ backfill source registry (if empty)
         ├─ install nginx vhost
         └─ certbot TLS
                                                              │
                                           backend ──DATABASE_URL──► Neon (external Postgres)
```

## Hosts, paths, ports

| Thing | Value |
|---|---|
| Dev host (SSH) | `webdev@134.209.189.154` |
| Deploy dir | `/home/webdev/sites/amfonica.com/dev` |
| Domain | `dev.amfonica.com` |
| Backend container port | `127.0.0.1:8090` → container `8000` (loopback only; host nginx proxies `/api`) |
| SPA web root | `$DEPLOY_DIR/public` (served by host nginx) |
| Backend image | `amfonica-dev-backend` |
| Compose file | `docker-compose.deploy.yml` (backend service only, no Postgres) |
| Database | Neon, external. `DATABASE_URL` in `.env.dev` |
| Backups | `~/backups/amfonica_dev_<ts>.tar.gz` on the host, keep newest 10 |

Prod (`www.amfonica.com`) is not wired up yet. `promote.sh` (dev → prod) is still to be written.

## Prerequisites on the dev machine

`deploy.sh` preflight checks these and dies early if any are missing:

- `.env.dev` in the repo root (git-ignored, holds all secrets including `DATABASE_URL` and
  `ADMIN_PASSWORD`). Copy `.env.example`, fill it in.
- `data/vectorstore/` (the prebuilt FAISS index: `index.faiss`, `index.pkl`, `metadata.json`).
- `../data-sources/` (the RAG corpus; `_archive/` is excluded from the rsync).
- SSH access to `webdev@134.209.189.154`.
- Docker running locally (the frontend builds in a throwaway `node:20-alpine` container).
- `psql` on PATH (for the migration step below).

## The recipe

### 1. Land your code on `main`

Deploy ships whatever is checked out. Merge your feature branch to `main`, push, and deploy from a
clean tree.

```bash
git checkout main
git merge <feature-branch>      # fast-forward where possible
git push
```

### 2. Apply pending migrations to Neon — DO NOT SKIP

`deploy.sh` does not touch the database. `remote-deploy.sh` says so in a comment: "Migrations are
applied once directly against Neon; not on every deploy." A fresh Docker Postgres would auto-run
`migrations/` via `docker-entrypoint-initdb`, but Neon is a long-lived external database, so new
migration files never get applied automatically. If your branch added migrations and you skip this
step, the deploy succeeds and the app then 500s the moment it touches a missing column.

Every migration in this repo is written idempotent (`ADD COLUMN IF NOT EXISTS`, `CREATE INDEX IF NOT
EXISTS`), so re-running a range is safe. Apply anything newer than what Neon has seen:

```bash
# Pull DATABASE_URL out of .env.dev without echoing it.
DATABASE_URL=$(grep '^DATABASE_URL=' .env.dev | cut -d= -f2- | tr -d '"')

# Apply a range (adjust the list to the migrations your branch added).
for m in 009 010 011; do
  f=$(ls migrations/${m}_*.sql)
  echo "applying $f"
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$f"
done
```

When in doubt, replay from the last known-applied number through HEAD; idempotency makes that a
no-op for already-applied files. There is no migration-tracking table yet, so this is on you to
track. (A `schema_migrations` ledger is worth adding; see "Known gaps".)

### 3. Run the deploy

```bash
./deploy.sh
```

It will, in order:

1. **Preflight** — check local artifacts and SSH reachability.
2. **Backup** — run `deploy/backup.sh` on the host, tarring the current deploy dir (minus the static
   corpus) to `~/backups/`. Neon itself is covered by point-in-time restore, so the DB is not in the
   tarball. First deploy skips cleanly.
3. **Build frontend** — `npm ci && npm run build` inside `node:20-alpine`, output to
   `frontend/dist/` via a bind mount. The host's `node_modules` is shadowed so it never gets
   clobbered.
4. **Sync** — rsync an allowlist of backend code + runtime config, then the built SPA, the corpus,
   the vectorstore, and `.env.dev` → `.env`. Uses `--delete` (never `--delete-excluded`), so state
   dirs (`data/`, `data-sources/`, `public/`, `.env`) survive.
5. **Remote deploy** — `remote-deploy.sh` builds the image, brings the backend up, syncs the admin
   password to `ADMIN_PASSWORD`, backfills the source registry if empty, installs the nginx vhost,
   and runs certbot.

A clean run ends with `dc ps` showing `dev-backend-1` healthy and a `Done. https://dev.amfonica.com`
line.

### 4. Verify

```bash
ssh webdev@134.209.189.154 \
  'cd /home/webdev/sites/amfonica.com/dev && docker compose -f docker-compose.deploy.yml ps'
ssh webdev@134.209.189.154 'curl -fsS http://127.0.0.1:8090/api/health'
curl -fsS https://dev.amfonica.com/api/health
```

If a new feature reads a column you migrated in step 2, exercise it. Check backend logs on a 500:

```bash
ssh webdev@134.209.189.154 \
  'cd /home/webdev/sites/amfonica.com/dev && docker compose -f docker-compose.deploy.yml logs --tail=100 backend'
```

## Dev site access control

`dev.amfonica.com` sits behind HTTP Basic Auth (`webdev` / `amfonica123`). Requests from
`80.249.28.0/24` (Matt's home ISP subnet) bypass the challenge. The vhost template lives at
`deploy/nginx/dev.amfonica.com.conf`; the htpasswd file is `/etc/nginx/.htpasswd-dev` on the host,
created once and not managed by `deploy.sh`.

## Rollback

Files: untar the latest backup over the deploy dir.

```bash
ssh webdev@134.209.189.154
mkdir -p /tmp/restore
tar xzf ~/backups/amfonica_dev_<ts>.tar.gz -C /tmp/restore
rsync -a /tmp/restore/ /home/webdev/sites/amfonica.com/dev/
cd /home/webdev/sites/amfonica.com/dev && docker compose -f docker-compose.deploy.yml up -d --build
```

Database: roll back through the **Neon dashboard** (branch restore / point-in-time restore). The
backup tarball does not contain the database.

## Known gaps and gotchas

- **No migration runner.** Step 2 is manual and there is no `schema_migrations` ledger. Easy to
  forget on a deploy that adds columns. This is the single most likely way to break a deploy.
- **Orphan `dev-postgres-1` container.** The host still runs a `postgres:15-alpine` container from
  before the Neon migration. It is unused (backend talks to Neon) but `docker compose` warns about
  it on every deploy. Remove it once you have confirmed nothing depends on it:
  ```bash
  ssh webdev@134.209.189.154 \
    'cd /home/webdev/sites/amfonica.com/dev && docker compose -f docker-compose.deploy.yml up -d --remove-orphans && docker rm -f dev-postgres-1'
  ```
- **`deploy.sh` still ships the corpus and vectorstore every run.** Once clean-boot hydrate from
  Bunny Storage is proven stable, trim those rsync steps (they are slow and large).
- **`promote.sh` does not exist.** Prod (`www.amfonica.com`) is unbuilt. When it lands, document the
  prod host, paths, and cutover flow here.
- **`DEPLOYMENT.md` is stale.** Gradio-era generic guide. Ignore it for the dev/prod flow.
