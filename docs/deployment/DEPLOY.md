# DEPLOY.md: the Amfonica release workflow

This is the authoritative deploy and release guide for NegotiatorPro/Amfonica. If anything here
disagrees with `DEPLOYMENT.md` in this folder, this file wins. `DEPLOYMENT.md` is a stale Gradio-era
guide.

## The workflow

Code moves through three places, and each step has one command.

```
local        feature branch, docker compose up / ./run-api.sh + ./run-frontend.sh
  │          merge to main, push
  ▼
dev          ./deploy.sh    -> https://dev.amfonica.com   (Neon amfonica_dev)
  │          verify by hand
  ▼
prod         ./promote.sh   -> https://www.amfonica.com   (Neon amfonica_prod)
             ./promote.sh --rollback  if it goes wrong
```

The rule that holds it together is **build once, promote the same artefact**. `deploy.sh` builds the
backend image, tagged with the git sha (`amfonica-backend:<sha>`), and the SPA bundle. `promote.sh`
builds nothing. It retags that same image `:prod` and copies dev's SPA bundle, corpus and vectorstore
into the prod directory. What you clicked through on dev is byte for byte what goes live.

Two more rules follow from it:

- **Only committed code deploys.** `deploy.sh` refuses uncommitted changes in shipped paths and
  records the sha in `RELEASE`. `ALLOW_DIRTY=1 ./deploy.sh` overrides that for a quick dev test, but
  `promote.sh` refuses a dirty release.
- **Migrations must be additive or backward-compatible.** Both scripts migrate before the new
  container starts, while the old one is still serving, and rollback never reverts the database.
  To rename or drop something, expand in one release (add the new thing, write to both) and contract
  in a later one.

## Hosts, paths, ports

Dev and prod share one DigitalOcean droplet. Each has its own directory, container, port, `.env`,
database and nginx vhost.

| | Dev | Prod |
|---|---|---|
| URL | `https://dev.amfonica.com` | `https://www.amfonica.com` (apex, `amfonica.ai` and `www.amfonica.ai` 301 here) |
| Host (SSH) | `webdev@134.209.189.154` | same |
| Directory | `/home/webdev/sites/amfonica.com/dev` | `/home/webdev/sites/amfonica.com/www` |
| Compose project | `dev` (container `dev-backend-1`) | `amfonica-prod` (container `amfonica-prod-backend-1`) |
| Backend port | `127.0.0.1:8090` | `127.0.0.1:8091` |
| Image tag | `amfonica-backend:<sha>` and `:dev` | `amfonica-backend:prod` (previous: `:prod-prev`) |
| Env file (local, git-ignored) | `.env.dev` | `.env.prod` |
| Database | Neon `amfonica_dev` | Neon `amfonica_prod` |
| Vectorstore prefix in Bunny | `dev/` | `prod/` |
| nginx vhost (repo) | `deploy/nginx/dev.amfonica.com.conf` | `deploy/nginx/amfonica-http.conf`, `www.amfonica.com.conf`, `amfonica.ai.conf` |
| Access | cookie gate, home subnet bypasses | cookie gate (private beta), no bypass |
| Release manifest | `dev/RELEASE` | `www/RELEASE` (previous: `RELEASE.prev`) |
| Backups | `~/backups/amfonica_dev_<ts>.tar.gz` | `~/backups/amfonica_prod_<ts>.tar.gz` |

DNS for `amfonica.com` and `amfonica.ai` is at Name.com (account `tmattoneill`), nameservers
ns1 to ns4.name.com only.

## Prerequisites on the dev machine

- `.env.dev` and `.env.prod` in the repo root, both git-ignored. Every key `promote.sh` needs is
  checked in its preflight.
- `data/vectorstore/` (the prebuilt FAISS index) and `../data-sources/` (the corpus).
- SSH access to `webdev@134.209.189.154`, and Docker running locally (the SPA builds in
  `node:20-alpine`).

## Deploy to dev

```bash
git checkout main && git pull
./deploy.sh
```

In order, it:

1. Checks the local artefacts, SSH, and that shipped paths have no uncommitted changes.
2. Backs up the dev directory with `deploy/backup.sh`.
3. Builds the SPA off-box and rsyncs code, SPA, corpus, vectorstore and `.env.dev` to the box.
4. On the box (`deploy/remote-deploy.sh`):
   1. Builds `amfonica-backend:<sha>`.
   2. Runs `scripts/migrate.py --apply` against Neon dev, using the new image.
   3. Starts the container and tags the image `:dev`.
   4. Syncs the admin password and backfills the source registry if it's empty.
   5. Installs the vhost, runs certbot, and writes `RELEASE`.
   6. Prunes old sha tags, keeping 5. The disk is shared and tight.

Then check it yourself on `https://dev.amfonica.com`.

## Promote to prod

```bash
./promote.sh
```

It shows the dev release and the current prod release and asks before going on (`--yes` skips the
prompt). Then it:

1. Checks `.env.prod`. It refuses if a key is empty, if `DEPLOY_ENV` isn't `prod`, if the port isn't
   8091, or if `DATABASE_URL` points at the dev database.
2. Backs up the prod directory and syncs `.env.prod` to `www/.env` (mode 600).
3. On the box (`deploy/remote-promote.sh`):
   1. Copies dev's migrations, scripts, prompts, runtime config, corpus, vectorstore and SPA into
      `www/`, keeping the old SPA in `public.prev/`. `data/config` and `data/uploads` are never
      copied, because they are per-environment state.
   2. Tags the current `:prod` as `:prod-prev`, and the dev sha as `:prod`.
   3. Migrates Neon prod. If that fails it restores the tag and SPA and stops, and nothing new has
      run.
   4. Starts the container and polls `/api/health` for 90 seconds. If it isn't healthy, it rolls
      back by itself.
   5. Syncs the admin password, backfills the source registry, and installs the nginx vhosts.
      Certificates are issued on first run once DNS resolves.
4. Tags the commit `prod-<UTC yyyymmdd-HHMM>` in git and pushes the tag.
5. Smoke-tests the public URLs from your machine.

### Rollback

```bash
./promote.sh --rollback
```

This puts `:prod-prev` back as `:prod`, restores `public.prev/` and `RELEASE.prev`, and recreates
the container. The database is untouched. Roll the data back through the Neon dashboard
(point-in-time restore) only if a migration itself was the problem.

## Migrations

`scripts/migrate.py` applies `migrations/NNN_*.sql` in order and records each one in
`schema_migrations`. Each file runs in its own transaction, so a failure rolls that file back and
stops. The deploy scripts pipe it into the container, so you rarely run it by hand:

```bash
# Status of dev, read-only
ssh webdev@134.209.189.154 'docker exec -i dev-backend-1 python3 - --status' < scripts/migrate.py
```

To add a migration, create the next number (`012_short_name.sql`), make it additive, test it locally,
and deploy. `--apply` refuses to run on a database that has tables but no `schema_migrations`
history. That case calls for `--baseline-through NNN`, which records files as applied without
running them. Dev was baselined through 011 on 2026-11-10.

## Access control

Both sites sit behind the signed-cookie gate in `backend/api/routes/dev_auth.py`. nginx calls
`/api/dev-auth/check` with `auth_request` and sends anyone without a valid cookie to
`/dev-auth/login`. A correct password sets a 30-day cookie.

- Dev: password in `.env.dev` (`DEV_AUTH_PASSWORD`). Requests from `80.249.28.0/24` (home ISP)
  skip the gate.
- Prod: its own `DEV_AUTH_PASSWORD` and `DEV_AUTH_SECRET` in `.env.prod`, no bypass, and the page
  titled by `SITE_GATE_TITLE`. Signup is open behind the gate, so it keeps strangers off the LLM
  keys. To launch publicly, remove the `auth_request` and `error_page` lines from each protected
  location in `deploy/nginx/www.amfonica.com.conf` and promote again.

## TLS

Dev's cert is managed by `certbot --nginx`, which edits the dev vhost in place. Prod uses
`certbot certonly --webroot -w /var/www/letsencrypt` with a reload hook, so the prod vhosts in the
repo are static and never edited on the box. Prod has two certificates, `www.amfonica.com` (also
covering `amfonica.com`) and `amfonica.ai` (with `www.amfonica.ai`). `certbot.timer` renews both.

## Verify by hand

```bash
curl -fsS https://dev.amfonica.com/api/health
ssh webdev@134.209.189.154 'curl -fsS http://127.0.0.1:8091/api/health; cat ~/sites/amfonica.com/www/RELEASE'
curl -sI https://www.amfonica.com | head -3      # 302 to /dev-auth/login while the gate is on
ssh webdev@134.209.189.154 \
  'cd ~/sites/amfonica.com/www && docker compose -p amfonica-prod -f docker-compose.deploy.yml logs --tail=100 backend'
```

## Known gaps

- **Dev and prod share one droplet.** That's cheap and fine at beta scale. A noisy dev rebuild can
  slow prod, and disk is the tightest resource. Move prod to its own droplet before real traffic.
- **`deploy.sh` still ships the corpus and vectorstore every run.** Trim those once clean-boot
  hydrate from Bunny Storage is proven stable.
- **Signup is open.** It's fine behind the beta gate, but it needs invite codes or rate limits
  before the gate comes off.
