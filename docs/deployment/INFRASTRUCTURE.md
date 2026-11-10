# Amfonica Infrastructure

Where everything lives and how the parts connect.

## Platform overview

| Platform | Role | What lives there |
|---|---|---|
| **Neon** | Managed PostgreSQL | Users, sessions, chat history, source registry, rebuild jobs |
| **Bunny Storage** | Object storage | RAG corpus (PDFs), user uploads, FAISS vectorstores |
| **DigitalOcean VPS** | Web server | Docker backend container, nginx, built React SPA |

---

## Neon (PostgreSQL)

Two branches, same schema. Connection strings are in `.env` / `.env.dev`.

| Branch | Host | Database |
|---|---|---|
| Dev | `ep-calm-salad-ab7opg98-pooler.eu-west-2.aws.neon.tech` | `amfonica_dev` |
| Prod | `ep-muddy-block-abfecsa0-pooler.eu-west-2.aws.neon.tech` | `amfonica_prod` |

Set via `DATABASE_URL` in the env file. The backend connects with asyncpg; no local Postgres container in the deploy compose.

Migrations in `migrations/` run once manually against each branch (not on every deploy). The deploy script syncs the admin password on each deploy but does not run schema migrations.

**What's in Neon:**
- `users` — accounts, encrypted API keys, preferences
- `sessions` — JWT session tokens
- `chat_messages` — conversation history per user/session
- `negotiations` — negotiation records
- `partner_personas` — persona templates
- `source_documents` — corpus registry (filename, sha256, tags, metadata)
- `rebuild_jobs` — async vectorstore rebuild tracking

---

## Bunny Storage

Three storage zones, one pull zone (CDN). Configured via env vars in `backend/storage.py`.

| Zone | Env prefix | Purpose |
|---|---|---|
| `amfonica-data-sources` | `BUNNY_CORPUS_*` (legacy: `BUNNY_NET_*`) | RAG corpus PDFs |
| `amfonica-user-data` | `BUNNY_UPLOADS_*` | User-uploaded documents (Phase 4, not yet wired) |
| `amfonica-vectorstores` | `BUNNY_VECTORSTORES_*` | FAISS index files |

**Pull zone** (browser-facing CDN URLs): `BUNNY_NET_PULL_ZONE` — e.g. `data.amfonica.com`.

**Vectorstore path convention:** Each environment writes to its own prefix inside the vectorstores zone: `dev/`, `prod/`, `local/`. A laptop with `DEPLOY_ENV=local` never overwrites the server's index.

**How it's used:**
- On deploy or container start, the backend syncs the corpus down from Bunny to the local `data-sources/` bind mount.
- After a vectorstore rebuild, the backend pushes the new FAISS index to the vectorstores zone (only when `DEPLOY_ENV` is `dev` or `prod`, never `local`).
- If Bunny is not configured (env vars absent), `storage.py` degrades to local filesystem — local dev works without credentials.

**Env vars required for full operation:**

```
BUNNY_CORPUS_URL=https://uk.storage.bunnycdn.com/amfonica-data-sources
BUNNY_CORPUS_RO_PASSWORD=...
BUNNY_CORPUS_RW_PASSWORD=...

BUNNY_UPLOADS_URL=https://uk.storage.bunnycdn.com/amfonica-user-data
BUNNY_UPLOADS_RO_PASSWORD=...
BUNNY_UPLOADS_RW_PASSWORD=...

BUNNY_VECTORSTORES_URL=https://uk.storage.bunnycdn.com/amfonica-vectorstores
BUNNY_VECTORSTORES_RO_PASSWORD=...
BUNNY_VECTORSTORES_RW_PASSWORD=...

BUNNY_NET_PULL_ZONE=data.amfonica.com
```

---

## DigitalOcean VPS (web server)

**IP:** `134.209.189.154`  
**User:** `webdev`  
**Sites:** `/home/webdev/sites/amfonica.com/dev` (dev.amfonica.com) and `/home/webdev/sites/amfonica.com/www` (www.amfonica.com)

Dev and prod share this droplet. Each has its own directory, `.env`, container and port, and both run the same image (`amfonica-backend`, tagged by git sha, `:dev` and `:prod`).

### What runs on the box

| Component | How it runs | Port |
|---|---|---|
| Dev backend | Docker, compose project `dev` | `127.0.0.1:8090` (loopback only) |
| Prod backend | Docker, compose project `amfonica-prod` | `127.0.0.1:8091` (loopback only) |
| React SPA | Static files served by host nginx from each site's `public/` | none |
| nginx | Host process (not in Docker) | 80 / 443 |

No Postgres container on the box. Both backends connect to Neon via `DATABASE_URL`.

### nginx role

For each site, nginx serves the built SPA from `public/` and reverse-proxies `/api/` to that site's backend port. Both sites sit behind the cookie gate (`backend/api/routes/dev_auth.py`). Dev lets the home subnet `80.249.28.0/24` skip it; prod has no bypass while it is a private beta. `amfonica.com`, `amfonica.ai` and `www.amfonica.ai` 301 to `https://www.amfonica.com`.

Vhost templates live in `deploy/nginx/`. Dev's TLS is managed by `certbot --nginx`, and prod's by `certbot certonly --webroot` with static vhosts.

### Deploy flow

`./deploy.sh` builds and ships to dev. `./promote.sh` moves the exact dev release to prod. See `DEPLOY.md` for the full workflow, migrations and rollback.

---

## Request flow (runtime)

```
Browser
  └─> nginx (dev.amfonica.com / www.amfonica.com, port 443)
        ├─> /api/*  → proxy → 127.0.0.1:8090 (dev) or :8091 (prod) → FastAPI container
        │                         ├─> Neon (asyncpg, DATABASE_URL)
        │                         ├─> Bunny Storage (corpus/vectorstore sync)
        │                         └─> LLM APIs (OpenAI / Anthropic / Ollama)
        └─> /*      → static files from $DEPLOY_DIR/public/
```

---

## Environment summary

| Env | DATABASE_URL branch | Vectorstore prefix | Bunny writes? |
|---|---|---|---|
| `local` (dev machine) | `amfonica_dev` | `local/` | No |
| `dev` (VPS) | `amfonica_dev` | `dev/` | Yes |
| `prod` (VPS) | `amfonica_prod` | `prod/` | Yes |

`DEPLOY_ENV` controls the vectorstore prefix and whether Bunny writes are allowed.
