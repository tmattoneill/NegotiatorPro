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
**Deploy path:** `/home/webdev/sites/amfonica.com/dev`  
**Domain:** `dev.amfonica.com`

### What runs on the box

| Component | How it runs | Port |
|---|---|---|
| FastAPI backend | Docker container (`amfonica-dev-backend`) | `127.0.0.1:8090` (loopback only) |
| React SPA | Static files served by host nginx | — |
| nginx | Host process (not in Docker) | 80 / 443 |

No Postgres container on the box — the backend connects to Neon directly via `DATABASE_URL`.

### nginx role

nginx handles two jobs:

1. Serves the built React SPA from `$DEPLOY_DIR/public/` for all non-`/api/` paths.
2. Reverse-proxies `/api/` to the backend container at `127.0.0.1:8090`.

TLS is issued by certbot (`--nginx` plugin). The vhost template is at `deploy/nginx/dev.amfonica.com.conf`; `deploy.sh` fills in `__WEBROOT__`, `__APP_PORT__`, and `__DOMAIN__` before installing it.

### Access control on dev

All routes are behind a cookie-based pre-auth gate:
- Requests from `80.249.28.0/24` (Matt's home ISP) bypass the check entirely.
- Everyone else hits a login form at `/dev-auth/login` on first visit; a 30-day signed cookie grants access after that.
- Route `backend/api/routes/dev_auth.py`, nginx sub-request at `/internal-dev-auth-check`.

### Deploy flow

```
./deploy.sh   (run from repo root, requires .env.dev)
  │
  ├── build React SPA locally (Docker, node:20-alpine)
  ├── rsync backend code + built SPA + corpus + vectorstore to VPS
  ├── rsync .env.dev -> VPS:.env
  │
  └── ssh: run deploy/remote-deploy.sh on the box
        ├── docker compose up -d --build   (backend container)
        ├── sync admin password to Neon
        ├── backfill source_documents registry if empty
        ├── install nginx vhost from template
        └── certbot TLS (if DNS resolves)
```

The React SPA source never goes to the box; only `frontend/dist/` is shipped.

---

## Request flow (runtime)

```
Browser
  └─> nginx (dev.amfonica.com, port 443)
        ├─> /api/*  → proxy → 127.0.0.1:8090 → FastAPI container
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
| `prod` (VPS, future) | `amfonica_prod` | `prod/` | Yes |

`DEPLOY_ENV` controls the vectorstore prefix and whether Bunny writes are allowed.
