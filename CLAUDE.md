# CLAUDE.md

Guidance for Claude Code when working in NegotiatorPro. This file is intentionally short; the detail
lives in `docs/` and is read on demand. Product and brand context (Amfonica, the NegotiatorPro +
SalesPro split) is in the meta `../CLAUDE.md`.

## Overview

NegotiatorPro is the active tool in the **Amfonica** platform — a RAG system that gives sales and
negotiation guidance from a corpus of negotiation and sales books. React 18 + TypeScript frontend
(Vite), FastAPI backend, PostgreSQL (Neon in production, local/Docker in dev), FAISS vectorstore,
and multi-LLM support (OpenAI, Anthropic, Ollama, DeepSeek). Retrieval is tag-scoped by `mode`
(`sales` / `negotiation` / `auto`). Bunny Storage hydrates the corpus and vectorstore on boot. The
on-disk `sales-partner/` path is working-title legacy; the product is Amfonica.

## Commands

```bash
# Dev servers
./run-api.sh                       # FastAPI backend, port 8000
./run-frontend.sh                  # Vite frontend, port 5173

# Tests (markers: unit / integration / docker)
pytest                 # full suite (a few legacy tests are known-failing — see todos)
pytest -m unit         # 100 fast component tests, no DB or Docker (green gate)

# Full stack (backend + frontend + PostgreSQL)
docker compose up -d
docker compose logs -f

# Rebuild the vectorstore
python scripts/rebuild_vectordb.py
```

Environment: copy `.env.example` to `.env` and set at least `OPENAI_API_KEY`. The example file lists
every supported key (other LLM providers, Postgres, encryption, dev auth, Bunny Storage).

## Reference docs (read on demand)

- Architecture, data flow, RAG internals → `docs/ARCHITECTURE.md`
- LLM backends and model setup → `docs/LLM_BACKENDS.md` (Ollama cloud: `docs/features/OLLAMA_CLOUD_SETUP.md`)
- Config system (`config.json` vs `llm_backend_config.json`) → `docs/CONFIGURATION.md`
- Database and user profiles → `docs/USER_PROFILE_SETUP.md`
- Testing → `docs/TESTING.md`
- Deploy to dev (`./deploy.sh`), the Neon migration step, hosts/paths, rollback → `docs/deployment/DEPLOY.md` (authoritative)
- Infrastructure (Neon, Bunny, VPS — where everything lives) → `docs/deployment/INFRASTRUCTURE.md`
- Docker internals and local stack → `docs/deployment/DOCKER-DEPLOY.md`
- Admin panel and Sources & RAG → `docs/features/ADMIN_FEATURES.md`
- Intent-aware prompting → `docs/features/INTENT_AWARE_PROMPTING.md`
- Quick start → `docs/features/QUICKSTART.md`

## Conventions

Editorial and code-style rules are in `~/.claude/CLAUDE.md` (plain prose, no em dashes, active voice,
type hints, real error handling). Platform context for the Amfonica meta repo is in `../CLAUDE.md`.
Keep this file lean — put detail in `docs/`, not here.
