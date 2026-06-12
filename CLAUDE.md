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

## Architecture at a glance

- **Frontend** (`frontend/`): React 18 + TS, Zustand, Axios; chat surface plus a built admin panel
  (`AdminPanel.tsx`) with a Sources & RAG tab.
- **Backend** (`backend/`): FastAPI in `backend/api/` (endpoints in `routes/`); core RAG in
  `backend/rag_engine.py`; LLM backends in `backend/llm_backend_config.py`; storage layer in
  `backend/storage.py`.
- **Data**: PostgreSQL (users, conversations, partner personas, source registry), FAISS
  `vectorstore/`, source corpus in `sources/`, schema in `migrations/`.

Full detail → `docs/ARCHITECTURE.md`.

## Commands

```bash
# Dev servers
./run-api.sh                       # FastAPI backend, port 8000
./run-frontend.sh                  # Vite frontend, port 5173

# Tests (markers: unit, integration, docker)
pytest
pytest -m unit

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

<!-- DEVCTX:START -->
## Project Context (auto-updated by devctx)

> **IMPORTANT:** When starting a new conversation, greet the user with a brief summary of the project context below — current focus, branch, and any active todos. Keep it to 2-3 sentences. Do not skip this greeting.

**Current Focus:** UI/UX pass on negotiation management (branch ux/negotiation-management) — fixing UI issues and UX bugs around creating, selecting, editing, and deleting negotiations

**Project:** Enhanced RAG-based negotiation guidance system with React frontend, FastAPI backend, multi-backend LLM support (OpenAI, Anthropic, Ollama), and document management.

**Branch:** `ui/feedback-panel`
**Last Updated:** 11/06/2026, 10:54:51

<!-- DEVCTX:END -->
