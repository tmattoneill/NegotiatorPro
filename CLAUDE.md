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

<!-- DEVCTX:START -->
## Project Context (auto-updated by devctx)

> **IMPORTANT:** When starting a new conversation, greet the user with a brief summary of the project context below — current focus, branch, and any active todos. Keep it to 2-3 sentences. Do not skip this greeting.

**Current Focus:** UI/UX pass on negotiation management (branch ux/negotiation-management) — fixing UI issues and UX bugs around creating, selecting, editing, and deleting negotiations

**Project:** Enhanced RAG-based negotiation guidance system with React frontend, FastAPI backend, multi-backend LLM support (OpenAI, Anthropic, Ollama), and document management.

**Branch:** `docs/cleanup-stale`
**Last Updated:** 12/06/2026, 07:17:39

### Active Todos
- [ ] [critical] Test vectorstore rebuild functionality end-to-end with the new bind mount fix to ensure no data loss (`main`)
- [ ] [high] Test the newly implemented admin LLM config tab and provider debug logging functionality (`main`)
- [ ] [high] Fix chat message typography: inconsistent fonts, sizes, and heading weights across markdown elements — needs a single coherent type scale applied to all rendered output (`main`)
- [ ] [high] Add context switching between negotiation and sales modes: NLP auto-detect intent from the query and switch mode automatically, inform the user which mode is active, allow explicit override ("what's a good negotiation strategy..." always works regardless of current mode) (`main`)
- [ ] [high] Create edit negotiation modal UI to utilize existing backend CRUD endpoints (`ux/negotiation-management`)
- [ ] [high] Implement delete negotiation confirmation dialog with backend integration (`ux/negotiation-management`)
- [ ] [high] Admin Sources & RAG: make rebuild auto-register unregistered files on disk. The rebuild/source-list assume every file in DATA_SOURCES_DIR has a source_documents row, but the original corpus was built by a CLI that never registered sources or stamped tags. Reconcile disk -> registry: auto-create rows deriving tags from top-level folder (negotiation/, sales/), and warn in UI when files lack rows. Stopgap: scripts/backfill_source_registry.py. (`main`)
- [ ] [high] Run the backfill_source_registry.py script to reconcile existing corpus files with the database registry (`main`)
- [ ] [high] Don't let the text preprocessor strip the negotiation briefing: get_advice preprocesses the whole enhanced_question (which includes the persona briefing block), degrading the >=240-char persona context we now require. Preprocess only the raw user question, not the briefing/file context. (`main`)
- [ ] [high] Test the complete deployment pipeline end-to-end using the new deploy.sh script (`main`)
- [ ] [high] Return to the original session plan: create and work on ux/negotiation-management branch to fix UI issues around creating, selecting, editing, and deleting negotiations (`main`)
- [ ] [high] Implement Phase 4: route chat-uploads to amfonica-user-data zone in Bunny Storage (`main`)
- [ ] [high] Create and switch to ux/negotiation-management branch to address the original session plan (`main`)
- [ ] [high] Push the 10 ahead commits on main to remote repository (`main`)
- [ ] [high] Verify the Neon PostgreSQL migration works correctly in all environments (`main`)
- [ ] [high] Merge ui/colour-scheme branch into main to integrate the new palette system (`ui/colour-scheme`)
- [ ] [high] Push ui/feedback-panel branch with all 15 commits to remote (`ui/feedback-panel`)
- [ ] [high] Test the complete feedback panel functionality end-to-end, especially PLEASE scoring and RAG citations (`ui/feedback-panel`)
- [ ] [high] tests/test_docker.py asserts EXPOSE 7860 and port 7860 (Gradio era) — will fail against the current 8000 backend / 5173 frontend. Update the port assertions. (`docs/cleanup-stale`)
- [ ] [medium] Verify admin login route works correctly with the new authentication changes (`main`)
- [ ] [medium] Replace 'YOU' label with the active Persona Name in chat messages (`main`)
- [ ] [medium] Show RAG source citations in right margin of chat messages, with hover tooltip revealing the relevant source text (`main`)
- [ ] [medium] Add copy button to chat prompts and responses — copies raw markdown to clipboard (`main`)
- [ ] [medium] PDF upload: use Anthropic native document API (base64 document content block) instead of pypdf text extraction when backend is Anthropic — handles scanned PDFs via server-side OCR, same as claude.ai. Fall back to pypdf for OpenAI/Ollama. May need to bypass langchain-anthropic and call Anthropic SDK directly for the multipart message. (`main`)
- [ ] [medium] Test the complete negotiation lifecycle: create, edit, delete with the new UI components (`ux/negotiation-management`)
- [ ] [medium] Verify the negotiation modal improvements work correctly across different screen sizes and browsers (`main`)
- [ ] [medium] Test chat message rendering fixes with various markdown content types including code blocks and lists (`main`)
- [ ] [medium] Fix per-model pricing units then compute real cost in metering: cost_per_1k_input/output in llm_backend_config.py actually hold per-1M values, so log_usage cost stays 0.0. Relabel to per-1M (or divide by 1000) and populate cost in EnhancedNegotiationRAG.get_advice usage logging. (`main`)
- [ ] [medium] Sales-mode retrieval returns 0 chunks for negotiation-themed queries (minority tag: top fetch_k candidates are all negotiation before the sales filter). Raise/auto-scale fetch_k or rebalance before SalesPro ships. Negotiation mode is unaffected. (`main`)
- [ ] [medium] Test the provider-aware prompt caching performance improvements across OpenAI, Anthropic, and Ollama backends (`main`)
- [ ] [medium] Verify the slimmed negotiation persona (~0.7k tokens) maintains quality while improving performance (`main`)
- [ ] [medium] Apply migrations 006 (partner_personas.negotiation_id) and 007 (partner_personas.cloned_from) to any existing non-dev database on deploy. They auto-run only on a fresh DB via docker-entrypoint-initdb; an existing DB needs them run manually (docker exec psql). (`main`)
- [ ] [medium] Build deploy.sh: deploy the app to dev (dev.amfonica.com). Rsync/build the Docker services (frontend, backend, Postgres) to the dev host and bring them up. Must run pending DB migrations (see todo to apply 006/007). This is stage 1 of the two-stage release in the meta CLAUDE.md. (`main`)
- [ ] [medium] Build promote.sh: promote dev -> prod, from dev.amfonica.com to www.amfonica.com (live). Ship the dev-verified build/images to the prod host, run DB migrations, and cut over with minimal downtime (and a rollback path). This is stage 2 of the two-stage release in the meta CLAUDE.md. Document the exact hosts, paths, and flow here once written. (`main`)
- [ ] [medium] Trim deploy.sh to remove corpus/vectorstore shipping once clean-boot hydrate stability is confirmed (`main`)
- [ ] [medium] Monitor Bunny Storage hydrate performance on next deployment to validate clean-boot stability (`main`)
- [ ] [medium] Push commit 1c33063 (HTTP Basic Auth nginx template) to remote (`main`)
- [ ] [medium] Commit or revert the modified usage_stats.json file (`main`)
- [ ] [medium] Test the new empty state UI and profile page improvements end-to-end (`main`)
- [ ] [medium] Test the new color scheme across all UI components to ensure consistent application (`ui/colour-scheme`)
- [ ] [medium] Merge ui/feedback-panel into main after testing to integrate the feedback systems (`ui/feedback-panel`)
- [ ] [medium] Create integration tests for the live gutter analytics (leverage, parties, vitals) (`ui/feedback-panel`)
- [ ] [medium] tests/test_model_config.py exercises the deprecated ModelConfig.get_model_kwargs_legacy path — update to the current logical-key get_model_kwargs pattern. (`docs/cleanup-stale`)
- [ ] [medium] Dead-code audit (with code verification, not date heuristics): runpod_llm.py is wired into llm_backend_config/rag_engine/routes.models but may never trigger in production; scripts/init_user_profile.py is now only referenced by deleted docs. Decide keep/remove for each. NOTE: the earlier stale-scan wrongly flagged config_loader.py and routes/config.py as dead — both are live, do not delete. (`docs/cleanup-stale`)
- [ ] [medium] Test rot: fix and then add the `unit` marker to the 4 currently-failing test files left unmarked — test_model_config.py (calls renamed ModelConfig.get_model_kwargs, now _legacy), test_prompt_renderer.py (1 assertion drifted from the slimmed negotiation persona), test_document_manager.py (4 fail), test_admin_config.py (2 fail + 1 error). Until fixed they run only under bare `pytest`, not `pytest -m unit`. (`docs/cleanup-stale`)
- [ ] [low] Document the new admin features and LLM configuration capabilities in user documentation (`main`)
- [ ] [low] Partner copy-on-write: private partner copies created before migration 007 (e.g. "Partner - Buyer") have a NULL cloned_from, so the modal's "also update the shared template" option silently no-ops for them. Decide whether to backfill cloned_from for pre-existing copies or accept it (re-cloning fixes it going forward). (`main`)
- [ ] [low] Document the palette system architecture and available color tokens for future development (`ui/colour-scheme`)
- [ ] [low] Document the PLEASE scoring system and gutter analytics in user documentation (`ui/feedback-panel`)

<!-- DEVCTX:END -->
