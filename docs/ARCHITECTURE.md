# Architecture

How NegotiatorPro fits together. NegotiatorPro is the active tool in the **Amfonica** platform
(SalesPro is the planned sales counterpart — see the meta `../CLAUDE.md`). Both tools share this
RAG/LLM machinery; what differs is the corpus tag (`negotiation` vs `sales`) and the surface
prompts.

## Stack at a glance

- **Frontend** (`frontend/`): React 18 + TypeScript on Vite. Zustand for conversation/UI state,
  Axios API client. Components include the chat surface (Sidebar, ChatContainer, ChatMessage,
  ChatInput) and the built admin panel (`AdminPanel.tsx`, including the Sources & RAG tab).
- **Backend** (`backend/`): FastAPI in `backend/api/`, core RAG in `backend/rag_engine.py`.
- **Data**: PostgreSQL for user profiles, conversation history, partner personas, and the source
  registry (Neon in production, local/Docker Postgres in dev). FAISS vectorstore in `vectorstore/`.
  Source corpus in `sources/`. Bunny Storage hydrates corpus and vectorstore on boot
  (`backend/storage.py`).

## Backend core (`backend/`)

- **EnhancedNegotiationRAG** (`rag_engine.py`): core RAG system. Processes PDF/DOCX/TXT, creates
  embeddings, manages QA chains. Accepts `tags_filter` on retrieval and `mode`
  (`auto`/`sales`/`negotiation`) on advice generation.
- **ModelConfig** (`rag_engine.py`): middleware that handles model-specific parameters and creates
  LLM instances.
- **LLMBackendManager** (`llm_backend_config.py`): centralised management of LLM backends (OpenAI,
  Anthropic, Ollama, DeepSeek). `backend_manager` is a global singleton used throughout the app.
  See `LLM_BACKENDS.md`.
- **AdminConfig** (`admin_config.py`): admin authentication, sessions, system prompts, usage stats.
- **DocumentManager** (`document_manager.py`): file uploads, validation, SHA-256 deduplication.
- **SourceMetadataManager** (`source_metadata.py`): CRUD against the `source_documents` and
  `rebuild_jobs` tables. Tracks provenance (title, author, year, tags, sha256, page/word counts)
  for every file in `sources/`.
- **VectorstoreBuilder** (`vectorstore_builder.py`): programmatic rebuild. `build_index()` builds to
  a staging dir and stamps chunk metadata with source tags; `promote_staging()` atomically swaps
  staging into live. Used by the admin API and wrappable from the CLI.
- **EmbeddingConfig** (`embedding_config.py`): detects which embedding model built the current
  vectorstore and guarantees compatibility.
- **TextPreprocessor** (`text_preprocessor.py`): optional token optimisation. Strips email
  signatures, footers, forwarding headers, and legal boilerplate; context-aware stop-word removal
  that preserves negotiation-critical content (emotions, numbers, prices, commitments, deadlines,
  names). Can cut token usage substantially. Toggled via the admin interface.
- **PromptManager** (`prompt_manager.py`): system and user prompt templates, including the
  intent-aware mode prompts.
- **UserProfile** (`user_profile.py`) + **Database** (`database.py`): async PostgreSQL via asyncpg.
  See `USER_PROFILE_SETUP.md`.
- **Storage** (`storage.py`): Bunny Storage abstraction for corpus/vectorstore hydration and
  chat-upload routing.

## API layer (`backend/api/`)

- `main.py`: FastAPI entry point with lifespan events (DB connect on startup, disconnect on
  shutdown).
- `routes/`: endpoint groups — chat, auth, dev_auth, health, models, users, config, conversations,
  negotiations, personas, admin, admin_rag (the Sources & RAG / vectorstore endpoints). See the
  directory for the current set rather than relying on a list here.
- `models/`: Pydantic request/response models. `middleware/`: JWT auth and CORS.

## Document processing flow

1. Files (PDF, TXT, DOCX, DOC) in `sources/` are loaded via format-appropriate loaders.
2. Documents are chunked with RecursiveCharacterTextSplitter (1000 chars, 200 overlap by default).
3. Each chunk's LangChain metadata is stamped with `source_file`, `sha256`, `tags`, and `title`
   from the `source_documents` registry.
4. FAISS is built to `vectorstore_staging/`, then atomically promoted to `vectorstore/`.
5. The vectorstore is persisted with metadata for embedding-model compatibility.
6. Admins upload and rebuild via the Sources & RAG tab — no `docker exec` needed. See
   `features/ADMIN_FEATURES.md`.

The system loads the existing vectorstore on startup and regenerates only when explicitly
requested, so it does not reprocess documents on every boot.

## Tag-scoped retrieval

- Each source document carries `tags TEXT[]` — currently `sales`, `negotiation`, or both.
- The chat endpoint accepts `mode: "sales" | "negotiation" | "auto"` (default `auto`).
- `auto` applies no filter; `sales`/`negotiation` restrict the FAISS `similarity_search` to chunks
  matching that tag. `fetch_k = max(k*6, 30)` keeps enough raw candidates before filtering.

## Corpus registry schema

- **`source_documents`**: `id UUID PK`, `filename VARCHAR UNIQUE`, `sha256 VARCHAR(64) UNIQUE`,
  `title`, `author`, `year`, `tags TEXT[]` (GIN indexed), `enabled BOOL`, `page_count`,
  `word_count`, `size_bytes`, `extension`, `added_at`, `last_indexed_at`.
- **`rebuild_jobs`**: `id UUID PK`, `status VARCHAR(20)` (pending/running/done/error), `percent INT`,
  `current_file TEXT`, `errors JSONB`, `params JSONB`, `started_at`, `finished_at`.

Schema migrations live in `migrations/` (see `migrations/README.md`). On a fresh DB they run via
docker-entrypoint-initdb; an existing DB needs newer migrations run manually.

## Component interaction

- **Model creation flow**: React UI → FastAPI `/chat` → `ModelConfig.create_llm()` →
  `LLMBackendManager.create_llm_instance()` → LangChain ChatModel.
- **Auth**: JWT tokens for the React frontend (validated against PostgreSQL); UUID tokens for admin
  sessions (AdminConfig).
- **API-key resolution**: the system checks a user's encrypted stored keys first, then falls back to
  system keys.
- **Usage tracking**: LLM calls are logged with token counts.
- **Error handling**: backend failures fall back to the OpenAI default model with user notification.
- **Configuration**: dual-config split between `config.json` and `llm_backend_config.json` — see
  `CONFIGURATION.md`.

## Partner personas and intent-aware prompting

- **Partner personas**: negotiation counterparts modelled as reusable personas with CRUD
  (`backend/api/routes/personas.py`), optionally scoped to a negotiation and version-tracked
  (`cloned_from`). Migrations 006-007 add scope and copy-on-write versioning.
- **Intent-aware prompting**: queries are classified and routed to mode-specific prompts. See
  `features/INTENT_AWARE_PROMPTING.md`. Migration 008 persists the detected intent.

## PLEASE response framework

The negotiation system prompt asks for a structured response: a negotiation breakdown, calibrated
questions, draft responses, scenario planning, and a self-assessment score across Polite, Logical,
Empathetic, Assertive, Strategic, Engaging.

## Directory map

```
frontend/         React 18 + TS (Vite). src/{components,store,services,types}
backend/
  api/            FastAPI: main.py, routes/, models/, middleware/
  rag_engine.py   EnhancedNegotiationRAG + ModelConfig
  *.py            llm_backend_config, admin_config, document_manager,
                  source_metadata, vectorstore_builder, embedding_config,
                  text_preprocessor, prompt_manager, database, user_profile,
                  storage, config_loader
sources/          RAG corpus (PDF/TXT/DOCX/DOC)
vectorstore/      Persisted FAISS index (auto-generated)
uploads/          Temp upload storage
migrations/       PostgreSQL schema (+ README.md)
scripts/          Utilities (rebuild_vectordb, init_user_profile, backfill_source_registry, …)
deploy/           Remote deploy + nginx configs
tests/            Pytest suite (markers: unit, integration, docker)
docs/             This documentation tree
```
