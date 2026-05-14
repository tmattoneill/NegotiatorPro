# Amfonica: Admin RAG Tools + Corpus Re-ingest

## Context

NegotiatorPro already exists in this directory as a fully built RAG system — FastAPI backend, React frontend, LangChain + FAISS, multi-LLM (OpenAI / Anthropic / Ollama / Gemma). The "Amfonica" framing is a rebrand + expansion. Two real gaps remain before it does what's wanted:

1. **No admin UI for source/vectorstore management.** `backend/document_manager.py` has the CRUD logic (upload, list, delete, info, stats) but is not wired to any FastAPI route. The CLI rebuild script (`scripts/rebuild_vectordb.py`) works but is interactive and needs `docker exec`. The admin panel has tabs for Users / Negotiations / Usage / LLM Config / Database — nothing for sources.

2. **The indexed corpus is negotiation-only.** `NegotiatorPro/sources/` has the negotiation classics (Voss, Fisher, Ury). The sales canon — Challenger Sale, SPIN, Gap Selling, Jolt Effect, Mind for Sales — sits in `../data-sources/negotiaton-sources/` but has never been ingested. So today the app can't actually advise on sales, only negotiation.

This plan ships **full corpus-governance admin tools** AND uses them to **re-ingest the expanded corpus** so the work ends with a working sales+negotiation RAG. Out of scope: rebranding strings to "Amfonica", building named skills, agentic flows — those come after the foundation is in place.

## Recommended approach

Seven phases. A–D are the admin tooling, E is the re-ingest, F is the retrieval hook for future "skills", G is housekeeping. Each phase is independently testable.

### Phase A — Source metadata layer (NEW)

Today there's no way to tag a source as sales / negotiation / both, no provenance fields, no dedupe-by-hash. Add a metadata layer.

- **Storage:** PostgreSQL table `source_documents` (consistent with existing user/conversation data, supports transactions for atomic rebuild swap, enables future joins to usage stats). Migration file `migrations/002_source_documents.sql`.
- **Schema:** `id`, `filename`, `sha256` (unique), `title`, `author`, `year`, `tags` (text[]: `sales`, `negotiation`, or both), `enabled` (bool, default true), `page_count`, `word_count`, `size_bytes`, `extension`, `added_at`, `last_indexed_at`.
- **NEW file:** `backend/source_metadata.py` — `SourceMetadataManager` class: CRUD against the table.
- Extend `backend/document_manager.py` to compute SHA256 on save and call the metadata manager. Reject uploads whose hash already exists.

### Phase B — Rebuild logic as a service (refactor)

Today `scripts/rebuild_vectordb.py` mixes I/O prompts with the actual rebuild logic. Refactor.

- **NEW file:** `backend/vectorstore_builder.py` — pure functions: `build_index(source_filter, chunk_size, chunk_overlap, embedding_model, target_dir) -> BuildResult`. Returns counts, errors, token estimate.
- Only ingest sources where `enabled=true`. Honour an optional `tags_filter` arg.
- Each chunk's LangChain `metadata` dict gets stamped with `source_file`, `sha256`, `tags`, `title` — hook for tag-scoped retrieval in Phase F.
- Build to `vectorstore_staging/`; on success, atomically swap with `vectorstore/` (rename old → `vectorstore_backup_<ts>/`, rename staging → `vectorstore/`). Failure = leave live store untouched.
- Keep the CLI script working — refactor it to be a thin wrapper that calls `build_index()`.

### Phase C — Admin RAG API routes (NEW)

**NEW file:** `backend/api/routes/admin_rag.py`. All endpoints guarded by existing `verify_admin` dependency.

- `GET    /api/admin/sources` — list with metadata + storage stats
- `POST   /api/admin/sources/upload` — multipart upload; auto-extracts page/word count; rejects duplicate hash
- `PATCH  /api/admin/sources/{filename}` — update tags, title, author, year, enabled
- `DELETE /api/admin/sources/{filename}` — delete file + metadata row
- `GET    /api/admin/vectorstore/status` — current model, chunk count, last_build, total size, dimensions
- `POST   /api/admin/vectorstore/rebuild` — body: `{chunk_size, chunk_overlap, embedding_model, tags_filter}`. Kicks off background work, returns `job_id`.
- `GET    /api/admin/vectorstore/jobs/{job_id}` — poll: `{status, percent, current_file, started_at, errors[]}`
- `POST   /api/admin/vectorstore/test-query` — body: `{query, k, target: "live"|"staging", tags_filter}`. Returns top-k chunks with scores + source.

**Background jobs:** start with **FastAPI `BackgroundTasks` + a `rebuild_jobs` table** (job_id, status, percent, current_file, errors_json, started_at, finished_at). UI polls the `/jobs/{id}` endpoint every 2s. Simpler than Celery; swap to Celery later if rebuilds outlive container restarts. No API change needed.

Register the router in `backend/api/main.py`.

### Phase D — Frontend admin tab

Add a 6th tab to `frontend/src/components/AdminPanel.tsx`: **"Sources & RAG"**.

**Layout — two stacked sections:**

1. **Sources** — table: title (editable), filename, tags (multi-select chips: sales / negotiation), enabled (toggle), size, pages, added. Per-row edit / delete. Drag-drop upload zone at top.
2. **Vectorstore** — status card (model, chunks, last build, size) + rebuild form (chunk size, overlap, embedding model dropdown sourced from `embedding_config`, tags filter) + progress bar + test-query box.

**NEW files:**
- `frontend/src/components/admin/SourcesTab.tsx`
- `frontend/src/components/admin/VectorstorePanel.tsx`
- `frontend/src/components/admin/SourceUploadDropzone.tsx`
- Extend `frontend/src/services/api.ts` with the new admin endpoints.

### Phase E — Re-ingest the expanded corpus

The substantive deliverable. After A–D ship:

1. **One-time migration script** `scripts/migrate_expanded_corpus.py`:
   - Copy missing sales books from `../data-sources/negotiaton-sources/` into `NegotiatorPro/sources/`: `challenger_sale.pdf`, `gap-selling.pdf`, `jolt_effect.pdf`, `mind-for-sales.pdf`, `spin_sellin.pdf`, `power-postive-no.pdf`, `win-friends-infulence.pdf`.
   - Skip if hash already in metadata table.
   - Backfill metadata rows for all existing sources in `NegotiatorPro/sources/` — title from filename, tags inferred (sales canon → `sales`; Voss/Fisher/Ury → `negotiation`; Carnegie / power-positive-no → both).
2. **Note on `.pptx`:** `The_Art_of_Negotiation_ChrisVoss.pptx` in `data-sources/` — current loaders don't handle `.pptx`. Skip for now; already have Voss content via TXT and `never-split-the-difference.pdf`.
3. Run rebuild from the new UI. Verify with test queries:
   - `"how do I anchor a price?"` → expect chunks from Challenger / SPIN / Gap Selling
   - `"how should I respond when they say no?"` → expect Voss / Ury / power-positive-no

### Phase F — Tag-scoped retrieval (foundation for future skills)

- Extend `backend/rag_engine.py`'s retrieval call to accept `tags_filter: list[str] | None`. Pass through to FAISS `similarity_search(filter=lambda md: ...)`.
- Extend the chat endpoint (`backend/api/routes/chat.py`) to accept optional `mode: "sales" | "negotiation" | "auto"`. `auto` = no filter (current behaviour).
- Future named skills (e.g. "Pricing Call Prep", "Objection Handling") just set their own `mode` + system prompt and reuse the same endpoint.

### Phase G — Housekeeping

- Delete or resolve the `.sync-conflict-*` duplicate files in `backend/`, `backend/api/routes/`, `frontend/src/components/` (Syncthing collision artefacts). Diff each pair first to make sure no unique content is lost.
- Update `dev-docs/CLAUDE.md` to document the new admin RAG tab, source metadata schema, and tag-scoped retrieval.
- Update root `README.md` "Adding New Sources & Rebuilding the RAG" section — point to UI instead of `docker exec`.

## Critical files

**Modify:**
- `backend/document_manager.py` — add SHA256 + metadata integration
- `backend/rag_engine.py` — add `tags_filter` to retrieval
- `backend/api/routes/chat.py` — accept `mode` param
- `backend/api/main.py` — register new router
- `frontend/src/components/AdminPanel.tsx` — add tab
- `frontend/src/services/api.ts` — add endpoints
- `scripts/rebuild_vectordb.py` — refactor to thin wrapper over `vectorstore_builder.py`

**Create:**
- `migrations/002_source_documents.sql` + `migrations/003_rebuild_jobs.sql`
- `backend/source_metadata.py`
- `backend/vectorstore_builder.py`
- `backend/api/routes/admin_rag.py`
- `scripts/migrate_expanded_corpus.py`
- `frontend/src/components/admin/SourcesTab.tsx`
- `frontend/src/components/admin/VectorstorePanel.tsx`
- `frontend/src/components/admin/SourceUploadDropzone.tsx`

**Reuse (no change needed):**
- `backend/embedding_config.py` — already lists embedding models with cost
- `backend/api/routes/admin.py` — pattern for `verify_admin` dep
- Existing FAISS save/load patterns

## Verification

End-to-end test, in order:

1. `docker compose up -d --build` — boots clean
2. Run new migrations — new tables created
3. `docker exec negotiator-pro-backend python scripts/migrate_expanded_corpus.py` — sales books copied in, metadata backfilled
4. Browse to `http://localhost:5173`, log in as admin
5. Sources & RAG tab loads, shows ~15 books with tags pre-populated
6. Trigger rebuild with defaults (text-embedding-3-small, 1000/200 chunks) — progress bar ticks through files, completes
7. Run test query `"how do I anchor in a price negotiation?"` from the test-query box — top results include Challenger / SPIN chunks
8. Open a chat session with `mode=negotiation` — verify replies cite Voss/Fisher only
9. Open a chat session with `mode=sales` — verify replies cite Challenger / Gap Selling / SPIN
10. Open a chat session with `mode=auto` (default) — should freely mix
11. `pytest` — existing suite still passes; add tests for `source_metadata.py`, `vectorstore_builder.py`, and the new routes

## Open questions for execution time (not blocking the plan)

- **Embedding model default:** stay on `text-embedding-3-small` (cheap, 1536-dim, good enough) or upgrade to `text-embedding-3-large` (3072-dim, ~6x cost)? Suggest staying on small for the rebuild; user can upgrade later from the UI now that it's easy.
- **Tag schema:** `sales` / `negotiation` covers the corpus. Column is `text[]` so finer-grained tags (`discovery`, `objection-handling`, `pricing`) can be added without migration.
- **Background job model:** starting with FastAPI `BackgroundTasks`. Swap in Celery if rebuilds outlive container restarts in practice.
