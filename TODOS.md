# TODOS — Admin RAG Tools + Corpus Re-ingest

Companion to `PLAN_GRAPH-ADMIN.md`. Tick boxes as you go. Phases are sequential but each is independently shippable.

---

## Phase A — Source metadata layer

- [ ] Write migration `migrations/002_source_documents.sql` creating `source_documents` table (id, filename, sha256 UNIQUE, title, author, year, tags text[], enabled bool, page_count, word_count, size_bytes, extension, added_at, last_indexed_at)
- [ ] Apply migration via existing init scripts or `docker exec ... psql`
- [ ] Create `backend/source_metadata.py` with `SourceMetadataManager` class (async CRUD: create, get_by_filename, get_by_hash, list_all, list_enabled, update, delete)
- [ ] Add `compute_sha256(file_path)` helper to `backend/document_manager.py`
- [ ] Extend `DocumentManager.save_uploaded_file()` to compute hash, reject duplicates, create metadata row
- [ ] Extend `DocumentManager.delete_document()` to also delete metadata row
- [ ] Unit tests: `tests/test_source_metadata.py` (create/read/update/delete, unique-hash rejection)

## Phase B — Rebuild logic as a service

- [ ] Create `backend/vectorstore_builder.py` with:
    - [ ] `BuildResult` dataclass (chunks_created, files_processed, files_skipped, embedding_tokens, estimated_cost, errors, duration_seconds)
    - [ ] `build_index(sources_dir, target_dir, chunk_size, chunk_overlap, embedding_model, tags_filter, progress_callback)` returning `BuildResult`
    - [ ] Only includes sources where `enabled=true` and (tags_filter is None OR any tag in tags_filter ∈ source.tags)
    - [ ] Stamps each chunk's `metadata` with `source_file`, `sha256`, `tags`, `title`
    - [ ] Builds to `vectorstore_staging/`
    - [ ] `promote_staging(staging_dir, live_dir)` — atomic swap: rename live → `vectorstore_backup_<ts>/`, rename staging → live
- [ ] Refactor `scripts/rebuild_vectordb.py` into a thin CLI wrapper that prompts for args then calls `build_index()` + `promote_staging()`
- [ ] Verify legacy CLI flow still works (`docker exec ... python scripts/rebuild_vectordb.py`)
- [ ] Unit tests: `tests/test_vectorstore_builder.py` (with mock embedding model, tiny fixture corpus)

## Phase C — Admin RAG API routes

- [ ] Migration `migrations/003_rebuild_jobs.sql` — `rebuild_jobs` table (job_id UUID PK, status, percent, current_file, errors_json, started_at, finished_at, params_json)
- [ ] Create `backend/api/routes/admin_rag.py` with all endpoints (see plan section C)
- [ ] Pydantic models for requests/responses
- [ ] Implement `GET /api/admin/sources`
- [ ] Implement `POST /api/admin/sources/upload` (multipart, hash dedupe)
- [ ] Implement `PATCH /api/admin/sources/{filename}`
- [ ] Implement `DELETE /api/admin/sources/{filename}`
- [ ] Implement `GET /api/admin/vectorstore/status` (reads `embedding_config` + counts chunks from FAISS index)
- [ ] Implement `POST /api/admin/vectorstore/rebuild`:
    - [ ] Create `rebuild_jobs` row with status=`pending`
    - [ ] Add `BackgroundTasks` task calling `build_index()` with progress_callback that updates the row
    - [ ] On success: call `promote_staging()`, update `source_documents.last_indexed_at`, set status=`done`
    - [ ] On failure: status=`error`, capture exception
- [ ] Implement `GET /api/admin/vectorstore/jobs/{job_id}`
- [ ] Implement `POST /api/admin/vectorstore/test-query` (load staging or live FAISS, run similarity_search, optionally filter by tags)
- [ ] Register router in `backend/api/main.py`
- [ ] Integration tests: `tests/test_admin_rag_routes.py`

## Phase D — Frontend admin tab

- [ ] Add `'sources'` to `TabView` union in `AdminPanel.tsx`
- [ ] Add tab button to nav row (icon: 📚)
- [ ] Create `frontend/src/components/admin/SourcesTab.tsx` (table + upload dropzone)
- [ ] Create `frontend/src/components/admin/VectorstorePanel.tsx` (status card + rebuild form + progress + test-query box)
- [ ] Create `frontend/src/components/admin/SourceUploadDropzone.tsx` (drag-drop, multi-file, calls upload API per file)
- [ ] Extend `frontend/src/services/api.ts`:
    - [ ] `adminListSources()`
    - [ ] `adminUploadSource(file)`
    - [ ] `adminUpdateSource(filename, updates)`
    - [ ] `adminDeleteSource(filename)`
    - [ ] `adminGetVectorstoreStatus()`
    - [ ] `adminTriggerRebuild(params)`
    - [ ] `adminGetRebuildJob(jobId)`
    - [ ] `adminTestQuery(params)`
- [ ] Implement 2-second polling on active job (clear interval on done/error)
- [ ] Tag chip multi-select component (or reuse existing if any)
- [ ] Manual UI verification — load tab, edit a source's tags, trigger rebuild, watch progress, run test query

## Phase E — Re-ingest the expanded corpus

- [ ] Create `scripts/migrate_expanded_corpus.py`:
    - [ ] Source list with explicit tags map (filename → [tags])
    - [ ] For each source in `../data-sources/negotiaton-sources/`: compute hash, copy to `sources/` if not already in metadata, create metadata row
    - [ ] For each existing file in `sources/`: ensure metadata row exists (backfill)
    - [ ] Tag inference rules: Challenger / SPIN / Gap Selling / Jolt Effect / Mind for Sales → `sales`; Voss / Fisher / Ury / Bazerman → `negotiation`; Carnegie (Win Friends) / Power of a Positive No → both
    - [ ] Dry-run mode (`--dry-run`) that prints the plan without writing
- [ ] Run script: `docker exec negotiator-pro-backend python scripts/migrate_expanded_corpus.py --dry-run`
- [ ] Run script: `docker exec negotiator-pro-backend python scripts/migrate_expanded_corpus.py`
- [ ] Trigger rebuild from the new UI (Phase D)
- [ ] Verify test queries:
    - [ ] `"how do I anchor a price?"` — top results from Challenger / SPIN / Gap Selling
    - [ ] `"how should I respond when they say no?"` — top results from Voss / Ury / Power of Positive No
    - [ ] `"discovery questions for a first sales meeting"` — top results from SPIN / Gap Selling
- [ ] Note `.pptx` handling deferred — `The_Art_of_Negotiation_ChrisVoss.pptx` skipped (Voss content already present via TXT + PDF)

## Phase F — Tag-scoped retrieval

- [ ] Extend `backend/rag_engine.py` retrieval methods to accept `tags_filter: list[str] | None`
- [ ] Pass through to FAISS `similarity_search(filter=...)` using chunk metadata
- [ ] Add `mode: "sales" | "negotiation" | "auto"` to chat request model in `backend/api/routes/chat.py`
- [ ] Wire `mode` → `tags_filter` in chat handler (auto = None)
- [ ] Add optional mode selector to chat UI (top-of-conversation toggle: Auto / Sales / Negotiation) — or leave for follow-up
- [ ] Unit tests for tag-filtered retrieval

## Phase G — Housekeeping

- [ ] Diff each `.sync-conflict-*` file against its canonical counterpart in `backend/`, `backend/api/routes/`, `frontend/src/components/`
- [ ] Merge any unique content, then delete the conflict files
- [ ] Update `dev-docs/CLAUDE.md`:
    - [ ] Document Sources & RAG admin tab
    - [ ] Document `source_documents` schema
    - [ ] Document `tags_filter` / `mode` on retrieval
- [ ] Update root `README.md` "Adding New Sources & Rebuilding the RAG" section to reference the UI flow
- [ ] Commit and push when each phase is complete

## Out of scope (track separately)

- [ ] Rebrand strings to "Amfonica" (frontend copy, logos, page title)
- [ ] Named "skills" (prompt templates per use-case: pricing call, objection handling, discovery, contract review)
- [ ] LangGraph-style multi-step agents
- [ ] `.pptx` ingestion support
- [ ] Celery worker swap if `BackgroundTasks` proves insufficient
