-- RAG Source Document Registry + Rebuild Jobs
-- Migration 004 — run on an existing database after 003_provider_preferences.sql

-- ============================================================================
-- SOURCE DOCUMENTS — RAG corpus registry with provenance and tagging
-- ============================================================================

CREATE TABLE IF NOT EXISTS source_documents (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename    VARCHAR(255) NOT NULL UNIQUE,
    sha256      VARCHAR(64)  NOT NULL UNIQUE,
    title       VARCHAR(255),
    author      VARCHAR(255),
    year        SMALLINT,
    tags        TEXT[]       NOT NULL DEFAULT '{}',
    enabled     BOOLEAN      NOT NULL DEFAULT TRUE,
    page_count  INTEGER,
    word_count  INTEGER,
    size_bytes  BIGINT       NOT NULL,
    extension   VARCHAR(20)  NOT NULL,
    added_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_indexed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_source_docs_enabled ON source_documents(enabled);
CREATE INDEX IF NOT EXISTS idx_source_docs_tags    ON source_documents USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_source_docs_added   ON source_documents(added_at DESC);

COMMENT ON TABLE  source_documents IS 'Registry of RAG corpus source files with provenance and topic tagging';
COMMENT ON COLUMN source_documents.sha256  IS 'SHA-256 hex digest of file content — used for dedupe';
COMMENT ON COLUMN source_documents.tags    IS 'Topic tags, e.g. {sales,negotiation}';
COMMENT ON COLUMN source_documents.enabled IS 'When false, file is excluded from next vectorstore rebuild';
COMMENT ON COLUMN source_documents.last_indexed_at IS 'Timestamp of last successful vectorstore rebuild that included this file';

-- ============================================================================
-- REBUILD JOBS — background vectorstore rebuild tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS rebuild_jobs (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    status          VARCHAR(20) NOT NULL DEFAULT 'pending',
    percent         SMALLINT    NOT NULL DEFAULT 0,
    current_file    VARCHAR(255),
    errors          JSONB       NOT NULL DEFAULT '[]',
    params          JSONB       NOT NULL DEFAULT '{}',
    started_at      TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at     TIMESTAMP,

    CONSTRAINT valid_job_status CHECK (status IN ('pending', 'running', 'done', 'error'))
);

CREATE INDEX IF NOT EXISTS idx_rebuild_jobs_status     ON rebuild_jobs(status);
CREATE INDEX IF NOT EXISTS idx_rebuild_jobs_started_at ON rebuild_jobs(started_at DESC);

COMMENT ON TABLE  rebuild_jobs IS 'Background vectorstore rebuild job tracking — polled by the admin UI';
COMMENT ON COLUMN rebuild_jobs.params IS 'JSON snapshot of rebuild parameters: chunk_size, chunk_overlap, embedding_model, tags_filter';
