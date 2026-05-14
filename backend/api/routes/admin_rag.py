"""
Admin RAG API Routes

Source document management and vectorstore administration.
All endpoints require admin role (via verify_admin dependency).
"""
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from pydantic import BaseModel

from backend.document_manager import DocumentManager
from backend.embedding_config import EmbeddingConfig
from backend.source_metadata import SourceMetadataManager
from backend.vectorstore_builder import BuildResult, build_index, promote_staging
from ..middleware.auth import get_current_user
from .admin import verify_admin  # reuse existing admin auth dependency

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin-rag"])

# Paths. DATA_SOURCES_DIR points at the canonical corpus, which lives outside
# the project dir (../data-sources by default, mounted at /app/data-sources in
# Docker). Subdirs sales/ and negotiation/ organise content by topic.
SOURCES_DIR = os.getenv("DATA_SOURCES_DIR", "../data-sources")
UPLOAD_DIR = "uploads"
LIVE_VECTORSTORE = "vectorstore"
STAGING_VECTORSTORE = "vectorstore_staging"

_doc_manager = DocumentManager(sources_dir=SOURCES_DIR, upload_dir=UPLOAD_DIR)
_embed_config = EmbeddingConfig()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SourceResponse(BaseModel):
    id: str
    filename: str
    sha256: str
    title: Optional[str]
    author: Optional[str]
    year: Optional[int]
    tags: List[str]
    enabled: bool
    page_count: Optional[int]
    word_count: Optional[int]
    size_bytes: int
    extension: str
    added_at: str
    last_indexed_at: Optional[str]


class SourceUpdate(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[int] = None
    tags: Optional[List[str]] = None
    enabled: Optional[bool] = None


class VectorstoreStatus(BaseModel):
    embedding_model: Optional[str]
    chunks_count: Optional[int]
    last_build: Optional[str]
    size_bytes: int
    dimensions: Optional[int]
    metadata_path: str
    index_exists: bool


class RebuildRequest(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "text-embedding-3-small"
    tags_filter: Optional[List[str]] = None


class RebuildJobResponse(BaseModel):
    id: str
    status: str
    percent: int
    current_file: Optional[str]
    errors: List[str]
    params: dict
    started_at: str
    finished_at: Optional[str]


class TestQueryRequest(BaseModel):
    query: str
    k: int = 5
    target: str = "live"   # "live" or "staging"
    tags_filter: Optional[List[str]] = None


class TestQueryResult(BaseModel):
    content: str
    score: float
    source_file: str
    title: Optional[str]
    tags: List[str]
    page: Optional[int]


# ---------------------------------------------------------------------------
# Source document endpoints
# ---------------------------------------------------------------------------

@router.get("/sources", response_model=List[SourceResponse])
async def list_sources(admin=Depends(verify_admin)):
    """List all registered RAG source documents with metadata."""
    docs = await SourceMetadataManager.list_all()
    return [SourceResponse(**d.to_dict()) for d in docs]


@router.post("/sources/upload")
async def upload_source(
    file: UploadFile,
    admin=Depends(verify_admin),
):
    """
    Upload a new source document (PDF/TXT/DOCX).

    Rejected if a file with the same SHA-256 hash is already registered.
    Triggers page/word count extraction and creates a metadata row.
    """
    filename = file.filename or "upload"

    if not _doc_manager.is_supported_file(filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {list(_doc_manager.SUPPORTED_EXTENSIONS)}",
        )

    # Stream to a temp file so we can hash and validate before committing
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(filename).suffix
    ) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Check for duplicate hash before touching sources/
        sha256 = DocumentManager.compute_sha256(tmp_path)
        existing = await SourceMetadataManager.get_by_sha256(sha256)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"A file with the same content already exists: {existing.filename}",
            )

        # Move into sources/
        save_result = _doc_manager.save_uploaded_file(tmp_path, filename)
        if not save_result["success"]:
            raise HTTPException(status_code=400, detail=save_result.get("message", "Save failed"))

        saved_name = save_result["filename"]
        saved_path = Path(SOURCES_DIR) / saved_name
        size_bytes = save_result["info"]["size"]
        ext = Path(saved_name).suffix.lower()

        # Extract page/word count in the background (can be slow for large PDFs)
        info = _doc_manager.get_document_info(saved_name)
        page_count = info.get("pages") if info else None
        word_count = info.get("word_count") if info else None

        # Create metadata row
        source_doc = await SourceMetadataManager.create(
            filename=saved_name,
            sha256=sha256,
            size_bytes=size_bytes,
            extension=ext,
            page_count=page_count,
            word_count=word_count,
        )

        return {
            "success": True,
            "source": SourceResponse(**source_doc.to_dict()),
            "warnings": save_result.get("warnings", []),
        }

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.patch("/sources/{filename}")
async def update_source(
    filename: str,
    updates: SourceUpdate,
    admin=Depends(verify_admin),
):
    """Update source document metadata (tags, title, author, year, enabled)."""
    source = await SourceMetadataManager.update(
        filename=filename,
        title=updates.title,
        author=updates.author,
        year=updates.year,
        tags=updates.tags,
        enabled=updates.enabled,
    )
    if source is None:
        raise HTTPException(status_code=404, detail=f"Source not found: {filename}")
    return SourceResponse(**source.to_dict())


@router.delete("/sources/{filename}")
async def delete_source(filename: str, admin=Depends(verify_admin)):
    """Delete a source file and its metadata row."""
    deleted_meta = await SourceMetadataManager.delete(filename)
    deleted_file = _doc_manager.delete_document(filename)

    if not deleted_meta and not deleted_file["success"]:
        raise HTTPException(status_code=404, detail=f"Source not found: {filename}")

    return {
        "success": True,
        "metadata_deleted": deleted_meta,
        "file_deleted": deleted_file["success"],
    }


# ---------------------------------------------------------------------------
# Vectorstore status
# ---------------------------------------------------------------------------

@router.get("/vectorstore/status", response_model=VectorstoreStatus)
async def get_vectorstore_status(admin=Depends(verify_admin)):
    """Return current vectorstore metadata: model, chunk count, build time, size."""
    live = Path(LIVE_VECTORSTORE)
    index_exists = (live / "index.faiss").exists()

    meta_path = live / "metadata.json"
    embedding_model = None
    chunks_count = None
    last_build = None
    dimensions = None

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            embedding_model = meta.get("embedding_model")
            chunks_count = meta.get("chunks_count")
            last_build = meta.get("created_at")
            specs = EmbeddingConfig.EMBEDDING_MODELS.get(embedding_model or "", {})
            dimensions = specs.get("dimensions")
        except Exception as exc:
            logger.warning(f"Could not read vectorstore metadata: {exc}")

    # Total directory size
    size_bytes = sum(
        f.stat().st_size for f in live.rglob("*") if f.is_file()
    ) if live.exists() else 0

    return VectorstoreStatus(
        embedding_model=embedding_model,
        chunks_count=chunks_count,
        last_build=last_build,
        size_bytes=size_bytes,
        dimensions=dimensions,
        metadata_path=str(meta_path),
        index_exists=index_exists,
    )


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------

async def _run_rebuild(job_id: str, params: RebuildRequest):
    """Background coroutine that drives a full rebuild + promote cycle."""
    errors: list[str] = []

    async def progress(current: int, total: int, filename: str):
        pct = int(current / total * 90) if total else 0  # reserve last 10% for promote
        await SourceMetadataManager.update_job(
            job_id,
            status="running",
            percent=pct,
            current_file=filename,
        )

    try:
        await SourceMetadataManager.update_job(job_id, status="running", percent=0)

        result: BuildResult = await build_index(
            sources_dir=SOURCES_DIR,
            target_dir=STAGING_VECTORSTORE,
            chunk_size=params.chunk_size,
            chunk_overlap=params.chunk_overlap,
            embedding_model=params.embedding_model,
            tags_filter=params.tags_filter,
            progress_callback=progress,
        )
        errors = result.errors

        if result.chunks_created == 0:
            raise RuntimeError("No chunks created — check source files and tags filter.")

        if result.errors:
            raise RuntimeError(
                f"Build completed with {len(result.errors)} error(s); refusing to promote. "
                f"First error: {result.errors[0]}"
            )

        await SourceMetadataManager.update_job(job_id, percent=92, current_file="promoting...")

        promote_staging(staging_dir=STAGING_VECTORSTORE, live_dir=LIVE_VECTORSTORE)

        # Mark all indexed files as having been indexed now
        await SourceMetadataManager.mark_indexed(result.indexed_filenames)

        await SourceMetadataManager.update_job(
            job_id,
            status="done",
            percent=100,
            current_file=None,
            errors=errors,
            finished=True,
        )
        logger.info(
            f"Rebuild job {job_id} complete: {result.chunks_created} chunks from "
            f"{result.files_processed} files in {result.duration_seconds:.1f}s"
        )

    except Exception as exc:
        errors.append(str(exc))
        logger.error(f"Rebuild job {job_id} failed: {exc}")
        await SourceMetadataManager.update_job(
            job_id,
            status="error",
            errors=errors,
            finished=True,
        )


@router.post("/vectorstore/rebuild")
async def trigger_rebuild(
    request: RebuildRequest,
    background_tasks: BackgroundTasks,
    admin=Depends(verify_admin),
):
    """
    Kick off a background vectorstore rebuild.

    Returns a job_id to poll with GET /api/admin/vectorstore/jobs/{job_id}.
    """
    job = await SourceMetadataManager.create_job(request.model_dump())
    background_tasks.add_task(_run_rebuild, str(job.id), request)
    return {"job_id": str(job.id), "status": "pending"}


@router.get("/vectorstore/jobs/{job_id}", response_model=RebuildJobResponse)
async def get_rebuild_job(job_id: str, admin=Depends(verify_admin)):
    """Poll the status of a background rebuild job."""
    job = await SourceMetadataManager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return RebuildJobResponse(**job.to_dict())


@router.get("/vectorstore/jobs", response_model=List[RebuildJobResponse])
async def list_rebuild_jobs(admin=Depends(verify_admin)):
    """List the 10 most recent rebuild jobs."""
    jobs = await SourceMetadataManager.list_recent_jobs()
    return [RebuildJobResponse(**j.to_dict()) for j in jobs]


# ---------------------------------------------------------------------------
# Test query
# ---------------------------------------------------------------------------

@router.post("/vectorstore/test-query", response_model=List[TestQueryResult])
async def test_query(request: TestQueryRequest, admin=Depends(verify_admin)):
    """
    Run a similarity search against the live or staging vectorstore.

    Useful for verifying a rebuild before promoting, or inspecting retrieval quality.
    Results are optionally filtered by tags (post-retrieval).
    """
    target_dir = STAGING_VECTORSTORE if request.target == "staging" else LIVE_VECTORSTORE
    index_path = Path(target_dir) / "index.faiss"

    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No index found at {target_dir}. Run a rebuild first.",
        )

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings

        # Load embedding model from stored metadata
        meta_path = Path(target_dir) / "metadata.json"
        embedding_model = "text-embedding-3-small"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            embedding_model = meta.get("embedding_model", embedding_model)

        embeddings = OpenAIEmbeddings(model=embedding_model)
        vs = FAISS.load_local(
            target_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        # Fetch many more than k so post-retrieval tag filtering surfaces
        # results even for under-represented tags (matches rag_engine behavior).
        fetch_k = max(request.k * 40, 200) if request.tags_filter else max(request.k * 4, 20)
        results = vs.similarity_search_with_score(request.query, k=fetch_k)

        output: list[TestQueryResult] = []
        for doc, score in results:
            tags: list[str] = doc.metadata.get("tags", [])

            # Apply tag filter (post-retrieval)
            if request.tags_filter:
                if not any(t in request.tags_filter for t in tags):
                    continue

            output.append(
                TestQueryResult(
                    content=doc.page_content[:500],
                    score=float(score),
                    source_file=doc.metadata.get("source_file", ""),
                    title=doc.metadata.get("title"),
                    tags=tags,
                    page=doc.metadata.get("page"),
                )
            )

            if len(output) >= request.k:
                break

        return output

    except Exception as exc:
        logger.error(f"Test query failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")
