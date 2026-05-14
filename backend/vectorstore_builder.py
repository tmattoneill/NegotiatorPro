"""
Vectorstore Builder

Callable service for building and promoting FAISS vectorstores.
Designed to be called from background tasks and the CLI rebuild script.

Build flow:
  1. List enabled sources (filtered by tags if requested)
  2. Load + chunk documents, stamping each chunk with source metadata
  3. Embed and write to a staging directory
  4. Caller promotes staging → live with promote_staging()
"""
import asyncio
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .document_manager import DocumentManager
from .embedding_config import EmbeddingConfig
from .source_metadata import SourceDocument, SourceMetadataManager

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


@dataclass
class BuildResult:
    files_attempted: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    pages_loaded: int = 0
    chunks_created: int = 0
    embedding_model: str = ""
    estimated_tokens: int = 0
    estimated_cost_usd: float = 0.0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    indexed_filenames: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_document(path: Path) -> list:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    if ext == ".txt":
        return TextLoader(str(path), encoding="utf-8").load()
    if ext == ".docx":
        return Docx2txtLoader(str(path)).load()
    raise ValueError(f"Unsupported extension: {ext}")


def _stamp_metadata(docs: list, source_doc: Optional[SourceDocument], filename: str) -> None:
    """Attach provenance fields to every LangChain Document in-place."""
    tags = source_doc.tags if source_doc else []
    title = source_doc.title or filename if source_doc else filename
    sha256 = source_doc.sha256 if source_doc else ""

    for doc in docs:
        doc.metadata.update(
            {
                "source_file": filename,
                "sha256": sha256,
                "tags": tags,
                "title": title,
            }
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def build_index(
    sources_dir: Optional[str] = None,
    target_dir: str = "vectorstore_staging",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = "text-embedding-3-small",
    tags_filter: Optional[list[str]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> BuildResult:
    """
    Build a FAISS index from source documents and write it to target_dir.

    Args:
        sources_dir:        Directory containing source files.
        target_dir:         Where to write the new index (staging area).
        chunk_size:         LangChain text-splitter chunk size in characters.
        chunk_overlap:      Overlap between adjacent chunks.
        embedding_model:    OpenAI embedding model name.
        tags_filter:        If set, only include sources whose tags overlap.
                            Unregistered files (no metadata row) are included
                            only when tags_filter is None.
        progress_callback:  Optional sync or async callable(current, total, filename).

    Returns:
        BuildResult with counts, cost estimate, and any errors.
    """
    result = BuildResult(embedding_model=embedding_model)
    t0 = time.monotonic()

    if sources_dir is None:
        import os
        sources_dir = os.getenv("DATA_SOURCES_DIR", "../data-sources")

    sources_path = Path(sources_dir).resolve()
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Build the candidate file list
    # -----------------------------------------------------------------------
    # Query metadata for all enabled sources (with optional tag filter)
    try:
        registered = {
            sd.filename: sd
            for sd in await SourceMetadataManager.list_enabled(tags_filter)
        }
    except Exception as exc:
        logger.warning(f"Could not query source metadata: {exc}. Falling back to directory scan.")
        registered = {}

    # Recursive walk — sources_path may contain subdirs (sales/, negotiation/),
    # skip the _archive/ subdir which holds legacy/duplicate content.
    all_files = sorted(
        p for p in sources_path.rglob('*')
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and '_archive' not in p.parts
    )

    candidates: list[tuple[Path, Optional[SourceDocument]]] = []
    for path in all_files:
        # Filename in DB is the path relative to sources_path (e.g. "sales/foo.pdf"),
        # so registered files are looked up by relative path, not basename.
        rel = path.relative_to(sources_path).as_posix()
        if rel in registered:
            candidates.append((path, registered[rel]))
        else:
            # File exists on disk but has no metadata row
            if tags_filter is None:
                # No filter — include unregistered files with no tags
                candidates.append((path, None))
            else:
                # Tag filter active — skip unregistered files (can't confirm they match)
                result.files_skipped += 1

    result.files_attempted = len(candidates)

    # -----------------------------------------------------------------------
    # 2. Load, stamp, and chunk
    # -----------------------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    all_chunks: list = []

    for idx, (path, source_doc) in enumerate(candidates):
        # Use the relative path as the canonical identifier (matches DB and
        # makes mark_indexed work correctly when filenames include subdirs).
        rel = path.relative_to(sources_path).as_posix()

        if progress_callback:
            await _maybe_await(progress_callback(idx, len(candidates), rel))

        try:
            raw_docs = _load_document(path)
            _stamp_metadata(raw_docs, source_doc, rel)
            chunks = splitter.split_documents(raw_docs)
            all_chunks.extend(chunks)
            result.pages_loaded += len(raw_docs)
            result.files_processed += 1
            result.indexed_filenames.append(rel)
            logger.info(f"Loaded {rel}: {len(raw_docs)} pages → {len(chunks)} chunks")
        except Exception as exc:
            msg = f"Failed to load {rel}: {exc}"
            logger.error(msg)
            result.errors.append(msg)
            result.files_skipped += 1

    if progress_callback:
        await _maybe_await(progress_callback(len(candidates), len(candidates), "embedding..."))

    result.chunks_created = len(all_chunks)

    # Estimate cost
    token_estimate = sum(len(c.page_content.split()) for c in all_chunks)
    result.estimated_tokens = int(token_estimate / 0.75)  # words → tokens
    specs = EmbeddingConfig.EMBEDDING_MODELS.get(embedding_model, {})
    result.estimated_cost_usd = (result.estimated_tokens / 1000) * specs.get("cost_per_1k", 0.0)

    if not all_chunks:
        result.errors.append("No content extracted — vectorstore not written.")
        result.duration_seconds = time.monotonic() - t0
        return result

    # -----------------------------------------------------------------------
    # 3. Embed and persist to target_dir
    # -----------------------------------------------------------------------
    try:
        embeddings = OpenAIEmbeddings(model=embedding_model)
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        vectorstore.save_local(str(target_path))

        # Write build metadata alongside the index
        import json
        meta = {
            "embedding_model": embedding_model,
            "model_specs": specs,
            "chunks_count": result.chunks_created,
            "files_processed": result.files_processed,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "tags_filter": tags_filter,
            "created_at": datetime.utcnow().isoformat(),
        }
        (target_path / "metadata.json").write_text(json.dumps(meta, indent=2))

        logger.info(
            f"Vectorstore written to {target_path} — "
            f"{result.chunks_created} chunks, model={embedding_model}"
        )
    except Exception as exc:
        msg = f"Embedding/save failed: {exc}"
        logger.error(msg)
        result.errors.append(msg)

    result.duration_seconds = time.monotonic() - t0
    return result


def promote_staging(
    staging_dir: str = "vectorstore_staging",
    live_dir: str = "vectorstore",
) -> str:
    """
    Atomically swap staging into the live vectorstore directory.

    1. Rename live_dir → vectorstore_backup_<timestamp> (if it exists)
    2. Rename staging_dir → live_dir

    Returns the backup directory path (empty string if no backup was needed).
    """
    staging = Path(staging_dir)
    live = Path(live_dir)

    if not staging.exists():
        raise FileNotFoundError(f"Staging directory not found: {staging}")

    # Refuse to promote a staging dir that doesn't contain a real FAISS index.
    # Without this guard, a build that failed at the embedding step (empty
    # staging dir) would wipe out the live vectorstore.
    if not (staging / "index.faiss").exists():
        raise RuntimeError(
            f"Staging dir {staging} has no index.faiss — refusing to promote. "
            "This usually means the embedding step failed; check build logs."
        )

    backup_path = ""
    if live.exists():
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup = live.parent / f"vectorstore_backup_{ts}"
        shutil.move(str(live), str(backup))
        backup_path = str(backup)
        logger.info(f"Backed up live vectorstore to {backup}")

    shutil.move(str(staging), str(live))
    logger.info(f"Promoted {staging} → {live}")
    return backup_path


# ---------------------------------------------------------------------------
# Internal: await the callback result if it is a coroutine
# ---------------------------------------------------------------------------

async def _maybe_await(result):
    """Await `result` if it is a coroutine; otherwise return it unchanged."""
    if asyncio.iscoroutine(result):
        return await result
    return result
