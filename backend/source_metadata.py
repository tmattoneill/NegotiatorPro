"""
Source Document Metadata Manager

Async CRUD for the source_documents and rebuild_jobs Postgres tables.
File-system operations live in DocumentManager; this module owns only DB state.
"""
import json
import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from .database import db

logger = logging.getLogger(__name__)


class SourceDocument:
    """In-memory representation of a source_documents row."""

    def __init__(self, row):
        d = dict(row)
        self.id: UUID = d["id"]
        self.filename: str = d["filename"]
        self.sha256: str = d["sha256"]
        self.title: Optional[str] = d.get("title")
        self.author: Optional[str] = d.get("author")
        self.year: Optional[int] = d.get("year")
        self.tags: list[str] = list(d.get("tags") or [])
        self.enabled: bool = d["enabled"]
        self.page_count: Optional[int] = d.get("page_count")
        self.word_count: Optional[int] = d.get("word_count")
        self.size_bytes: int = d["size_bytes"]
        self.extension: str = d["extension"]
        self.added_at: datetime = d["added_at"]
        self.last_indexed_at: Optional[datetime] = d.get("last_indexed_at")

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "filename": self.filename,
            "sha256": self.sha256,
            "title": self.title,
            "author": self.author,
            "year": self.year,
            "tags": self.tags,
            "enabled": self.enabled,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "size_bytes": self.size_bytes,
            "extension": self.extension,
            "added_at": self.added_at.isoformat(),
            "last_indexed_at": self.last_indexed_at.isoformat() if self.last_indexed_at else None,
        }


class RebuildJob:
    """In-memory representation of a rebuild_jobs row."""

    def __init__(self, row):
        d = dict(row)
        self.id: UUID = d["id"]
        self.status: str = d["status"]
        self.percent: int = d["percent"]
        self.current_file: Optional[str] = d.get("current_file")
        self.errors: list = json.loads(d["errors"]) if isinstance(d["errors"], str) else (d["errors"] or [])
        self.params: dict = json.loads(d["params"]) if isinstance(d["params"], str) else (d["params"] or {})
        self.started_at: datetime = d["started_at"]
        self.finished_at: Optional[datetime] = d.get("finished_at")

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "status": self.status,
            "percent": self.percent,
            "current_file": self.current_file,
            "errors": self.errors,
            "params": self.params,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class SourceMetadataManager:
    """Async CRUD for source_documents and rebuild_jobs tables."""

    # ------------------------------------------------------------------
    # source_documents
    # ------------------------------------------------------------------

    @staticmethod
    async def create(
        filename: str,
        sha256: str,
        size_bytes: int,
        extension: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        year: Optional[int] = None,
        tags: Optional[list[str]] = None,
        page_count: Optional[int] = None,
        word_count: Optional[int] = None,
    ) -> SourceDocument:
        row = await db.fetchrow(
            """
            INSERT INTO source_documents
                (filename, sha256, size_bytes, extension, title, author, year,
                 tags, page_count, word_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING *
            """,
            filename, sha256, size_bytes, extension,
            title, author, year,
            tags or [],
            page_count, word_count,
        )
        return SourceDocument(row)

    @staticmethod
    async def get_by_filename(filename: str) -> Optional[SourceDocument]:
        row = await db.fetchrow(
            "SELECT * FROM source_documents WHERE filename = $1", filename
        )
        return SourceDocument(row) if row else None

    @staticmethod
    async def get_by_sha256(sha256: str) -> Optional[SourceDocument]:
        row = await db.fetchrow(
            "SELECT * FROM source_documents WHERE sha256 = $1", sha256
        )
        return SourceDocument(row) if row else None

    @staticmethod
    async def list_all() -> list[SourceDocument]:
        rows = await db.fetch(
            "SELECT * FROM source_documents ORDER BY added_at DESC"
        )
        return [SourceDocument(r) for r in rows]

    @staticmethod
    async def list_enabled(tags_filter: Optional[list[str]] = None) -> list[SourceDocument]:
        """Return enabled sources, optionally filtered so at least one tag overlaps."""
        if tags_filter:
            rows = await db.fetch(
                """
                SELECT * FROM source_documents
                WHERE enabled = TRUE AND tags && $1::text[]
                ORDER BY added_at DESC
                """,
                tags_filter,
            )
        else:
            rows = await db.fetch(
                "SELECT * FROM source_documents WHERE enabled = TRUE ORDER BY added_at DESC"
            )
        return [SourceDocument(r) for r in rows]

    @staticmethod
    async def update(
        filename: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        year: Optional[int] = None,
        tags: Optional[list[str]] = None,
        enabled: Optional[bool] = None,
        page_count: Optional[int] = None,
        word_count: Optional[int] = None,
    ) -> Optional[SourceDocument]:
        """Partial update — only supplied (non-None) fields are changed."""
        fields, values = [], []
        idx = 1

        def add(col, val):
            nonlocal idx
            fields.append(f"{col} = ${idx}")
            values.append(val)
            idx += 1

        if title is not None:
            add("title", title)
        if author is not None:
            add("author", author)
        if year is not None:
            add("year", year)
        if tags is not None:
            add("tags", tags)
        if enabled is not None:
            add("enabled", enabled)
        if page_count is not None:
            add("page_count", page_count)
        if word_count is not None:
            add("word_count", word_count)

        if not fields:
            return await SourceMetadataManager.get_by_filename(filename)

        values.append(filename)
        sql = f"UPDATE source_documents SET {', '.join(fields)} WHERE filename = ${idx} RETURNING *"
        row = await db.fetchrow(sql, *values)
        return SourceDocument(row) if row else None

    @staticmethod
    async def mark_indexed(filenames: list[str]) -> None:
        """Stamp last_indexed_at = now for all named source files."""
        await db.execute(
            "UPDATE source_documents SET last_indexed_at = NOW() WHERE filename = ANY($1::text[])",
            filenames,
        )

    @staticmethod
    async def delete(filename: str) -> bool:
        result = await db.execute(
            "DELETE FROM source_documents WHERE filename = $1", filename
        )
        return result == "DELETE 1"

    # ------------------------------------------------------------------
    # rebuild_jobs
    # ------------------------------------------------------------------

    @staticmethod
    async def create_job(params: dict) -> RebuildJob:
        row = await db.fetchrow(
            """
            INSERT INTO rebuild_jobs (status, percent, params)
            VALUES ('pending', 0, $1::jsonb)
            RETURNING *
            """,
            json.dumps(params),
        )
        return RebuildJob(row)

    @staticmethod
    async def get_job(job_id: str) -> Optional[RebuildJob]:
        row = await db.fetchrow(
            "SELECT * FROM rebuild_jobs WHERE id = $1", job_id
        )
        return RebuildJob(row) if row else None

    @staticmethod
    async def update_job(
        job_id: str,
        status: Optional[str] = None,
        percent: Optional[int] = None,
        current_file: Optional[str] = None,
        errors: Optional[list] = None,
        finished: bool = False,
    ) -> None:
        fields, values = [], []
        idx = 1

        def add(col, val):
            nonlocal idx
            fields.append(f"{col} = ${idx}")
            values.append(val)
            idx += 1

        if status is not None:
            add("status", status)
        if percent is not None:
            add("percent", percent)
        if current_file is not None:
            add("current_file", current_file)
        if errors is not None:
            add("errors", json.dumps(errors))
        if finished:
            add("finished_at", datetime.utcnow())

        if not fields:
            return

        values.append(job_id)
        sql = f"UPDATE rebuild_jobs SET {', '.join(fields)} WHERE id = ${idx}"
        await db.execute(sql, *values)

    @staticmethod
    async def list_recent_jobs(limit: int = 10) -> list[RebuildJob]:
        rows = await db.fetch(
            "SELECT * FROM rebuild_jobs ORDER BY started_at DESC LIMIT $1", limit
        )
        return [RebuildJob(r) for r in rows]
