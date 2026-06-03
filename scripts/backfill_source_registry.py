"""
Backfill the source_documents registry from the files already on disk.

Why this exists: the live vectorstore was built by an older CLI process that
never registered sources or stamped tags onto chunks. As a result the
source_documents table is nearly empty and tag-scoped retrieval (negotiation/
sales modes) matches zero chunks. This script registers every file in
DATA_SOURCES_DIR with a tag derived from its top-level folder, so the next
rebuild stamps the correct tags onto every chunk.

Tag rule: a file's tag is its first path segment under the sources dir
(negotiation/ -> ['negotiation'], sales/ -> ['sales']). _archive/ is skipped.

Idempotent: clears source_documents, then re-inserts one row per file keyed by
its relative path. Safe to re-run. Does NOT touch the vectorstore or embed
anything; run the rebuild afterwards to apply.

Run inside the backend container:
    docker exec negotiatorpro-backend-1 python scripts/backfill_source_registry.py
Add --apply to actually write; without it the script only prints the plan.
"""
import asyncio
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, "/app")

from backend.database import db
from backend.source_metadata import SourceMetadataManager

SUPPORTED = {".pdf", ".txt", ".docx"}
TAG_BY_FOLDER = {"negotiation": "negotiation", "sales": "sales"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _title_from(stem: str) -> str:
    # "never-split-the-difference" -> "Never Split The Difference"
    cleaned = stem.replace("_", " ").replace("-", " ").strip()
    return " ".join(w.capitalize() for w in cleaned.split())


async def main(apply: bool) -> None:
    sources_dir = Path(os.getenv("DATA_SOURCES_DIR", "/app/data-sources")).resolve()

    files = sorted(
        p for p in sources_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED
        and "_archive" not in p.parts
    )

    plan = []
    for path in files:
        rel = path.relative_to(sources_dir).as_posix()
        top = path.relative_to(sources_dir).parts[0]
        tag = TAG_BY_FOLDER.get(top)
        if tag is None:
            print(f"SKIP (no tag folder): {rel}")
            continue
        plan.append((path, rel, tag))

    print(f"\n{'APPLYING' if apply else 'DRY RUN'} — {len(plan)} files to register\n")
    for _, rel, tag in plan:
        print(f"  [{tag:11}] {rel}")

    if not apply:
        print("\nDry run only. Re-run with --apply to write.")
        return

    await db.connect()
    try:
        existing = await SourceMetadataManager.list_all()
        for sd in existing:
            await SourceMetadataManager.delete(sd.filename)
        print(f"\nCleared {len(existing)} stale row(s).")

        for path, rel, tag in plan:
            await SourceMetadataManager.create(
                filename=rel,
                sha256=_sha256(path),
                size_bytes=path.stat().st_size,
                extension=path.suffix.lower(),
                title=_title_from(path.stem),
                tags=[tag],
            )
            print(f"  registered {rel}")
        print(f"\nDone. {len(plan)} sources registered.")
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main(apply="--apply" in sys.argv))
