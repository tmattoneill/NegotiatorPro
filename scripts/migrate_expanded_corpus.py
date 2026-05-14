#!/usr/bin/env python3
"""
Migrate Expanded Corpus

One-time script that:
  1. Copies missing sales/negotiation books from ../data-sources/negotiaton-sources/
     into sources/, skipping any file whose SHA-256 is already registered.
  2. Backfills source_documents metadata rows for every file in sources/
     that has no row yet (existing + newly copied), inferring tags from filename.

Run with --dry-run to see what would happen without writing anything.

Usage (from NegotiatorPro/ directory):
    python scripts/migrate_expanded_corpus.py [--dry-run]
"""
import argparse
import asyncio
import os
import shutil
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Corpus mapping: filename (in data-sources) → inferred metadata
# ---------------------------------------------------------------------------

# NOTE: This script was the one-off used to seed data-sources/ from the legacy
# corpus. After that migration ran, NegotiatorPro/sources/ was removed and the
# canonical corpus moved into data-sources/{sales,negotiation}/. The script is
# left here for reference. To onboard a new book, drop it into the appropriate
# subdir of data-sources/ and register metadata via the admin UI (or upload via
# the UI, which routes through this same metadata layer).
CORPUS_SOURCES = Path(__file__).parent.parent.parent / "data-sources" / "_archive" / "negotiaton-sources"
DEST = Path(os.getenv("DATA_SOURCES_DIR", Path(__file__).parent.parent.parent / "data-sources"))
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# Explicit metadata per file (inferred where absent)
FILE_METADATA: dict[str, dict] = {
    "challenger_sale.pdf":          {"title": "The Challenger Sale",                 "author": "Matthew Dixon & Brent Adamson", "year": 2011, "tags": ["sales"]},
    "gap-selling.pdf":              {"title": "Gap Selling",                          "author": "Keenan",                        "year": 2018, "tags": ["sales"]},
    "jolt_effect.pdf":              {"title": "The JOLT Effect",                     "author": "Matthew Dixon & Ted McKenna",   "year": 2022, "tags": ["sales"]},
    "mind-for-sales.pdf":           {"title": "A Mind for Sales",                    "author": "Mark Hunter",                   "year": 2020, "tags": ["sales"]},
    "spin_sellin.pdf":              {"title": "SPIN Selling",                         "author": "Neil Rackham",                  "year": 1988, "tags": ["sales"]},
    "fisher-getting-to-yes.pdf":    {"title": "Getting to Yes",                      "author": "Roger Fisher & William Ury",    "year": 1981, "tags": ["negotiation"]},
    "power-postive-no.pdf":         {"title": "The Power of a Positive No",          "author": "William Ury",                   "year": 2007, "tags": ["negotiation"]},
    "win-friends-infulence.pdf":    {"title": "How to Win Friends and Influence People", "author": "Dale Carnegie",             "year": 1936, "tags": ["sales"]},
    # Already in sources/ — metadata backfill targets
    "never-split-the-difference.pdf":   {"title": "Never Split the Difference",     "author": "Chris Voss",                    "year": 2016, "tags": ["negotiation"]},
    "never-split-the-differene.pdf":    {"title": "Never Split the Difference",     "author": "Chris Voss",                    "year": 2016, "tags": ["negotiation"]},
    "Getting to Yes, Roger Fisher and William Ury.pdf": {"title": "Getting to Yes", "author": "Roger Fisher & William Ury",    "year": 1981, "tags": ["negotiation"]},
    "Getting_past_NO_personal_and_professiona.pdf": {"title": "Getting Past No",    "author": "William Ury",                   "year": 1991, "tags": ["negotiation"]},
    "Negotiation_Conflict_Management.pdf": {"title": "Negotiation & Conflict Management", "author": None,                      "year": None,  "tags": ["negotiation"]},
    "Better, Not Perfect: A Realist's Guide to Maximum Sustainable Goodness, Max H. Bazernam.pdf": {
        "title": "Better, Not Perfect", "author": "Max H. Bazerman", "year": 2020, "tags": ["negotiation"],
    },
    "upward-spiral-workbook.pdf":   {"title": "The Upward Spiral Workbook",          "author": "Alex Korb",                    "year": 2019, "tags": ["negotiation"]},
    "you-dont-get-what-you-deserve.pdf": {"title": "You Don't Get What You Deserve", "author": None,                          "year": None,  "tags": ["negotiation"]},
    "The Art of Negotiation, Chris Voss.txt": {"title": "The Art of Negotiation",    "author": "Chris Voss",                   "year": None,  "tags": ["negotiation"]},
}


def infer_metadata(filename: str) -> dict:
    """Fall back for files not in FILE_METADATA — always single-tag."""
    name_lower = filename.lower()
    if any(k in name_lower for k in ["challeng", "spin", "gap-sell", "jolt", "mind-for-sales", "carnegie", "win-friends"]):
        tags = ["sales"]
    else:
        # Default to negotiation for anything that isn't clearly a sales book
        tags = ["negotiation"]
    return {"title": None, "author": None, "year": None, "tags": tags}


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

async def run(dry_run: bool):
    from backend.database import db
    from backend.document_manager import DocumentManager
    from backend.source_metadata import SourceMetadataManager

    print(f"{'[DRY RUN] ' if dry_run else ''}Corpus migration starting…\n")

    await db.connect()

    # Step 1: Copy missing files from data-sources → sources/
    if not CORPUS_SOURCES.exists():
        print(f"WARNING: source corpus directory not found: {CORPUS_SOURCES}")
        print("Skipping copy step.\n")
    else:
        print(f"Source corpus: {CORPUS_SOURCES}")
        print(f"Destination:   {DEST}\n")
        DEST.mkdir(exist_ok=True)

        # Pre-compute SHAs for every existing supported file in DEST so we can
        # dedupe against content already on disk, not just against metadata rows.
        # (On a first migration the metadata table is empty, so DB lookup alone
        # would happily copy in a byte-identical duplicate under a different name.)
        dest_shas: dict[str, str] = {}
        for f in DEST.iterdir():
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                dest_shas[DocumentManager.compute_sha256(str(f))] = f.name

        for src_file in sorted(CORPUS_SOURCES.iterdir()):
            if not src_file.is_file() or src_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            sha256 = DocumentManager.compute_sha256(str(src_file))

            # Dedupe by content against existing on-disk files
            if sha256 in dest_shas:
                print(f"  SKIP (same content as {dest_shas[sha256]}): {src_file.name}")
                continue

            # Dedupe against metadata table (catches files moved/renamed after
            # earlier ingestion)
            existing = await SourceMetadataManager.get_by_sha256(sha256)
            if existing:
                print(f"  SKIP (hash registered as {existing.filename}): {src_file.name}")
                continue

            dest_path = DEST / src_file.name
            print(f"  COPY: {src_file.name}")
            if not dry_run:
                shutil.copy2(str(src_file), str(dest_path))
                dest_shas[sha256] = src_file.name  # track for in-loop dedupe

    # Step 2: Backfill metadata rows for every file in sources/
    print(f"\nBackfilling source_documents metadata for sources/ …\n")
    for dest_file in sorted(DEST.iterdir()):
        if not dest_file.is_file() or dest_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        sha256 = DocumentManager.compute_sha256(str(dest_file))

        existing = await SourceMetadataManager.get_by_sha256(sha256)
        if existing:
            print(f"  SKIP (metadata exists): {dest_file.name}")
            continue

        meta = FILE_METADATA.get(dest_file.name) or infer_metadata(dest_file.name)
        stat = dest_file.stat()

        print(
            f"  CREATE metadata: {dest_file.name}"
            f" | tags={meta['tags']}"
            f" | title={meta.get('title', '?')}"
        )

        if not dry_run:
            await SourceMetadataManager.create(
                filename=dest_file.name,
                sha256=sha256,
                size_bytes=stat.st_size,
                extension=dest_file.suffix.lower(),
                title=meta.get("title"),
                author=meta.get("author"),
                year=meta.get("year"),
                tags=meta.get("tags", []),
            )

    await db.disconnect()
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Done.")
    if dry_run:
        print("Run without --dry-run to apply changes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate expanded corpus into NegotiatorPro sources/")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without making changes")
    args = parser.parse_args()
    asyncio.run(run(dry_run=args.dry_run))
