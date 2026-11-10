#!/usr/bin/env python3
"""
Apply numbered SQL migrations (migrations/NNN_*.sql) and record them in a
schema_migrations table, so every environment knows exactly what it has run.

Most migration files are not idempotent, which is why we track them rather
than re-run them. Each pending file runs in its own transaction: a failure
rolls that file back and stops, leaving earlier files applied and recorded.

Usage (DATABASE_URL from the environment):
    python3 scripts/migrate.py --status
    python3 scripts/migrate.py --apply
    python3 scripts/migrate.py --baseline-through 011

Deploy scripts pipe this file into the backend container on stdin
(python3 - --apply < scripts/migrate.py), so it must stay a single file
with no imports from the repo. Inside the container migrations live at
/app/migrations; MIGRATIONS_DIR overrides that.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

import asyncpg

MIGRATION_PATTERN = re.compile(r"^(\d{3})_[\w-]+\.sql$")

# Some older files wrap themselves in BEGIN;/COMMIT;. We already run each file
# in a transaction, and an inner COMMIT would end ours early and record the
# migration outside it. plpgsql blocks use a bare BEGIN with no semicolon, so
# they don't match.
TOP_LEVEL_TRANSACTION = re.compile(r"^\s*(BEGIN|COMMIT)\s*;\s*$", re.IGNORECASE | re.MULTILINE)

TRACKING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    baselined   BOOLEAN NOT NULL DEFAULT FALSE
)
"""


def find_migrations_dir() -> Path:
    override = os.environ.get("MIGRATIONS_DIR")
    candidates = [Path(override)] if override else []
    candidates += [Path("/app/migrations"), Path.cwd() / "migrations"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    sys.exit(f"migrate: no migrations directory found (tried {', '.join(map(str, candidates))})")


def list_migrations(migrations_dir: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for path in sorted(migrations_dir.iterdir()):
        match = MIGRATION_PATTERN.match(path.name)
        if match:
            found.append((match.group(1), path))
    versions = [version for version, _ in found]
    duplicates = {v for v in versions if versions.count(v) > 1}
    if duplicates:
        sys.exit(f"migrate: duplicate migration numbers: {', '.join(sorted(duplicates))}")
    return found


async def applied_versions(conn: asyncpg.Connection) -> set[str]:
    exists = await conn.fetchval("SELECT to_regclass('public.schema_migrations') IS NOT NULL")
    if not exists:
        return set()
    rows = await conn.fetch("SELECT version FROM schema_migrations")
    return {row["version"] for row in rows}


async def count_user_tables(conn: asyncpg.Connection) -> int:
    return await conn.fetchval(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name <> 'schema_migrations'"
    )


async def run(args: argparse.Namespace) -> int:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        sys.exit("migrate: DATABASE_URL is not set")

    migrations = list_migrations(find_migrations_dir())
    conn = await asyncpg.connect(database_url)
    try:
        done = await applied_versions(conn)
        pending = [(v, p) for v, p in migrations if v not in done]

        if args.status:
            tables = await count_user_tables(conn)
            print(f"migrate: {len(done)} applied, {len(pending)} pending, {tables} user tables")
            for version, path in migrations:
                print(f"  [{'x' if version in done else ' '}] {path.name}")
            return 0

        if args.baseline_through:
            target = args.baseline_through.zfill(3)
            if target not in {v for v, _ in migrations}:
                sys.exit(f"migrate: no migration numbered {target}")
            await conn.execute(TRACKING_TABLE_SQL)
            marked = 0
            for version, path in migrations:
                if version > target or version in done:
                    continue
                await conn.execute(
                    "INSERT INTO schema_migrations (version, filename, baselined) VALUES ($1, $2, TRUE)",
                    version, path.name,
                )
                marked += 1
            print(f"migrate: baselined {marked} migration(s) through {target} without running them")
            return 0

        # --apply. Refuse to run the full schema over a database that already
        # has tables but no tracking: that is a baseline job, not an apply.
        if not done and await count_user_tables(conn) > 0:
            sys.exit(
                "migrate: database has tables but no schema_migrations history. "
                "Check which migrations it already has, then run --baseline-through NNN."
            )

        await conn.execute(TRACKING_TABLE_SQL)
        if not pending:
            print("migrate: nothing to apply")
            return 0
        for version, path in pending:
            print(f"migrate: applying {path.name}")
            try:
                async with conn.transaction():
                    await conn.execute(TOP_LEVEL_TRANSACTION.sub("", path.read_text()))
                    await conn.execute(
                        "INSERT INTO schema_migrations (version, filename) VALUES ($1, $2)",
                        version, path.name,
                    )
            except asyncpg.PostgresError as error:
                print(f"migrate: FAILED on {path.name}, rolled back: {type(error).__name__}: {error}", file=sys.stderr)
                return 1
        print(f"migrate: applied {len(pending)} migration(s)")
        return 0
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true", help="show applied and pending migrations (read-only)")
    mode.add_argument("--apply", action="store_true", help="apply pending migrations in order")
    mode.add_argument("--baseline-through", metavar="NNN", help="record migrations up to NNN as applied without running them")
    sys.exit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
