#!/usr/bin/env python3
"""
Set the 'admin' user's password to ADMIN_PASSWORD, writing only when the stored
hash doesn't already match.

Migration 001 seeds a placeholder hash that matches no known password, so
without this a fresh environment has an admin nobody can log in as.

Piped into the backend container by the deploy scripts:
    docker compose exec -T backend python3 - < scripts/sync_admin_password.py
"""
import asyncio
import os
import sys

import asyncpg
import bcrypt


async def main() -> int:
    password = os.environ.get("ADMIN_PASSWORD")
    if not password:
        print("    ADMIN_PASSWORD not set; skipping")
        return 0

    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        stored_hash = await conn.fetchval("SELECT password_hash FROM users WHERE username = 'admin'")
        if stored_hash is None:
            print("    no 'admin' user found; skipping", file=sys.stderr)
            return 1
        if bcrypt.checkpw(password.encode(), stored_hash.encode()):
            print("    admin password already in sync")
            return 0
        new_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(12)).decode()
        await conn.execute(
            "UPDATE users SET password_hash = $1, is_active = TRUE WHERE username = 'admin'",
            new_hash,
        )
        print("    admin password synced to ADMIN_PASSWORD")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
