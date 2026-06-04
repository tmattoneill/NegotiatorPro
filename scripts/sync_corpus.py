#!/usr/bin/env python3
"""Sync the RAG corpus between the local filesystem and Bunny Storage.

Bunny Storage is the canonical home of the source PDFs. Build the vectorstore
by pulling the corpus locally, building the index, then shipping only the index
(the box is serving-only and never needs the PDFs).

Auth + endpoint come from the environment, falling back to .env.local:
  BUNNY_NET_URL          e.g. https://uk.storage.bunnycdn.com/amfonica-data-sources
  BUNNY_NET_RO_PASSWORD  read-only AccessKey (needed for `pull`)
  BUNNY_NET_RW_PASSWORD  read-write AccessKey (needed for `push`)

Usage:
  python scripts/sync_corpus.py pull [--dest ../data-sources] [--force]
  python scripts/sync_corpus.py push [--src  ../data-sources] [--dry-run]
  python scripts/sync_corpus.py list

The Bunny Storage API: list = GET on a path ending '/', download = GET on the
file path, upload = PUT. All authenticated with the `AccessKey` header.
https://docs.bunny.net/reference/storage-api
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def load_env_local() -> None:
    """Load .env.local into os.environ without overriding existing values."""
    env_file = REPO / ".env.local"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


def base_url() -> str:
    url = os.environ.get("BUNNY_NET_URL")
    if not url:
        sys.exit("BUNNY_NET_URL not set (env or .env.local)")
    return url.rstrip("/")


def request(method: str, url: str, key: str, data: bytes | None = None):
    req = urllib.request.Request(url, method=method, data=data)
    req.add_header("AccessKey", key)
    if data is not None:
        req.add_header("Content-Type", "application/octet-stream")
    return urllib.request.urlopen(req, timeout=120)


def list_dir(base: str, key: str, rel: str = "") -> list[dict]:
    """List one storage directory (non-recursive). rel is '' or 'sales/'."""
    url = f"{base}/{rel}" if rel else f"{base}/"
    with request("GET", url, key) as resp:
        return json.loads(resp.read().decode())


def walk(base: str, key: str, rel: str = "") -> list[str]:
    """Return all file paths (relative to the zone root) under rel, recursively."""
    files: list[str] = []
    for obj in list_dir(base, key, rel):
        name = obj["ObjectName"]
        path = f"{rel}{name}"
        if obj.get("IsDirectory"):
            files.extend(walk(base, key, f"{path}/"))
        else:
            files.append(path)
    return files


def cmd_list(args) -> None:
    base, key = base_url(), require_key("ro")
    for p in sorted(walk(base, key)):
        print(p)


def cmd_pull(args) -> None:
    base, key = base_url(), require_key("ro")
    dest = (REPO / args.dest).resolve() if not os.path.isabs(args.dest) else Path(args.dest)
    paths = sorted(walk(base, key))
    if not paths:
        sys.exit("zone is empty — nothing to pull")
    pulled = skipped = 0
    for rel in paths:
        target = dest / rel
        url = f"{base}/{rel}"
        with request("GET", url, key) as resp:
            body = resp.read()
        if target.exists() and target.stat().st_size == len(body) and not args.force:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        pulled += 1
        print(f"  pulled {rel} ({len(body):,} bytes)")
    print(f"done: {pulled} pulled, {skipped} already current -> {dest}")


def cmd_push(args) -> None:
    base, key = base_url(), require_key("rw")
    src = (REPO / args.src).resolve() if not os.path.isabs(args.src) else Path(args.src)
    if not src.is_dir():
        sys.exit(f"source dir not found: {src}")
    files = [p for p in src.rglob("*") if p.is_file() and not p.name.startswith(".")]
    if not files:
        sys.exit(f"no files under {src}")
    pushed = 0
    for p in sorted(files):
        rel = p.relative_to(src).as_posix()
        url = f"{base}/{rel}"
        if args.dry_run:
            print(f"  would push {rel} ({p.stat().st_size:,} bytes)")
            continue
        with open(p, "rb") as fh:
            request("PUT", url, key, data=fh.read())
        pushed += 1
        print(f"  pushed {rel} ({p.stat().st_size:,} bytes)")
    print(f"done: {len(files) if args.dry_run else pushed} file(s){' (dry-run)' if args.dry_run else ''} -> {base}")


def require_key(kind: str) -> str:
    var = "BUNNY_NET_RO_PASSWORD" if kind == "ro" else "BUNNY_NET_RW_PASSWORD"
    key = os.environ.get(var)
    if not key:
        sys.exit(f"{var} not set — needed for this operation (env or .env.local)")
    return key


def main() -> None:
    load_env_local()
    ap = argparse.ArgumentParser(description="Sync the RAG corpus with Bunny Storage")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list all files in the storage zone")

    p_pull = sub.add_parser("pull", help="download the corpus from Bunny")
    p_pull.add_argument("--dest", default="../data-sources", help="local dir to write to")
    p_pull.add_argument("--force", action="store_true", help="overwrite even if size matches")

    p_push = sub.add_parser("push", help="upload a local corpus dir to Bunny (needs RW key)")
    p_push.add_argument("--src", default="../data-sources", help="local dir to upload")
    p_push.add_argument("--dry-run", action="store_true", help="show what would upload, do nothing")

    args = ap.parse_args()
    {"list": cmd_list, "pull": cmd_pull, "push": cmd_push}[args.cmd](args)


if __name__ == "__main__":
    main()
