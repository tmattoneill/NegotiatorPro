"""Object storage backed by Bunny Storage, used as if it were local.

The corpus (and later the vectorstore and user uploads) live canonically in
Bunny Storage so admin works from any machine, not just the dev box: an upload
from a laptop and a rebuild on the server both read and write the same remote
copy. This module is the read-through (get / sync_down) + write-through
(put / sync_up) layer that makes remote files behave like local ones.

It degrades gracefully: if Bunny is not configured (no env), `is_configured()`
returns False and callers fall back to plain local-filesystem behaviour, so
local dev and tests keep working without any cloud credentials.

Config (environment, or .env.local for local dev):
  BUNNY_NET_URL          https://<region>.storage.bunnycdn.com/<zone>
  BUNNY_NET_RO_PASSWORD  read-only AccessKey  (get, list)
  BUNNY_NET_RW_PASSWORD  read-write AccessKey (put, delete)
  BUNNY_NET_CDN_URL      optional pull-zone base for browser-facing URLs

Bunny Storage API: list = GET on a path ending '/', download = GET on the file
path, upload = PUT, delete = DELETE; all authenticated with the `AccessKey`
header. https://docs.bunny.net/reference/storage-api
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_TIMEOUT = 120


class StorageError(Exception):
    """Raised when a Bunny Storage operation fails."""


class ObjectStore:
    """Thin client over a single Bunny Storage zone."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        ro_key: Optional[str] = None,
        rw_key: Optional[str] = None,
        cdn_url: Optional[str] = None,
    ):
        self.base_url = (base_url or os.getenv("BUNNY_NET_URL", "")).rstrip("/")
        self.ro_key = ro_key or os.getenv("BUNNY_NET_RO_PASSWORD", "")
        self.rw_key = rw_key or os.getenv("BUNNY_NET_RW_PASSWORD", "")
        # Pull-zone for browser-facing URLs: accept a full URL (BUNNY_NET_CDN_URL)
        # or a bare hostname / custom domain (BUNNY_NET_PULL_ZONE, e.g.
        # data.amfonica.com), normalising the latter to https://<host>.
        cdn = (cdn_url or os.getenv("BUNNY_NET_CDN_URL") or os.getenv("BUNNY_NET_PULL_ZONE", "")).strip().rstrip("/")
        if cdn and not cdn.startswith(("http://", "https://")):
            cdn = f"https://{cdn}"
        self.cdn_base = cdn

    # -- capability checks ----------------------------------------------------
    def is_configured(self) -> bool:
        """True if at least read access is configured."""
        return bool(self.base_url and self.ro_key)

    def can_write(self) -> bool:
        return bool(self.base_url and self.rw_key)

    # -- low-level HTTP -------------------------------------------------------
    def _request(self, method: str, rel: str, key: str, data: Optional[bytes] = None):
        if not self.base_url:
            raise StorageError("BUNNY_NET_URL not configured")
        if not key:
            raise StorageError(f"missing AccessKey for {method} (RO/RW password not set)")
        url = f"{self.base_url}/{rel.lstrip('/')}"
        req = urllib.request.Request(url, method=method, data=data)
        req.add_header("AccessKey", key)
        if data is not None:
            req.add_header("Content-Type", "application/octet-stream")
        try:
            return urllib.request.urlopen(req, timeout=_TIMEOUT)
        except urllib.error.HTTPError as e:
            raise StorageError(f"{method} {rel} -> HTTP {e.code} {e.reason}") from e
        except urllib.error.URLError as e:
            raise StorageError(f"{method} {rel} -> {e.reason}") from e

    # -- operations ----------------------------------------------------------
    def list(self, prefix: str = "") -> List[str]:
        """Recursively list file paths under prefix (relative to the zone root)."""
        prefix = prefix.strip("/")
        listing_path = f"{prefix}/" if prefix else ""
        with self._request("GET", listing_path, self.ro_key) as resp:
            entries = json.loads(resp.read().decode())
        files: List[str] = []
        for obj in entries:
            name = obj["ObjectName"]
            path = f"{prefix}/{name}" if prefix else name
            if obj.get("IsDirectory"):
                files.extend(self.list(path))
            else:
                files.append(path)
        return files

    def exists(self, remote_path: str) -> bool:
        try:
            with self._request("GET", remote_path, self.ro_key):
                return True
        except StorageError:
            return False

    def get(self, remote_path: str) -> bytes:
        with self._request("GET", remote_path, self.ro_key) as resp:
            return resp.read()

    def get_to(self, remote_path: str, local_path: Path) -> Path:
        """Download a file to local_path, creating parent dirs."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.get(remote_path))
        return local_path

    def put(self, remote_path: str, data: bytes) -> None:
        with self._request("PUT", remote_path, self.rw_key, data=data):
            pass

    def put_file(self, remote_path: str, local_path: Path) -> None:
        with open(local_path, "rb") as fh:
            self.put(remote_path, fh.read())

    def delete(self, remote_path: str) -> None:
        with self._request("DELETE", remote_path, self.rw_key):
            pass

    # -- bulk sync (the "looks local" part) ----------------------------------
    def sync_down(self, prefix: str, local_dir: Path, force: bool = False) -> int:
        """Mirror remote prefix -> local_dir. Returns number of files written.

        Skips files already present locally with a matching size unless force.
        """
        local_dir = Path(local_dir)
        written = 0
        for rel in self.list(prefix):
            # store under local_dir using the path relative to prefix
            sub = rel[len(prefix):].lstrip("/") if prefix and rel.startswith(prefix) else rel
            target = local_dir / sub
            data = self.get(rel)
            if target.exists() and target.stat().st_size == len(data) and not force:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            written += 1
        return written

    def sync_up(self, local_dir: Path, prefix: str = "") -> int:
        """Mirror local_dir -> remote prefix. Returns number of files uploaded."""
        local_dir = Path(local_dir)
        count = 0
        for p in sorted(local_dir.rglob("*")):
            if not p.is_file() or p.name.startswith("."):
                continue
            rel = p.relative_to(local_dir).as_posix()
            remote = f"{prefix.strip('/')}/{rel}" if prefix.strip("/") else rel
            self.put_file(remote, p)
            count += 1
        return count

    # -- browser-facing URLs (pull zone) -------------------------------------
    def cdn_url(self, remote_path: str) -> Optional[str]:
        """Public pull-zone URL for a file, or None if no CDN is configured."""
        if not self.cdn_base:
            return None
        return f"{self.cdn_base}/{remote_path.lstrip('/')}"


# Module-level singleton built from the environment.
store = ObjectStore()
