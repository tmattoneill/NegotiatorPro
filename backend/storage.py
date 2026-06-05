"""Object storage backed by Bunny Storage, used as if it were local.

The corpus (and later the vectorstore and user uploads) live canonically in
Bunny Storage so admin works from any machine, not just the dev box: an upload
from a laptop and a rebuild on the server both read and write the same remote
copy. This module is the read-through (get / sync_down) + write-through
(put / sync_up) layer that makes remote files behave like local ones.

It degrades gracefully: if Bunny is not configured (no env), `is_configured()`
returns False and callers fall back to plain local-filesystem behaviour, so
local dev and tests keep working without any cloud credentials.

Config (environment, or .env for local dev):
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
        # Config comes from the explicit args only. Env reading (and any legacy
        # fallback) lives in _make_store, so an unconfigured zone stays
        # unconfigured rather than silently resolving to another zone.
        self.base_url = (base_url or "").rstrip("/")
        self.ro_key = ro_key or ""
        self.rw_key = rw_key or ""
        # Pull-zone for browser-facing URLs: accept a full URL or a bare host /
        # custom domain (e.g. data.amfonica.com), normalising to https://<host>.
        cdn = (cdn_url or "").strip().rstrip("/")
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


def _make_store(prefix: str, *, legacy_corpus: bool = False) -> ObjectStore:
    """Build an ObjectStore from BUNNY_<prefix>_{URL,RO_PASSWORD,RW_PASSWORD}.

    The corpus store also accepts the legacy BUNNY_NET_* names so existing
    config keeps working.
    """
    def g(suffix: str) -> Optional[str]:
        val = os.getenv(f"{prefix}_{suffix}")
        if not val and legacy_corpus:
            val = os.getenv(f"BUNNY_NET_{suffix}")
        return val

    return ObjectStore(
        base_url=g("URL"),
        ro_key=g("RO_PASSWORD"),
        rw_key=g("RW_PASSWORD"),
        cdn_url=os.getenv("BUNNY_NET_PULL_ZONE"),
    )


# One store per Bunny zone. Each degrades to "not configured" if its env is
# absent, so partial setup (e.g. corpus only) is fine.
corpus = _make_store("BUNNY_CORPUS", legacy_corpus=True)   # amfonica-data-sources
uploads = _make_store("BUNNY_UPLOADS")                     # amfonica-user-data
vectorstores = _make_store("BUNNY_VECTORSTORES")           # amfonica-vectorstores

# Backwards-compatible alias for the corpus store.
store = corpus


def vectorstore_prefix() -> str:
    """Path prefix inside the vectorstores zone, keyed by environment.

    dev/prod/local share one zone, so each writes under its own prefix
    (dev/, prod/, local/) to avoid clobbering the others' index.
    """
    return os.getenv("DEPLOY_ENV", "dev")


def should_push_vectorstore() -> bool:
    """Whether a rebuild should push the new index to the vectorstores zone.

    Only when the zone is writable AND we're not a throwaway local env — a
    laptop (DEPLOY_ENV=local) must never overwrite the box's shared index.
    """
    return vectorstores.can_write() and os.getenv("DEPLOY_ENV", "dev") != "local"
