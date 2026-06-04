"""Unit tests for backend/storage.py (Bunny Storage layer).

These are pure-logic tests — no network. They cover the graceful-degradation
guards and the env-prefix helpers that gate the write-through / push behaviour
wired into the admin RAG routes.
"""
import importlib

import pytest

from backend import storage
from backend.storage import ObjectStore


def test_unconfigured_store_is_inert():
    """No URL/keys -> not configured, not writable (pure-local fallback)."""
    s = ObjectStore(base_url=None, ro_key=None, rw_key=None)
    assert s.is_configured() is False
    assert s.can_write() is False
    assert s.cdn_url("x/y.pdf") is None


def test_configured_read_only_vs_read_write():
    ro = ObjectStore(base_url="https://uk.storage.bunnycdn.com/zone", ro_key="r")
    assert ro.is_configured() is True
    assert ro.can_write() is False

    rw = ObjectStore(base_url="https://uk.storage.bunnycdn.com/zone", ro_key="r", rw_key="w")
    assert rw.is_configured() is True
    assert rw.can_write() is True


def test_no_silent_fallback_between_zones():
    """An empty store must NOT inherit another zone's config from env."""
    s = ObjectStore(base_url="", ro_key="", rw_key="")
    assert s.is_configured() is False


def test_cdn_url_normalises_bare_host():
    s = ObjectStore(base_url="https://x/z", ro_key="r", cdn_url="data.amfonica.com")
    assert s.cdn_url("negotiation/a.pdf") == "https://data.amfonica.com/negotiation/a.pdf"

    s2 = ObjectStore(base_url="https://x/z", ro_key="r", cdn_url="https://cdn.example.com")
    assert s2.cdn_url("a.pdf") == "https://cdn.example.com/a.pdf"


def test_vectorstore_prefix_defaults_to_dev(monkeypatch):
    monkeypatch.delenv("DEPLOY_ENV", raising=False)
    assert storage.vectorstore_prefix() == "dev"
    monkeypatch.setenv("DEPLOY_ENV", "prod")
    assert storage.vectorstore_prefix() == "prod"


def test_should_push_vectorstore_guards(monkeypatch):
    # writable + non-local -> push
    monkeypatch.setenv("DEPLOY_ENV", "dev")
    monkeypatch.setattr(storage.vectorstores, "can_write", lambda: True)
    assert storage.should_push_vectorstore() is True

    # local env -> never push (don't clobber the shared index)
    monkeypatch.setenv("DEPLOY_ENV", "local")
    assert storage.should_push_vectorstore() is False

    # not writable -> don't push
    monkeypatch.setenv("DEPLOY_ENV", "dev")
    monkeypatch.setattr(storage.vectorstores, "can_write", lambda: False)
    assert storage.should_push_vectorstore() is False


def test_sync_down_strips_prefix(monkeypatch, tmp_path):
    """sync_down writes remote 'dev/index.faiss' to local '<dir>/index.faiss'."""
    s = ObjectStore(base_url="https://x/z", ro_key="r")
    monkeypatch.setattr(s, "list", lambda prefix="": ["dev/index.faiss", "dev/index.pkl"])
    monkeypatch.setattr(s, "get", lambda rel: b"data")
    written = s.sync_down("dev", tmp_path)
    assert written == 2
    assert (tmp_path / "index.faiss").read_bytes() == b"data"
    assert (tmp_path / "index.pkl").exists()
    assert not (tmp_path / "dev").exists()  # prefix stripped, not nested
