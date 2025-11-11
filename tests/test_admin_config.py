"""
Tests for AdminConfig class
"""
import pytest
import json
import os
from pathlib import Path
from admin_config import AdminConfig


class TestAdminConfig:
    """Test AdminConfig functionality"""

    def test_init_creates_config_file(self, temp_dir, monkeypatch):
        """Test that AdminConfig creates config file if it doesn't exist"""
        monkeypatch.chdir(temp_dir)
        config = AdminConfig()
        assert Path("admin_config.json").exists()

    def test_default_password(self, temp_dir, monkeypatch):
        """Test that default password is set"""
        monkeypatch.chdir(temp_dir)
        config = AdminConfig()
        assert config.verify_password("admin123")

    def test_password_verification(self, temp_dir, monkeypatch):
        """Test password verification"""
        monkeypatch.chdir(temp_dir)
        config = AdminConfig()

        # Test correct password
        assert config.verify_password("admin123")

        # Test incorrect password
        assert not config.verify_password("wrongpassword")

    def test_change_password(self, temp_dir, monkeypatch):
        """Test changing admin password"""
        monkeypatch.chdir(temp_dir)
        config = AdminConfig()

        # Change password
        config.change_password("newpassword123")

        # Old password should not work
        assert not config.verify_password("admin123")

        # New password should work
        assert config.verify_password("newpassword123")

    def test_password_persistence(self, temp_dir, monkeypatch):
        """Test that password persists across instances"""
        monkeypatch.chdir(temp_dir)

        # Create first instance and change password
        config1 = AdminConfig()
        config1.change_password("testpass123")

        # Create second instance and verify password
        config2 = AdminConfig()
        assert config2.verify_password("testpass123")

    def test_create_session(self, temp_dir, monkeypatch):
        """Test session creation"""
        monkeypatch.chdir(temp_dir)
        config = AdminConfig()

        session_id = "test-session-123"
        config.create_session(session_id)

        assert config.is_valid_session(session_id)

    def test_invalid_session(self, temp_dir, monkeypatch):
        """Test invalid session detection"""
        monkeypatch.chdir(temp_dir)
        config = AdminConfig()

        assert not config.is_valid_session("nonexistent-session")

    def test_session_expiration(self, temp_dir, monkeypatch, freezegun):
        """Test session expiration"""
        monkeypatch.chdir(temp_dir)
        config = AdminConfig()

        import time
        from datetime import datetime, timedelta

        session_id = "test-session-123"
        config.create_session(session_id)

        # Session should be valid initially
        assert config.is_valid_session(session_id)

        # Mock time passing (default session timeout is typically 24 hours)
        with freezegun.freeze_time(datetime.now() + timedelta(hours=25)):
            assert not config.is_valid_session(session_id)

    def test_log_usage(self, temp_dir, monkeypatch):
        """Test usage logging"""
        monkeypatch.chdir(temp_dir)
        config = AdminConfig()

        # Log some usage
        config.log_usage("gpt-4o-mini", 1000)
        config.log_usage("gpt-4o-mini", 500)
        config.log_usage("o3-mini", 2000)

        # Verify usage was logged
        usage_file = Path("usage_stats.json")
        assert usage_file.exists()

        with open(usage_file, "r") as f:
            usage_data = json.load(f)

        assert len(usage_data) > 0

    def test_get_usage_summary(self, temp_dir, monkeypatch):
        """Test usage summary generation"""
        monkeypatch.chdir(temp_dir)
        config = AdminConfig()

        # Log usage for different models
        config.log_usage("gpt-4o-mini", 1000)
        config.log_usage("o3-mini", 2000)

        # Get usage summary
        summary = config.get_usage_summary(days=30)

        assert "total_requests" in summary
        assert "total_tokens" in summary
        assert "total_cost" in summary
        assert "models" in summary

        assert summary["total_requests"] >= 2
        assert summary["total_tokens"] >= 3000

    def test_get_default_user_prompt(self, temp_dir, monkeypatch):
        """Test getting default user prompt"""
        monkeypatch.chdir(temp_dir)
        config = AdminConfig()

        prompt = config.get_default_user_prompt()
        # Default should be empty or None
        assert prompt is None or prompt == ""

    def test_config_file_structure(self, temp_dir, monkeypatch):
        """Test admin config file has correct structure"""
        monkeypatch.chdir(temp_dir)
        config = AdminConfig()

        config_file = Path("admin_config.json")
        assert config_file.exists()

        with open(config_file, "r") as f:
            config_data = json.load(f)

        # Verify required fields
        assert "admin_password_hash" in config_data
        assert "session_timeout_hours" in config_data or "settings" in config_data

    def test_multiple_sessions(self, temp_dir, monkeypatch):
        """Test handling multiple concurrent sessions"""
        monkeypatch.chdir(temp_dir)
        config = AdminConfig()

        # Create multiple sessions
        session1 = "session-1"
        session2 = "session-2"
        session3 = "session-3"

        config.create_session(session1)
        config.create_session(session2)
        config.create_session(session3)

        # All sessions should be valid
        assert config.is_valid_session(session1)
        assert config.is_valid_session(session2)
        assert config.is_valid_session(session3)
