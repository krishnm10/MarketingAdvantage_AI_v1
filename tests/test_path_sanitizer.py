"""
Tests for app/utils/path_sanitizer.py
Covers: sanitize_client_id, safe_config_path, validate_slug
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


class TestSanitizeClientId:
    """Test client ID sanitization and path traversal prevention."""

    def test_normal_id_passes(self):
        from app.utils.path_sanitizer import sanitize_client_id
        assert sanitize_client_id("acme_corp") == "acme_corp"

    def test_uuid_passes(self):
        from app.utils.path_sanitizer import sanitize_client_id
        result = sanitize_client_id("550e8400-e29b-41d4-a716-446655440000")
        assert result == "550e8400-e29b-41d4-a716-446655440000"

    def test_none_returns_default(self):
        from app.utils.path_sanitizer import sanitize_client_id
        assert sanitize_client_id(None) == "default"

    def test_empty_returns_default(self):
        from app.utils.path_sanitizer import sanitize_client_id
        assert sanitize_client_id("") == "default"

    def test_path_traversal_blocked(self):
        from app.utils.path_sanitizer import sanitize_client_id
        with pytest.raises(ValueError, match="traversal"):
            sanitize_client_id("../../etc/passwd")

    def test_slash_blocked(self):
        from app.utils.path_sanitizer import sanitize_client_id
        with pytest.raises(ValueError):
            sanitize_client_id("foo/bar")

    def test_backslash_blocked(self):
        from app.utils.path_sanitizer import sanitize_client_id
        with pytest.raises(ValueError):
            sanitize_client_id("foo\\bar")

    def test_uppercase_lowered(self):
        from app.utils.path_sanitizer import sanitize_client_id
        assert sanitize_client_id("AcMe_Corp") == "acme_corp"

    def test_special_chars_stripped(self):
        from app.utils.path_sanitizer import sanitize_client_id
        result = sanitize_client_id("acme@corp!#$%")
        assert "@" not in result
        assert "!" not in result

    def test_max_length_enforced(self):
        from app.utils.path_sanitizer import sanitize_client_id
        long_id = "a" * 200
        result = sanitize_client_id(long_id)
        assert len(result) <= 64


class TestSafeConfigPath:
    """Test safe config path construction with containment checks."""

    def test_normal_path(self, tmp_path):
        from app.utils.path_sanitizer import safe_config_path
        result = safe_config_path(tmp_path, "acme_corp", "json")
        assert result.name == "acme_corp.json"
        assert str(result.resolve()).startswith(str(tmp_path.resolve()))

    def test_containment_violation(self, tmp_path):
        from app.utils.path_sanitizer import safe_config_path
        with pytest.raises(ValueError):
            safe_config_path(tmp_path, "../../etc/passwd", "json")

    def test_yaml_extension(self, tmp_path):
        from app.utils.path_sanitizer import safe_config_path
        result = safe_config_path(tmp_path, "client1", "yaml")
        assert result.name == "client1.yaml"


class TestValidateSlug:
    """Test slug validation for template IDs and similar identifiers."""

    def test_valid_slug(self):
        from app.utils.path_sanitizer import validate_slug
        assert validate_slug("basic-rag") == "basic-rag"

    def test_valid_underscore(self):
        from app.utils.path_sanitizer import validate_slug
        assert validate_slug("secure_rag_v2") == "secure_rag_v2"

    def test_empty_rejected(self):
        from app.utils.path_sanitizer import validate_slug
        with pytest.raises(ValueError):
            validate_slug("")

    def test_special_chars_rejected(self):
        from app.utils.path_sanitizer import validate_slug
        with pytest.raises(ValueError):
            validate_slug("foo/bar")

    def test_too_long_rejected(self):
        from app.utils.path_sanitizer import validate_slug
        with pytest.raises(ValueError):
            validate_slug("a" * 100, max_length=64)

    def test_uppercase_normalized_to_lowercase(self):
        from app.utils.path_sanitizer import validate_slug
        assert validate_slug("FooBar") == "foobar"
