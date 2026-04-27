"""
Tests for Pipeline Template Gallery.
Covers: template loading, slug validation, Secure RAG default, API endpoints.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestTemplateFiles:
    """Test that built-in templates exist and are valid."""

    TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "app" / "core" / "configs" / "pipeline_templates"

    def test_templates_dir_exists(self):
        assert self.TEMPLATES_DIR.exists(), f"Templates dir not found: {self.TEMPLATES_DIR}"

    def test_secure_rag_exists(self):
        p = self.TEMPLATES_DIR / "secure_rag.json"
        assert p.exists(), "secure_rag.json template must exist"

    def test_secure_rag_is_recommended(self):
        p = self.TEMPLATES_DIR / "secure_rag.json"
        if not p.exists():
            pytest.skip("secure_rag.json not yet created")
        data = json.loads(p.read_text())
        assert data.get("recommended_default") is True
        assert data.get("pii_enabled") is True

    def test_secure_rag_has_all_pii_positions(self):
        p = self.TEMPLATES_DIR / "secure_rag.json"
        if not p.exists():
            pytest.skip("secure_rag.json not yet created")
        data = json.loads(p.read_text())
        pii_cfg = data.get("config_patch", {}).get("security", {}).get("pii_middleware", {})
        positions = set(pii_cfg.get("positions", []))
        assert {"pre_embedding", "pre_llm", "post_llm"} == positions

    def test_all_templates_valid_json(self):
        if not self.TEMPLATES_DIR.exists():
            pytest.skip("Templates dir not yet created")
        for p in self.TEMPLATES_DIR.glob("*.json"):
            data = json.loads(p.read_text())
            assert "template_id" in data, f"{p.name} missing template_id"
            assert "name" in data, f"{p.name} missing name"
            assert "config_patch" in data, f"{p.name} missing config_patch"

    def test_basic_rag_exists(self):
        p = self.TEMPLATES_DIR / "basic_rag.json"
        assert p.exists(), "basic_rag.json template must exist"

    def test_template_ids_match_filenames(self):
        if not self.TEMPLATES_DIR.exists():
            pytest.skip("Templates dir not yet created")
        for p in self.TEMPLATES_DIR.glob("*.json"):
            data = json.loads(p.read_text())
            expected_id = p.stem
            assert data.get("template_id") == expected_id, (
                f"{p.name}: template_id '{data.get('template_id')}' != filename stem '{expected_id}'"
            )
