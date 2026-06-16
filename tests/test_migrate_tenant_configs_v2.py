"""Tests for Phase 7 tenant JSON migration script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.migrations.migrate_tenant_configs_v2 import (
    run_migration,
    transform_tenant_dict,
)


def test_transform_injects_secrets_backend_and_secret_refs() -> None:
    raw = {
        "client_id": "acme",
        "embedder": {
            "type": "gemini",
            "gemini": {"model": "gemini-embedding-2", "api_key_env": "GOOGLE_API_KEY"},
        },
        "llm": {
            "single": {
                "type": "gemini",
                "model": "gemini-2.5-flash",
                "api_key_env": "GOOGLE_API_KEY",
                "base_url": "https://generativelanguage.googleapis.com/v1",
            }
        },
        "vectordb": {
            "type": "chroma",
            "collection": "docs",
            "chroma": {"persist_directory": "./db"},
        },
    }
    out = transform_tenant_dict(raw)
    assert out["secrets_backend"] == {"provider": "env"}
    assert out["embedder"]["gemini"]["secret_ref"] == {"uri": "env://GOOGLE_API_KEY"}
    assert "api_key_env" not in out["embedder"]["gemini"]
    assert out["llm"]["single"]["secret_ref"] == {"uri": "env://GOOGLE_API_KEY"}


def test_run_migration_dry_run_on_temp_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    legacy = {
        "client_id": "default",
        "vectordb": {
            "type": "chroma",
            "collection": "ingested_content",
            "chroma": {"persist_directory": "./pluggable_db"},
        },
        "embedder": {
            "type": "gemini",
            "gemini": {"model": "gemini-embedding-2", "api_key_env": "GOOGLE_API_KEY"},
        },
        "llm": {
            "single": {
                "type": "gemini",
                "model": "gemini-2.5-flash",
                "api_key_env": "GOOGLE_API_KEY",
                "base_url": "https://generativelanguage.googleapis.com/v1",
            }
        },
    }
    (cfg_dir / "default.json").write_text(json.dumps(legacy, indent=2), encoding="utf-8")

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    results = run_migration(cfg_dir, dry_run=True, backup=False)
    assert results
    assert all(not r.errors for r in results)
    assert any(r.changed for r in results)

    # Dry-run must not modify files on disk.
    reloaded = json.loads((cfg_dir / "default.json").read_text(encoding="utf-8"))
    assert "api_key_env" in reloaded["embedder"]["gemini"]
