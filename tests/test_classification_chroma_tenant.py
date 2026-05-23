"""
Gate 4 — classification + chroma_search_service tenant isolation tests.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from pathlib import Path

import pytest

from app.services.retrieval import chroma_search_service


GATE4_SCOPED_FILES = [
    Path("app/services/classification/classification_service.py"),
    Path("app/services/retrieval/chroma_search_service.py"),
]

FORBIDDEN_PATTERNS = (
    'os.getenv("CHROMA_PATH"',
    'os.getenv("MAI_COLLECTION"',
    "_get_fallback_vectordb",
    "_CHROMA_CLIENT",
)


class TestGate4EnvGrepGuard:
    def test_scoped_files_have_no_env_tenant_routing(self):
        violations: list[str] = []
        for rel in GATE4_SCOPED_FILES:
            text = rel.read_text(encoding="utf-8")
            for pat in FORBIDDEN_PATTERNS:
                if pat in text:
                    violations.append(f"{rel}: contains {pat!r}")
        assert not violations, "\n".join(violations)

    def test_classification_service_requires_client_id_in_source(self):
        text = Path("app/services/classification/classification_service.py").read_text(
            encoding="utf-8"
        )
        assert "_get_fallback_vectordb" not in text
        assert re.search(r"client_id:\s*str", text), "classify_chunk must require client_id: str"


class TestGate4RequiredClientId:
    def test_get_chroma_collection_rejects_empty_client_id(self):
        with pytest.raises(ValueError, match="client_id is required"):
            chroma_search_service.get_chroma_collection("")

    def test_health_check_rejects_empty_client_id(self):
        with pytest.raises(ValueError, match="client_id is required"):
            chroma_search_service.health_check("")

    def test_semantic_search_rejects_empty_client_id(self):
        with pytest.raises(ValueError, match="client_id is required"):
            asyncio.run(chroma_search_service.semantic_search([0.1], ""))

    def test_public_functions_require_client_id_parameter(self):
        for fn in (
            chroma_search_service.get_chroma_collection,
            chroma_search_service.health_check,
            chroma_search_service.semantic_search,
        ):
            sig = inspect.signature(fn)
            assert "client_id" in sig.parameters
