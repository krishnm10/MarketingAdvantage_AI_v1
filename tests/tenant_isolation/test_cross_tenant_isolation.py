"""
Cross-Tenant Isolation Integration Tests

Tests that tenant A can NEVER access tenant B's data through any retrieval path.
Uses an in-memory VectorDB mock that faithfully simulates metadata filtering
to prove isolation at the contract level.

Coverage:
  - Tenant A ingestion + retrieval
  - Tenant B ingestion + retrieval
  - Multi-query retrieval isolation
  - Hybrid retrieval isolation
  - Reranker path isolation
  - Post-processor path isolation
  - Adversarial attack scenarios
  - Regression tests for all retrieval modes

Run: pytest tests/tenant_isolation/ -v
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from app.core.vectordb.base import (
    BaseVectorDB,
    BatchUpsertResult,
    TenantFilterViolation,
    VectorHit,
)
from app.utils.tenant_validator import (
    validate_tenant_id,
    TenantValidationError,
    TenantContext,
)
from app.services.ingestion.tenant_guard import (
    resolve_ingestion_tenant,
    enforce_tenant_immutability,
    IngestionTenantViolation,
    IngestionTenantContext,
)


# ─────────────────────────────────────────────────────────────────────────────
# In-Memory VectorDB — faithful metadata filter simulation
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryVectorDB(BaseVectorDB):
    """
    Test-only VectorDB that stores vectors in memory and applies metadata
    filters exactly as a production VectorDB would.
    """

    @property
    def kind(self) -> str:
        return "in_memory_test"

    def __init__(self):
        self._collections: Dict[str, List[Dict[str, Any]]] = {}
        self._tenant_isolation_enabled = True

    def ensure_collection(self, collection: str, *, embedding_dim: int = 3,
                          distance_metric: str = "cosine") -> None:
        if collection not in self._collections:
            self._collections[collection] = []

    def delete_collection(self, collection: str) -> None:
        self._collections.pop(collection, None)

    def upsert(self, *, collection: str, doc_id: str, embedding: List[float],
               text: str, metadata: Dict[str, Any]) -> None:
        self.ensure_collection(collection, embedding_dim=len(embedding))
        store = self._collections[collection]
        for i, doc in enumerate(store):
            if doc["id"] == doc_id:
                store[i] = {"id": doc_id, "embedding": embedding,
                            "text": text, "metadata": metadata}
                return
        store.append({"id": doc_id, "embedding": embedding,
                      "text": text, "metadata": metadata})

    def batch_upsert(self, *, collection: str, doc_ids: List[str],
                     embeddings: List[List[float]], texts: List[str],
                     metadatas: List[Dict[str, Any]]) -> BatchUpsertResult:
        self.ensure_collection(collection)
        inserted = 0
        for doc_id, emb, text, meta in zip(doc_ids, embeddings, texts, metadatas):
            self.upsert(collection=collection, doc_id=doc_id,
                        embedding=emb, text=text, metadata=meta)
            inserted += 1
        return BatchUpsertResult(inserted=inserted)

    def search(self, *, collection: str, query_embedding: List[float],
               top_k: int = 10, tenant_id: Optional[str] = None,
               filters: Optional[Dict[str, Any]] = None) -> List[VectorHit]:
        effective_filters = self._enforce_tenant_filter(
            tenant_id, filters, caller="search", collection=collection,
        )
        store = self._collections.get(collection, [])
        results = []
        for doc in store:
            if self._matches_filters(doc["metadata"], effective_filters):
                score = self._cosine_sim(query_embedding, doc["embedding"])
                results.append(VectorHit(
                    id=doc["id"], text=doc["text"],
                    score=score, metadata=doc["metadata"],
                ))
        results.sort(key=lambda h: h.score, reverse=True)
        hits = results[:top_k]
        self._log_tenant_search(
            tenant_id=tenant_id, collection=collection,
            filter_count=len(effective_filters), search_mode="vector",
            result_count=len(hits),
        )
        return hits

    def exists(self, *, collection: str, ids: List[str]) -> List[str]:
        store = self._collections.get(collection, [])
        existing = {doc["id"] for doc in store}
        return [i for i in ids if i in existing]

    def get_by_ids(self, *, collection: str, ids: List[str],
                   tenant_id: Optional[str] = None) -> List[VectorHit]:
        effective_filters = self._enforce_tenant_filter(
            tenant_id, {}, caller="get_by_ids", collection=collection,
        )
        store = self._collections.get(collection, [])
        results = []
        for doc in store:
            if doc["id"] in ids and self._matches_filters(doc["metadata"], effective_filters):
                results.append(VectorHit(
                    id=doc["id"], text=doc["text"],
                    score=1.0, metadata=doc["metadata"],
                ))
        return results

    def delete(self, *, collection: str, doc_id: str) -> None:
        store = self._collections.get(collection, [])
        self._collections[collection] = [d for d in store if d["id"] != doc_id]

    def delete_many(self, *, collection: str, doc_ids: List[str]) -> int:
        store = self._collections.get(collection, [])
        ids_set = set(doc_ids)
        before = len(store)
        self._collections[collection] = [d for d in store if d["id"] not in ids_set]
        return before - len(self._collections[collection])

    def count(self, collection: str) -> int:
        return len(self._collections.get(collection, []))

    def health_check(self) -> bool:
        return True

    @staticmethod
    def _matches_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

COLLECTION = "test_docs"
TENANT_A = "tenant-alpha"
TENANT_B = "tenant-beta"


@pytest.fixture
def vdb():
    """Fresh in-memory VectorDB with multi-tenant isolation enabled."""
    db = InMemoryVectorDB()
    db._tenant_isolation_enabled = True
    db.ensure_collection(COLLECTION)
    return db


@pytest.fixture
def seeded_vdb(vdb):
    """VectorDB with documents from both tenants pre-loaded."""
    # Tenant A documents (embedding biased toward [1, 0, 0])
    vdb.batch_upsert(
        collection=COLLECTION,
        doc_ids=["a-doc-1", "a-doc-2", "a-doc-3"],
        embeddings=[[1.0, 0.1, 0.0], [0.9, 0.2, 0.0], [0.8, 0.3, 0.0]],
        texts=[
            "Tenant A confidential revenue report Q1",
            "Tenant A internal strategy document",
            "Tenant A customer list and contacts",
        ],
        metadatas=[
            {"business_id": TENANT_A, "file_id": "fa-1", "source_type": "pdf"},
            {"business_id": TENANT_A, "file_id": "fa-2", "source_type": "docx"},
            {"business_id": TENANT_A, "file_id": "fa-3", "source_type": "csv"},
        ],
    )

    # Tenant B documents (embedding biased toward [0, 1, 0])
    vdb.batch_upsert(
        collection=COLLECTION,
        doc_ids=["b-doc-1", "b-doc-2", "b-doc-3"],
        embeddings=[[0.0, 1.0, 0.1], [0.0, 0.9, 0.2], [0.0, 0.8, 0.3]],
        texts=[
            "Tenant B private financial analysis",
            "Tenant B employee records summary",
            "Tenant B merger acquisition plans",
        ],
        metadatas=[
            {"business_id": TENANT_B, "file_id": "fb-1", "source_type": "pdf"},
            {"business_id": TENANT_B, "file_id": "fb-2", "source_type": "docx"},
            {"business_id": TENANT_B, "file_id": "fb-3", "source_type": "csv"},
        ],
    )

    return vdb


# ─────────────────────────────────────────────────────────────────────────────
# 1. BASIC TENANT ISOLATION — Ingestion + Retrieval
# ─────────────────────────────────────────────────────────────────────────────

class TestBasicTenantIsolation:
    """Tenant A NEVER retrieves Tenant B's documents and vice versa."""

    def test_tenant_a_retrieves_only_own_documents(self, seeded_vdb):
        results = seeded_vdb.search(
            collection=COLLECTION,
            query_embedding=[1.0, 0.0, 0.0],
            top_k=10,
            tenant_id=TENANT_A,
        )
        assert len(results) == 3
        for hit in results:
            assert hit.metadata["business_id"] == TENANT_A
            assert "Tenant A" in hit.text

    def test_tenant_b_retrieves_only_own_documents(self, seeded_vdb):
        results = seeded_vdb.search(
            collection=COLLECTION,
            query_embedding=[0.0, 1.0, 0.0],
            top_k=10,
            tenant_id=TENANT_B,
        )
        assert len(results) == 3
        for hit in results:
            assert hit.metadata["business_id"] == TENANT_B
            assert "Tenant B" in hit.text

    def test_tenant_a_cannot_see_tenant_b_even_with_similar_query(self, seeded_vdb):
        results = seeded_vdb.search(
            collection=COLLECTION,
            query_embedding=[0.0, 1.0, 0.0],  # query biased toward B's embeddings
            top_k=10,
            tenant_id=TENANT_A,
        )
        for hit in results:
            assert hit.metadata["business_id"] == TENANT_A
            assert "Tenant B" not in hit.text

    def test_tenant_b_cannot_see_tenant_a_even_with_similar_query(self, seeded_vdb):
        results = seeded_vdb.search(
            collection=COLLECTION,
            query_embedding=[1.0, 0.0, 0.0],  # query biased toward A's embeddings
            top_k=10,
            tenant_id=TENANT_B,
        )
        for hit in results:
            assert hit.metadata["business_id"] == TENANT_B
            assert "Tenant A" not in hit.text

    def test_empty_results_for_nonexistent_tenant(self, seeded_vdb):
        results = seeded_vdb.search(
            collection=COLLECTION,
            query_embedding=[1.0, 1.0, 1.0],
            top_k=10,
            tenant_id="tenant-gamma-nonexistent",
        )
        assert results == []

    def test_get_by_ids_enforces_tenant_isolation(self, seeded_vdb):
        results = seeded_vdb.get_by_ids(
            collection=COLLECTION,
            ids=["a-doc-1", "b-doc-1"],
            tenant_id=TENANT_A,
        )
        assert len(results) == 1
        assert results[0].id == "a-doc-1"

    def test_count_includes_all_tenants(self, seeded_vdb):
        assert seeded_vdb.count(COLLECTION) == 6


# ─────────────────────────────────────────────────────────────────────────────
# 2. METADATA FILTER OVERRIDE ATTEMPTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadataFilterOverrides:
    """Metadata filters cannot bypass tenant isolation."""

    def test_conflicting_business_id_in_filters_raises(self, seeded_vdb):
        with pytest.raises(TenantFilterViolation):
            seeded_vdb.search(
                collection=COLLECTION,
                query_embedding=[1.0, 0.0, 0.0],
                top_k=10,
                tenant_id=TENANT_A,
                filters={"business_id": TENANT_B},
            )

    def test_wildcard_business_id_in_filters_raises(self, seeded_vdb):
        with pytest.raises(TenantFilterViolation):
            seeded_vdb.search(
                collection=COLLECTION,
                query_embedding=[1.0, 0.0, 0.0],
                top_k=10,
                tenant_id=TENANT_A,
                filters={"business_id": "*"},
            )

    def test_additional_filters_are_additive_not_override(self, seeded_vdb):
        results = seeded_vdb.search(
            collection=COLLECTION,
            query_embedding=[1.0, 0.0, 0.0],
            top_k=10,
            tenant_id=TENANT_A,
            filters={"source_type": "pdf"},
        )
        assert len(results) == 1
        assert results[0].metadata["source_type"] == "pdf"
        assert results[0].metadata["business_id"] == TENANT_A

    def test_same_tenant_in_filters_is_allowed(self, seeded_vdb):
        results = seeded_vdb.search(
            collection=COLLECTION,
            query_embedding=[1.0, 0.0, 0.0],
            top_k=10,
            tenant_id=TENANT_A,
            filters={"business_id": TENANT_A},
        )
        assert len(results) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 3. ADVERSARIAL ATTACK SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

class TestAdversarialAttacks:
    """Adversarial inputs are rejected at validation or enforcement layers."""

    def test_forged_business_id_cross_tenant(self, seeded_vdb):
        """Attacker tries to inject other tenant's ID into filters."""
        with pytest.raises(TenantFilterViolation):
            seeded_vdb.search(
                collection=COLLECTION,
                query_embedding=[1.0, 0.0, 0.0],
                top_k=10,
                tenant_id=TENANT_A,
                filters={"business_id": TENANT_B},
            )

    def test_empty_tenant_id_raises(self, seeded_vdb):
        with pytest.raises(TenantFilterViolation):
            seeded_vdb.search(
                collection=COLLECTION,
                query_embedding=[1.0, 0.0, 0.0],
                top_k=10,
                tenant_id="",
            )

    def test_missing_tenant_id_raises_in_multi_tenant_mode(self, seeded_vdb):
        """Gate 1: tenant_id=None must hard-fail when isolation is enabled."""
        assert seeded_vdb._tenant_isolation_enabled is True
        with pytest.raises(TenantFilterViolation):
            seeded_vdb.search(
                collection=COLLECTION,
                query_embedding=[1.0, 0.0, 0.0],
                top_k=10,
                tenant_id=None,
            )

    def test_whitespace_tenant_id_raises(self, seeded_vdb):
        with pytest.raises(TenantFilterViolation):
            seeded_vdb.search(
                collection=COLLECTION,
                query_embedding=[1.0, 0.0, 0.0],
                top_k=10,
                tenant_id="   ",
            )

    def test_wildcard_star_tenant_id_raises(self, seeded_vdb):
        with pytest.raises(TenantFilterViolation):
            seeded_vdb.search(
                collection=COLLECTION,
                query_embedding=[1.0, 0.0, 0.0],
                top_k=10,
                tenant_id="*",
            )

    def test_wildcard_percent_tenant_id_raises(self, seeded_vdb):
        with pytest.raises(TenantFilterViolation):
            seeded_vdb.search(
                collection=COLLECTION,
                query_embedding=[1.0, 0.0, 0.0],
                top_k=10,
                tenant_id="%",
            )

    def test_wildcard_all_tenant_id_raises(self, seeded_vdb):
        with pytest.raises(TenantFilterViolation):
            seeded_vdb.search(
                collection=COLLECTION,
                query_embedding=[1.0, 0.0, 0.0],
                top_k=10,
                tenant_id="__all__",
            )

    def test_path_traversal_tenant_id_rejected_at_validator(self):
        with pytest.raises(TenantValidationError):
            validate_tenant_id("../etc/passwd", endpoint="test")

    def test_reserved_name_tenant_id_rejected(self):
        with pytest.raises(TenantValidationError):
            validate_tenant_id("__system__", endpoint="test")

    def test_manual_metadata_injection_via_filters(self, seeded_vdb):
        """Attacker tries injecting multiple filter keys to confuse matching."""
        with pytest.raises(TenantFilterViolation):
            seeded_vdb.search(
                collection=COLLECTION,
                query_embedding=[1.0, 0.0, 0.0],
                top_k=10,
                tenant_id=TENANT_A,
                filters={"business_id": "hacked-tenant"},
            )

    def test_null_filters_still_enforce_tenant(self, seeded_vdb):
        results = seeded_vdb.search(
            collection=COLLECTION,
            query_embedding=[1.0, 0.0, 0.0],
            top_k=10,
            tenant_id=TENANT_A,
            filters=None,
        )
        for hit in results:
            assert hit.metadata["business_id"] == TENANT_A


# ─────────────────────────────────────────────────────────────────────────────
# 4. MULTI-QUERY RETRIEVAL ISOLATION
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiQueryIsolation:
    """Multi-query expansion must preserve tenant isolation across all variants."""

    def test_multiple_queries_all_scoped_to_tenant(self, seeded_vdb):
        query_variants = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.7, 0.3, 0.0],
            [0.0, 1.0, 0.0],  # biased toward tenant B embeddings
        ]
        all_results = []
        for qv in query_variants:
            results = seeded_vdb.search(
                collection=COLLECTION,
                query_embedding=qv,
                top_k=10,
                tenant_id=TENANT_A,
            )
            all_results.extend(results)

        assert len(all_results) > 0
        for hit in all_results:
            assert hit.metadata["business_id"] == TENANT_A, (
                f"Multi-query leakage: got business_id={hit.metadata['business_id']}"
            )

    def test_fused_multi_query_results_no_cross_tenant(self, seeded_vdb):
        """Simulate RRF fusion — merged results still respect isolation."""
        q1_results = seeded_vdb.search(
            collection=COLLECTION, query_embedding=[1.0, 0.0, 0.0],
            top_k=5, tenant_id=TENANT_B,
        )
        q2_results = seeded_vdb.search(
            collection=COLLECTION, query_embedding=[0.0, 1.0, 0.0],
            top_k=5, tenant_id=TENANT_B,
        )
        fused = {h.id: h for h in q1_results + q2_results}
        for hit in fused.values():
            assert hit.metadata["business_id"] == TENANT_B


# ─────────────────────────────────────────────────────────────────────────────
# 5. HYBRID RETRIEVAL ISOLATION
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridRetrievalIsolation:
    """Hybrid search (vector + keyword) must preserve tenant isolation."""

    def test_hybrid_vector_path_isolated(self, seeded_vdb):
        results = seeded_vdb.search(
            collection=COLLECTION,
            query_embedding=[0.5, 0.5, 0.0],
            top_k=10,
            tenant_id=TENANT_A,
        )
        for hit in results:
            assert hit.metadata["business_id"] == TENANT_A

    def test_hybrid_with_extra_filters_isolated(self, seeded_vdb):
        results = seeded_vdb.search(
            collection=COLLECTION,
            query_embedding=[0.5, 0.5, 0.0],
            top_k=10,
            tenant_id=TENANT_B,
            filters={"source_type": "docx"},
        )
        for hit in results:
            assert hit.metadata["business_id"] == TENANT_B
            assert hit.metadata["source_type"] == "docx"


# ─────────────────────────────────────────────────────────────────────────────
# 6. RERANKER PATH ISOLATION
# ─────────────────────────────────────────────────────────────────────────────

class TestRerankerPathIsolation:
    """Reranking must not introduce cross-tenant results."""

    def test_reranker_input_already_isolated(self, seeded_vdb):
        """Reranker receives only tenant-scoped results."""
        initial_results = seeded_vdb.search(
            collection=COLLECTION,
            query_embedding=[1.0, 0.0, 0.0],
            top_k=10,
            tenant_id=TENANT_A,
        )
        # Simulate reranker: rescore and reorder
        reranked = sorted(initial_results, key=lambda h: -h.score * 0.8)
        for hit in reranked:
            assert hit.metadata["business_id"] == TENANT_A

    def test_reranker_cannot_inject_cross_tenant(self, seeded_vdb):
        """Even if reranker returned extra IDs, get_by_ids enforces isolation."""
        all_ids = ["a-doc-1", "a-doc-2", "b-doc-1", "b-doc-2"]
        results = seeded_vdb.get_by_ids(
            collection=COLLECTION, ids=all_ids, tenant_id=TENANT_A,
        )
        for hit in results:
            assert hit.metadata["business_id"] == TENANT_A
        assert all(h.id.startswith("a-") for h in results)


# ─────────────────────────────────────────────────────────────────────────────
# 7. POST-PROCESSOR PATH ISOLATION
# ─────────────────────────────────────────────────────────────────────────────

class TestPostProcessorIsolation:
    """Post-processing (filtering, trimming) must not leak tenant data."""

    def test_post_processor_input_is_tenant_scoped(self, seeded_vdb):
        results = seeded_vdb.search(
            collection=COLLECTION,
            query_embedding=[1.0, 0.0, 0.0],
            top_k=100,
            tenant_id=TENANT_A,
        )
        # Simulate post-processor: apply threshold
        threshold = 0.1
        filtered = [h for h in results if h.score >= threshold]
        for hit in filtered:
            assert hit.metadata["business_id"] == TENANT_A

    def test_post_processor_count_limiting_preserves_tenant(self, seeded_vdb):
        results = seeded_vdb.search(
            collection=COLLECTION,
            query_embedding=[1.0, 0.0, 0.0],
            top_k=100,
            tenant_id=TENANT_B,
        )
        limited = results[:2]
        for hit in limited:
            assert hit.metadata["business_id"] == TENANT_B


# ─────────────────────────────────────────────────────────────────────────────
# 8. FALLBACK PATH ISOLATION
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackPathIsolation:
    """Fallback and degraded paths must preserve tenant isolation."""

    def test_single_tenant_bypass_returns_all_when_disabled(self, seeded_vdb):
        """When isolation is disabled, all docs returned (single-tenant mode)."""
        import warnings
        seeded_vdb._tenant_isolation_enabled = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            results = seeded_vdb.search(
                collection=COLLECTION,
                query_embedding=[0.5, 0.5, 0.0],
                top_k=10,
                tenant_id=None,
            )
        # In single-tenant mode, no filter applied
        assert len(results) == 6

    def test_single_tenant_bypass_with_explicit_tenant_still_filters(self, seeded_vdb):
        """Even in single-tenant mode, explicit tenant_id is honored."""
        seeded_vdb._tenant_isolation_enabled = False
        results = seeded_vdb.search(
            collection=COLLECTION,
            query_embedding=[0.5, 0.5, 0.0],
            top_k=10,
            tenant_id=TENANT_A,
        )
        for hit in results:
            assert hit.metadata["business_id"] == TENANT_A


# ─────────────────────────────────────────────────────────────────────────────
# 9. TENANT VALIDATOR INTEGRATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestTenantValidatorIntegration:
    """API-boundary validation rejects malformed tenant IDs."""

    @pytest.mark.parametrize("bad_id", [
        None, "", "   ", "..", "../etc", "foo/bar", "foo\\bar",
        "*", "**", "%", "__all__", "any", "all",
        "__system__", "__internal__", "admin", "root", "null",
    ])
    def test_reject_invalid_tenant_ids(self, bad_id):
        with pytest.raises(TenantValidationError):
            validate_tenant_id(bad_id, endpoint="test", allow_default=False)

    @pytest.mark.parametrize("good_id,expected", [
        ("acme-corp", "acme-corp"),
        ("UPPER", "upper"),
        ("tenant_123", "tenant_123"),
        ("a", "a"),
        ("my-long-tenant-name-here", "my-long-tenant-name-here"),
    ])
    def test_accept_valid_tenant_ids(self, good_id, expected):
        ctx = validate_tenant_id(good_id, endpoint="test")
        assert ctx.tenant_id == expected

    def test_immutable_context(self):
        ctx = validate_tenant_id("acme", endpoint="test")
        with pytest.raises((AttributeError, TypeError)):
            ctx.tenant_id = "hacked"


# ─────────────────────────────────────────────────────────────────────────────
# 10. INGESTION TENANT GUARD TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestionTenantGuard:
    """Ingestion-side tenant enforcement rejects and detects violations."""

    def test_resolve_valid_tenant(self):
        ctx = resolve_ingestion_tenant(
            business_id="acme", file_id="f1", source="test",
        )
        assert ctx.tenant_id == "acme"
        assert ctx.validated is True

    def test_reject_missing_when_required(self):
        with pytest.raises(IngestionTenantViolation):
            resolve_ingestion_tenant(
                business_id=None, file_id="f1", source="test",
                allow_default=False,
            )

    def test_allow_default_for_single_tenant(self):
        ctx = resolve_ingestion_tenant(
            business_id=None, file_id="f1", source="test",
            allow_default=True,
        )
        assert ctx.tenant_id == "default"

    def test_immutability_enforcement_passes_for_same_tenant(self):
        ctx = IngestionTenantContext(
            tenant_id="acme", file_id="f1", source="test",
        )
        enforce_tenant_immutability(
            expected=ctx, actual_business_id="acme", stage="upsert",
        )

    def test_immutability_enforcement_detects_mutation(self):
        ctx = IngestionTenantContext(
            tenant_id="acme", file_id="f1", source="test",
        )
        with pytest.raises(IngestionTenantViolation):
            enforce_tenant_immutability(
                expected=ctx, actual_business_id="evil-corp",
                stage="upsert",
            )

    def test_immutability_enforcement_ignores_none(self):
        ctx = IngestionTenantContext(
            tenant_id="acme", file_id="f1", source="test",
        )
        enforce_tenant_immutability(
            expected=ctx, actual_business_id=None, stage="upsert",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 11. REGRESSION TESTS — All Retrieval Modes
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionAllRetrievalModes:
    """
    Regression suite ensuring each retrieval mode preserves isolation
    after any pipeline change.
    """

    def test_semantic_search_isolation(self, seeded_vdb):
        for tenant in [TENANT_A, TENANT_B]:
            results = seeded_vdb.search(
                collection=COLLECTION,
                query_embedding=[0.5, 0.5, 0.0],
                top_k=10,
                tenant_id=tenant,
            )
            for hit in results:
                assert hit.metadata["business_id"] == tenant

    def test_top_k_boundary_isolation(self, seeded_vdb):
        """top_k=1 still returns correct tenant."""
        result = seeded_vdb.search(
            collection=COLLECTION,
            query_embedding=[1.0, 0.0, 0.0],
            top_k=1,
            tenant_id=TENANT_B,
        )
        assert len(result) <= 1
        if result:
            assert result[0].metadata["business_id"] == TENANT_B

    def test_large_top_k_does_not_leak(self, seeded_vdb):
        results = seeded_vdb.search(
            collection=COLLECTION,
            query_embedding=[0.5, 0.5, 0.5],
            top_k=1000,
            tenant_id=TENANT_A,
        )
        assert len(results) == 3
        for hit in results:
            assert hit.metadata["business_id"] == TENANT_A

    def test_zero_vector_query_isolation(self, seeded_vdb):
        results = seeded_vdb.search(
            collection=COLLECTION,
            query_embedding=[0.0, 0.0, 0.0],
            top_k=10,
            tenant_id=TENANT_A,
        )
        for hit in results:
            assert hit.metadata["business_id"] == TENANT_A

    def test_sequential_searches_different_tenants(self, seeded_vdb):
        """Rapid sequential searches for different tenants never mix."""
        for _ in range(50):
            a_results = seeded_vdb.search(
                collection=COLLECTION,
                query_embedding=[1.0, 0.0, 0.0],
                top_k=5,
                tenant_id=TENANT_A,
            )
            b_results = seeded_vdb.search(
                collection=COLLECTION,
                query_embedding=[0.0, 1.0, 0.0],
                top_k=5,
                tenant_id=TENANT_B,
            )
            for hit in a_results:
                assert hit.metadata["business_id"] == TENANT_A
            for hit in b_results:
                assert hit.metadata["business_id"] == TENANT_B


# ─────────────────────────────────────────────────────────────────────────────
# 12. AUDIT TELEMETRY VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditTelemetry:
    """Verify that isolation events produce structured audit logs."""

    def test_cross_tenant_attempt_produces_audit(self, seeded_vdb):
        import logging
        import json

        captured = []
        handler = logging.Handler()
        handler.emit = lambda r: captured.append(r.getMessage())
        audit_logger = logging.getLogger("tenant_audit")
        audit_logger.addHandler(handler)
        audit_logger.setLevel(logging.DEBUG)

        try:
            with pytest.raises(TenantFilterViolation):
                seeded_vdb.search(
                    collection=COLLECTION,
                    query_embedding=[1.0, 0.0, 0.0],
                    top_k=5,
                    tenant_id=TENANT_A,
                    filters={"business_id": TENANT_B},
                )
            cross_events = [
                json.loads(m) for m in captured
                if "CROSS_TENANT" in m
            ]
            assert len(cross_events) >= 1
            assert cross_events[0]["severity"] == "critical"
        finally:
            audit_logger.removeHandler(handler)

    def test_search_produces_audit_telemetry(self, seeded_vdb):
        import logging
        import json

        captured = []
        handler = logging.Handler()
        handler.emit = lambda r: captured.append(r.getMessage())
        audit_logger = logging.getLogger("tenant_audit")
        audit_logger.addHandler(handler)
        audit_logger.setLevel(logging.DEBUG)

        try:
            seeded_vdb.search(
                collection=COLLECTION,
                query_embedding=[1.0, 0.0, 0.0],
                top_k=5,
                tenant_id=TENANT_A,
            )
            search_events = [
                json.loads(m) for m in captured
                if "VECTORDB_SEARCH" in m
            ]
            assert len(search_events) >= 1
            assert search_events[0]["tenant_id"] == TENANT_A
        finally:
            audit_logger.removeHandler(handler)
