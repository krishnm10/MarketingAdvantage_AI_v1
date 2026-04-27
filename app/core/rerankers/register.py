"""
================================================================================
Marketing Advantage AI — Reranker Plugin Registration (Lazy Factories)
File: app/core/rerankers/register.py

All reranker imports are lazy — heavy dependencies (torch, transformers,
sentence-transformers) are NOT loaded at import/registration time.

After import, usage from factory:
  reranker_registry.build("crossencoder", model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")
  reranker_registry.build("bge_reranker", model_name="BAAI/bge-reranker-v2-m3")
  reranker_registry.build("flashrank",    model_name="ms-marco-MiniLM-L-12-v2")
  reranker_registry.build("cohere",       api_key="...", model="rerank-multilingual-v3.0")
================================================================================
"""

from app.core.plugin_registry import reranker_registry


# ── Lazy factory functions ────────────────────────────────────────────────────


def _crossencoder_factory(**kwargs):
    from app.core.rerankers.crossencoder_v1 import CrossEncoderReranker
    return CrossEncoderReranker(**kwargs)


def _bge_reranker_factory(**kwargs):
    from app.core.rerankers.bge_reranker_v1 import BGEReranker
    return BGEReranker(**kwargs)


def _flashrank_factory(**kwargs):
    from app.core.rerankers.flashrank_v1 import FlashRankReranker
    return FlashRankReranker(**kwargs)


def _cohere_factory(**kwargs):
    from app.core.rerankers.cohere_v1 import CohereReranker
    return CohereReranker(**kwargs)


def _colbert_factory(**kwargs):
    from app.core.rerankers.colbert_v1 import ColBERTReranker
    return ColBERTReranker(**kwargs)


def _mmr_factory(**kwargs):
    from app.core.rerankers.mmr_reranker import MMRReranker
    return MMRReranker(**kwargs)


def _score_threshold_factory(**kwargs):
    from app.core.rerankers.score_threshold_reranker import ScoreThresholdReranker
    return ScoreThresholdReranker(**kwargs)


# ── Registrations ─────────────────────────────────────────────────────────────

reranker_registry.register(
    "crossencoder",
    _crossencoder_factory,
    description=(
        "HuggingFace CrossEncoder — best free local reranker. "
        "Recommended: cross-encoder/ms-marco-MiniLM-L-12-v2"
    ),
)

reranker_registry.register(
    "bge_reranker",
    _bge_reranker_factory,
    description=(
        "BAAI BGE Reranker — MTEB benchmark leader (free, local). "
        "Use bge-reranker-v2-m3 for Indian multilingual content."
    ),
)

reranker_registry.register(
    "flashrank",
    _flashrank_factory,
    description=(
        "FlashRank — ultra-fast ONNX reranker for real-time APIs. "
        "4x faster than CrossEncoder on CPU. Best for latency-sensitive clients."
    ),
)

reranker_registry.register(
    "cohere",
    _cohere_factory,
    description=(
        "Cohere Rerank API — highest quality cloud reranker. "
        "Use rerank-multilingual-v3.0 for Hindi/Telugu/Tamil content."
    ),
)

reranker_registry.register(
    "colbert",
    _colbert_factory,
    description=(
        "ColBERT — late interaction retrieval/reranking model. "
        "Best for high-recall scenarios."
    ),
)

reranker_registry.register(
    "mmr",
    _mmr_factory,
    description="MMR — Maximal Marginal Relevance for diversity-aware reranking.",
)

reranker_registry.register(
    "score_threshold",
    _score_threshold_factory,
    description="Score Threshold — filter candidates below a calibrated score cutoff.",
)
