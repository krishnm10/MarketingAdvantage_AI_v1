"""
================================================================================
Marketing Advantage AI — Reranker Plugin Registration
File: app/core/rerankers/register.py

Import once at app startup to register all reranker connectors.
NO default reranker is set — client config must specify reranker type.

Rerankers are optional per client. If a client does not need reranking
(cost-sensitive or low-latency requirement), simply omit the 'reranker'
block from their ClientConfig.

After import, usage from factory:
  reranker_registry.build("crossencoder", model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")
  reranker_registry.build("bge_reranker", model_name="BAAI/bge-reranker-v2-m3")
  reranker_registry.build("flashrank",    model_name="ms-marco-MiniLM-L-12-v2")
  reranker_registry.build("cohere",       api_key="...", model="rerank-multilingual-v3.0")
================================================================================
"""

from app.core.plugin_registry import reranker_registry

from app.core.rerankers.crossencoder_v1 import CrossEncoderReranker
from app.core.rerankers.bge_reranker_v1 import BGEReranker
from app.core.rerankers.flashrank_v1    import FlashRankReranker
from app.core.rerankers.cohere_v1       import CohereReranker
from app.core.rerankers.colbert_v1 import ColBERTReranker


reranker_registry.register(
    "crossencoder",
    CrossEncoderReranker,
    description=(
        "HuggingFace CrossEncoder — best free local reranker. "
        "Recommended: cross-encoder/ms-marco-MiniLM-L-12-v2"
    ),
)

reranker_registry.register(
    "bge_reranker",
    BGEReranker,
    description=(
        "BAAI BGE Reranker — MTEB benchmark leader (free, local). "
        "Use bge-reranker-v2-m3 for Indian multilingual content."
    ),
)

reranker_registry.register(
    "flashrank",
    FlashRankReranker,
    description=(
        "FlashRank — ultra-fast ONNX reranker for real-time APIs. "
        "4x faster than CrossEncoder on CPU. Best for latency-sensitive clients."
    ),
)

reranker_registry.register(
    "cohere",
    CohereReranker,
    description=(
        "Cohere Rerank API — highest quality cloud reranker. "
        "Use rerank-multilingual-v3.0 for Hindi/Telugu/Tamil content."
    ),
)
