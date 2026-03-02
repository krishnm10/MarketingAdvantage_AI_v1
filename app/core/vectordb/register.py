# ============================================================
# app/core/vectordb/register.py
#
# WHAT THIS FILE IS:
#   Registers ALL vector DB connectors into vectordb_registry.
#   This file is imported ONCE by pipeline_factory.py at startup.
#
# KEY DESIGN PRINCIPLE — LAZY IMPORTS:
#   Each connector is imported ONLY when you actually BUILD it.
#   This means: if qdrant-client is not installed, the app still
#   starts fine as long as no one requests a Qdrant pipeline.
#   You can run with ONLY chromadb installed — zero crashes.
#
# HOW TO ADD A NEW VECTOR DB:
#   1. Create app/core/vectordb/mydb_v1.py implementing BaseVectorDB
#   2. Add a _mydb_factory() function below
#   3. Call _safe_register("mydb", _mydb_factory, description="...")
#   4. Add "mydb" to VectorDBType enum in client_config_schema.py
#   5. Add a _build_vectordb() case in pipeline_factory.py
#   That's it. Nothing else changes anywhere.
# ============================================================

import logging
from app.core.plugin_registry import vectordb_registry

logger = logging.getLogger(__name__)


def _safe_register(name: str, factory, description: str = "") -> None:
    """
    Register a connector safely.
    If the connector's library is not installed, log a warning
    instead of crashing the entire application at startup.

    Example: if qdrant-client is not installed, Qdrant registration
    simply logs a warning. Everything else still works.
    """
    try:
        vectordb_registry.register(name, factory, description=description)
        logger.debug("[vectordb.register] Registered '%s'", name)
    except Exception as exc:
        logger.warning(
            "[vectordb.register] Could not register '%s': %s", name, exc
        )


# ── ChromaDB ───────────────────────────────────────────────────────────
# Direct import is safe — chromadb is in your core requirements.txt
# so it will ALWAYS be installed.
from app.core.vectordb.chroma_v1 import ChromaVectorDB

_safe_register(
    "chroma",
    ChromaVectorDB,
    description=(
        "ChromaDB PersistentClient — local file-based vector store. "
        "Existing ingestion pipeline uses this. Do NOT remove."
    ),
)


# ── Qdrant ─────────────────────────────────────────────────────────────
# LAZY IMPORT: qdrant_v1 is only imported when a Qdrant pipeline
# is actually requested. App starts fine without qdrant-client installed.
def _qdrant_factory(**kw):
    from app.core.vectordb.qdrant_v1 import QdrantVectorDB
    return QdrantVectorDB(**kw)

_safe_register(
    "qdrant",
    _qdrant_factory,
    description=(
        "Qdrant — high-performance vector DB. "
        "Supports cloud (Qdrant Cloud) and local Docker."
    ),
)


# ── Weaviate ───────────────────────────────────────────────────────────
def _weaviate_factory(**kw):
    from app.core.vectordb.weaviate_v1 import WeaviateVectorDB
    return WeaviateVectorDB(**kw)

_safe_register(
    "weaviate",
    _weaviate_factory,
    description=(
        "Weaviate — multi-modal vector DB with built-in schema. "
        "Supports WCS cloud and local embedded mode."
    ),
)


# ── Pinecone ───────────────────────────────────────────────────────────
def _pinecone_factory(**kw):
    from app.core.vectordb.pinecone_v1 import PineconeVectorDB
    return PineconeVectorDB(**kw)

_safe_register(
    "pinecone",
    _pinecone_factory,
    description=(
        "Pinecone — fully managed serverless vector DB. "
        "Best for SaaS clients wanting zero infra."
    ),
)


# ── Milvus ─────────────────────────────────────────────────────────────
def _milvus_factory(**kw):
    from app.core.vectordb.milvus_v1 import MilvusVectorDB
    return MilvusVectorDB(**kw)

_safe_register(
    "milvus",
    _milvus_factory,
    description=(
        "Milvus — open-source billion-scale vector DB. "
        "Best for on-premise enterprise and data-residency requirements."
    ),
)
