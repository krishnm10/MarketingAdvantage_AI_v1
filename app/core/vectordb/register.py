"""
================================================================================
Marketing Advantage AI — VectorDB Plugin Auto-Registration
File: app/core/vectordb/register.py

Import this module ONCE at app startup (in app/main.py lifespan or similar).
After this import, any code can call:

    vectordb_registry.build("qdrant", url=..., api_key=..., embedding_dim=384)
    vectordb_registry.build("chroma", persist_directory="./chroma_db")
    vectordb_registry.build("pinecone", api_key=..., index_name=..., embedding_dim=384)
    vectordb_registry.build("weaviate", url=..., api_key=...)
    vectordb_registry.build("milvus", host="localhost", port=19530)

NO default is set here intentionally.
Clients must always specify their VectorDB type in their config.
================================================================================
"""

from app.core.plugin_registry import vectordb_registry

from app.core.vectordb.chroma_v1  import ChromaVectorDB
from app.core.vectordb.qdrant_v1  import QdrantVectorDB
from app.core.vectordb.weaviate_v1 import WeaviateVectorDB
from app.core.vectordb.pinecone_v1 import PineconeVectorDB
from app.core.vectordb.milvus_v1  import MilvusVectorDB

vectordb_registry.register(
    "chroma",
    ChromaVectorDB,
    description=(
        "ChromaDB PersistentClient — local file-based vector store. "
        "Existing ingestion pipeline uses this. Do NOT remove."
    ),
)

vectordb_registry.register(
    "qdrant",
    QdrantVectorDB,
    description=(
        "Qdrant — high-performance vector DB. "
        "Supports cloud (Qdrant Cloud) and local Docker."
    ),
)

vectordb_registry.register(
    "weaviate",
    WeaviateVectorDB,
    description=(
        "Weaviate — multi-modal vector DB with built-in schema. "
        "Supports WCS cloud and local embedded mode."
    ),
)

vectordb_registry.register(
    "pinecone",
    PineconeVectorDB,
    description=(
        "Pinecone — fully managed serverless vector DB. "
        "Best for SaaS clients wanting zero infra."
    ),
)

vectordb_registry.register(
    "milvus",
    MilvusVectorDB,
    description=(
        "Milvus — open-source billion-scale vector DB. "
        "Best for on-premise enterprise and data-residency requirements."
    ),
)
