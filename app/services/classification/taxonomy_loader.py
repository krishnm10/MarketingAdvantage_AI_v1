# services/classification/taxonomy_loader.py

import uuid
from typing import Dict, Any, List, Optional
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from sentence_transformers import SentenceTransformer
import os
import chromadb
from chromadb.config import Settings

from app.db.models.taxonomy import Taxonomy
from app.db.models.taxonomy_alias import TaxonomyAlias
from app.utils.logger import log_info, log_warning


# Global embedder for taxonomy entries
EMBEDDER = SentenceTransformer("BAAI/bge-large-en")

# ChromaDB client — remote or local
def _make_chroma_client():
    host = os.getenv("CHROMA_HOST") or None
    if host:
        port = int(os.getenv("CHROMA_PORT") or "8000")
        ssl = os.getenv("CHROMA_SSL", "").lower() in ("1", "true", "yes")
        api_key = os.getenv("CHROMA_API_KEY") or None
        return chromadb.HttpClient(
            host=host, port=port, ssl=ssl,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
            settings=Settings(anonymized_telemetry=False),
        )
    return chromadb.PersistentClient(
        path=os.getenv("CHROMA_PATH", "./pluggable_db"),
        settings=Settings(anonymized_telemetry=False),
    )

CHROMA = _make_chroma_client()
TAXONOMY_COLLECTION = CHROMA.get_or_create_collection(
    name="taxonomy_collection",
    metadata={"hnsw:space": "cosine"}
)


class TaxonomyNode:
    """In-memory DTO for taxonomy tree traversal."""
    def __init__(self, id, name, parent_id, level):
        self.id = id
        self.name = name
        self.parent_id = parent_id
        self.level = level
        self.children = []


class TaxonomyCache:
    taxonomy_nodes: Dict[str, TaxonomyNode] = {}
    aliases: Dict[str, str] = {}  # alias -> canonical
    loaded: bool = False


async def load_taxonomy(db: AsyncSession):
    """
    Loads the entire taxonomy table + alias table + embeddings.
    Builds parent-child relationships.
    """

    if TaxonomyCache.loaded:
        return

    log_info("[taxonomy_loader] Loading taxonomy...")

    # ------------------------------
    # Load taxonomy items
    # ------------------------------
    result = await db.execute(select(Taxonomy))
    rows = result.scalars().all()

    for row in rows:
        node = TaxonomyNode(
            id=str(row.id),
            name=row.name,
            parent_id=str(row.parent_id) if row.parent_id else None,
            level=row.level
        )
        TaxonomyCache.taxonomy_nodes[node.id] = node

    # ------------------------------
    # Build tree relationships
    # ------------------------------
    for node in TaxonomyCache.taxonomy_nodes.values():
        if node.parent_id and node.parent_id in TaxonomyCache.taxonomy_nodes:
            parent = TaxonomyCache.taxonomy_nodes[node.parent_id]
            parent.children.append(node)

    # ------------------------------
    # Load aliases
    # ------------------------------
    alias_rows = (await db.execute(select(TaxonomyAlias))).scalars().all()

    for a in alias_rows:
        TaxonomyCache.aliases[a.alias.lower()] = str(a.taxonomy_id)

    # ------------------------------
    # Load embeddings from DB or compute & save in Chroma
    # ------------------------------
    existing = TAXONOMY_COLLECTION.get(include=["embeddings", "metadatas"])

    existing_ids = set(existing["ids"]) if existing["ids"] else set()

    missing = [node for node in TaxonomyCache.taxonomy_nodes.values()
               if node.id not in existing_ids]

    if missing:
        log_info(f"[taxonomy_loader] Computing embeddings for {len(missing)} taxonomy items")

        texts = [m.name for m in missing]
        embeddings = EMBEDDER.encode(texts, normalize_embeddings=True)

        TAXONOMY_COLLECTION.upsert(
            ids=[m.id for m in missing],
            metadatas=[{"name": m.name, "level": m.level} for m in missing],
            documents=texts,
            embeddings=embeddings.tolist()
        )

    TaxonomyCache.loaded = True
    log_info("[taxonomy_loader] Taxonomy loaded successfully.")


def find_canonical(name: str) -> Optional[str]:
    """
    Attempts to map a given text to a canonical taxonomy ID.
    Uses alias table + exact match + lowercase fallbacks.
    """

    if not name:
        return None

    n = name.strip().lower()

    # Alias lookup
    if n in TaxonomyCache.aliases:
        return TaxonomyCache.aliases[n]

    # Direct name match
    for t_id, node in TaxonomyCache.taxonomy_nodes.items():
        if node.name.lower() == n:
            return t_id

    return None


def get_taxonomy_tree() -> Dict[str, TaxonomyNode]:
    """Returns all taxonomy nodes."""
    return TaxonomyCache.taxonomy_nodes


def get_taxonomy_node(taxonomy_id: str) -> Optional[TaxonomyNode]:
    return TaxonomyCache.taxonomy_nodes.get(taxonomy_id)
