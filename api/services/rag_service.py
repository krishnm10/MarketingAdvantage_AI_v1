# ==========================================
# 📚 rag_service.py — Persistent Local Vector RAG Layer
# ==========================================
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from typing import List

# ✅ Persistent ChromaDB directory
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/rag_db")
os.makedirs(CHROMA_PATH, exist_ok=True)

# ✅ Create persistent Chroma client
_client = chromadb.PersistentClient(path=CHROMA_PATH)

# ✅ Create or load collection
_collection = _client.get_or_create_collection(
    name="content_knowledge_base",
    metadata={"description": "Stores verified brand/blog/social content for creative RAG retrieval"}
)

# ✅ Free Embedding model
_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================
# 🧠 Ingest New Content
# ==========================================
def ingest_content(doc_id: str, text: str, metadata: dict = None):
    """Embed and store new content into the local RAG knowledge base."""
    if not text or not doc_id:
        return

    emb = _encoder.encode(text).tolist()
    _collection.add(
        ids=[doc_id],
        embeddings=[emb],
        documents=[text],
        metadatas=[metadata or {}]
    )
    print(f"[RAG] ✅ Ingested: {doc_id} ({len(text)} chars)")

# ==========================================
# 🔍 Retrieve Relevant Chunks
# ==========================================
def retrieve_relevant_chunks(query: str, n_results: int = 3, include_meta: bool = False) -> List[str]:
    """Retrieve semantically relevant text snippets."""
    if not query:
        return []

    query_emb = _encoder.encode(query).tolist()
    results = _collection.query(query_embeddings=[query_emb], n_results=n_results)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    print(f"[RAG] 🔎 Retrieved {len(docs)} chunks for query: '{query}'")

    if include_meta:
        return [{"text": d, "metadata": m} for d, m in zip(docs, metas)]
    return docs

# ==========================================
# 🧹 Clear Vector Store
# ==========================================
def clear_vector_store():
    """Delete all embeddings from the vector store."""
    _client.delete_collection("content_knowledge_base")
    print("[RAG] 🧹 Knowledge base cleared.")
