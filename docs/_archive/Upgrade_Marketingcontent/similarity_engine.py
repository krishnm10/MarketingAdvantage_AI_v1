# similarity_engine.py

from sqlalchemy import text
from chromadb.api import Client


def find_similar_taxonomy(db, canonical_name, top_n=5):
    """
    Combines trigram similarity + vector embedding similarity.
    """

    # ----------- Trigram similarity (Postgres) ----------------
    trigram_query = text("""
        SELECT id, canonical_name,
               similarity(canonical_name, :cname) AS score
        FROM taxonomy
        ORDER BY canonical_name <-> :cname
        LIMIT :limit
    """)
    trigram_matches = db.execute(trigram_query, {"cname": canonical_name, "limit": top_n}).fetchall()

    # ----------- Embedding-based similarity ----------------
    chroma = Client()
    collection = chroma.get_collection("taxonomy_embeddings")

    vector_results = collection.query(
        query_texts=[canonical_name],
        n_results=top_n
    )

    # Merge scores
    results = {}

    for row in trigram_matches:
        results[row.id] = {"id": row.id, "score": float(row.score)}

    for idx, tid in enumerate(vector_results["ids"][0]):
        if tid not in results:
            results[tid] = {"id": tid, "score": float(vector_results["distances"][0][idx])}

    # Sort by score
    return sorted(results.values(), key=lambda x: x["score"], reverse=True)
