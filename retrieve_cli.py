# retrieve_cli.py

import asyncio
import time
from app.config.ingestion_settings import EMBEDDING_MODEL_NAME
from sentence_transformers import SentenceTransformer

from app.retrieval.runtime import RetrievalRuntime
from app.retrieval.policy import DEFAULT_POLICY_REGISTRY
from app.retrieval.types_retrieve import QueryContext, RetrievalIntent
from app.retrieval.repository import RetrievalRepository
from app.db.session_v2 import get_async_session
from app.utils.logger import log_debug, log_info
from app.services.retrieval.chroma_search import get_chroma_search



# =========================================================
# QUERY EMBEDDING (MATCHES INGESTION EXACTLY)
# =========================================================

_EMBEDDER = None
#EMBED_MODEL_NAME = "BAAI/bge-large-en"


def get_query_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        print(f"🔧 Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("✅ Model loaded successfully")
    return _EMBEDDER


def embed_query(query: str) -> list[float]:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    embedder = get_query_embedder()
    embedding = embedder.encode(
        [query],
        normalize_embeddings=True
    )

    try:
        return embedding[0].tolist()
    except Exception:
        return list(embedding[0])


# =========================================================
# CLI RUNTIME
# =========================================================

async def run_cli():

    print("\n🧠 Enterprise Retrieval CLI")
    print("Type 'exit' to quit\n")

    async with get_async_session() as db:

        # ✅ CORRECT: Repository handles ChromaDB internally
        repository = RetrievalRepository(db_session=db)

        runtime = RetrievalRuntime(
            repository=repository,
            policy_registry=DEFAULT_POLICY_REGISTRY,
        )

        while True:
            query = input("🔎 Ask a question: ").strip()

            if query.lower() in {"exit", "quit"}:
                print("👋 Exiting.")
                break

            ctx = QueryContext(
                query=query,
                intent=RetrievalIntent.ANSWER,
                requested_at=int(time.time()),
            )

            print("\n⏳ Retrieving...\n")

            # -------------------------------------------------
            # 1. Embed query
            # -------------------------------------------------
            try:
                query_embedding = embed_query(query)
                log_info("Query embedding generated")
                log_debug(f"Embedding length: {len(query_embedding)}")
            except Exception as e:
                print(f"❌ Failed to embed query: {e}\n")
                continue

            # -------------------------------------------------
            # 2. Retrieve
            # -------------------------------------------------
            try:
                ranked_results, dropped = await runtime.retrieve(
                    ctx=ctx,
                    query_embedding=query_embedding,
                )
            except Exception as e:
                print(f"❌ Retrieval failed: {e}\n")
                import traceback
                traceback.print_exc()
                continue

            if not ranked_results:
                print("⚠️ No trusted answer found.\n")
                continue

            # -------------------------------------------------
            # 3. Render results
            # -------------------------------------------------
            print("\n" + "="*80)
            print(f"📊 RETRIEVAL RESULTS ({len(ranked_results)} found)")
            print("="*80 + "\n")

            for idx, result in enumerate(ranked_results, start=1):
                print(f"\n✅ RESULT #{idx}")
                print("-" * 80)
                print(result.text[:500] + ("..." if len(result.text) > 500 else ""))
                print("-" * 80)
                print(f"📊 Confidence: {round(result.score, 4)}")

                if result.explanation:
                    print("\n🔍 Why this result:")
                    signals = result.explanation.get("signals", {})
                    for k, v in signals.items():
                        print(f"  • {k}: {v}")

            print("\n" + "="*80 + "\n")


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    asyncio.run(run_cli())
