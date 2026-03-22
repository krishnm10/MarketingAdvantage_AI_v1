# services/classification/classification_service.py

import uuid
from typing import Dict, Any
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert

from app.services.classification.embedding_ranker import rank_taxonomy_candidates
from app.services.classification.llm_classifier import classify_chunk_with_llm
from app.services.classification.canonicalizer import canonicalize_llm_output
from app.services.classification.taxonomy_loader import load_taxonomy

from app.db.models.business_classification import BusinessClassification
from app.db.models.classification_logs import ClassificationLogs

from app.utils.logger import log_info, log_warning
import os
import chromadb
from chromadb.config import Settings


# Chroma client (update metadata for chunks) — remote or local
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
CONTENT_COLLECTION = CHROMA.get_or_create_collection(
    name=os.getenv("MAI_COLLECTION", "ingested_content"),
    metadata={"hnsw:space": "cosine"}
)


class ClassificationService:

    @staticmethod
    async def classify_chunk(
        db: AsyncSession,
        chunk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Full classification pipeline for a single chunk.

        Input chunk:
            {
                "id": UUID,
                "cleaned_text": "...",
                ...
            }

        Return:
            {
                "classification_id": UUID,
                "industry_id": ...,
                "pending_taxonomy_id": ...
            }
        """

        chunk_id = str(chunk["id"])
        text = chunk.get("cleaned_text") or ""

        log_info(f"[classification_service] Classifying chunk: {chunk_id}")

        # -----------------------------------------------------
        # 1. Ensure taxonomy is loaded
        # -----------------------------------------------------
        await load_taxonomy(db)

        # -----------------------------------------------------
        # 2. Ranking via Embedding Matcher
        # -----------------------------------------------------
        ranked = rank_taxonomy_candidates(text)

        # -----------------------------------------------------
        # 3. LLM classification
        # -----------------------------------------------------
        llm_output = classify_chunk_with_llm(text, ranked)

        # -----------------------------------------------------
        # 4. Canonicalization & Pending Taxonomy Logic
        # -----------------------------------------------------
        canonical = await canonicalize_llm_output(
            db=db,
            chunk_id=chunk_id,
            llm_output=llm_output
        )

        # -----------------------------------------------------
        # 5. Insert into business_classification
        # -----------------------------------------------------
        classification_id = uuid.uuid4()

        row = {
            "id": classification_id,
            "content_id": chunk_id,
            "industry_id": canonical["industry_id"],
            "sub_industry_id": canonical["sub_industry_id"],
            "sub_sub_industry_id": canonical["sub_sub_industry_id"],
            "pending_taxonomy_id": canonical["pending_taxonomy_id"],
            "confidence": llm_output.get("confidence", 0.0),
            "llm_model": "llama-3.1-8b",
            "raw_output": llm_output.get("llm_raw"),
            "created_at": datetime.utcnow(),
        }

        await db.execute(insert(BusinessClassification).values(**row))
        await db.commit()

        # -----------------------------------------------------
        # 6. Insert into classification_logs
        # -----------------------------------------------------
        taxonomy_path = (
            f"{llm_output.get('industry') or ''} > "
            f"{llm_output.get('sub_industry') or ''} > "
            f"{llm_output.get('sub_sub_industry') or ''}"
        ).strip(" > ")

        log_row = {
            "id": uuid.uuid4(),
            "content_id": chunk_id,
            "taxonomy_path": taxonomy_path,
            "confidence": llm_output.get("confidence"),
            "embed_scores": ranked,
            "llm_scores": llm_output,
            "created_at": datetime.utcnow(),
        }

        await db.execute(insert(ClassificationLogs).values(**log_row))
        await db.commit()

        # -----------------------------------------------------
        # 7. Update Chroma metadata
        # -----------------------------------------------------
        try:
            CONTENT_COLLECTION.update(
                ids=[chunk_id],
                metadatas=[
                    {
                        "industry_id": canonical["industry_id"],
                        "sub_industry_id": canonical["sub_industry_id"],
                        "sub_sub_industry_id": canonical["sub_sub_industry_id"],
                        "pending_taxonomy_id": canonical["pending_taxonomy_id"],
                        "confidence": llm_output.get("confidence"),
                    }
                ]
            )
        except Exception as e:
            log_warning(f"[classification_service] Chroma update failed: {e}")

        # -----------------------------------------------------
        # 8. Return result
        # -----------------------------------------------------
        return {
            "classification_id": str(classification_id),
            "industry_id": canonical["industry_id"],
            "sub_industry_id": canonical["sub_industry_id"],
            "sub_sub_industry_id": canonical["sub_sub_industry_id"],
            "pending_taxonomy_id": canonical["pending_taxonomy_id"],
            "confidence": llm_output.get("confidence"),
        }
