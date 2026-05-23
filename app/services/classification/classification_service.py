# services/classification/classification_service.py

import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert

from app.services.classification.embedding_ranker import rank_taxonomy_candidates
from app.services.classification.llm_classifier import classify_chunk_with_llm
from app.services.classification.canonicalizer import canonicalize_llm_output
from app.services.classification.taxonomy_loader import load_taxonomy

from app.db.models.business_classification import BusinessClassification
from app.db.models.classification_logs import ClassificationLogs

from app.core.vectordb.base import BaseVectorDB
from app.core.config.client_config_schema import VectorDBType
from app.utils.logger import log_info, log_warning
from app.utils.tenant_storage_uuid import storage_uuid_str_for_vectordb_metadata


class ClassificationService:

    @staticmethod
    async def classify_chunk(
        db: AsyncSession,
        chunk: Dict[str, Any],
        client_id: str,
        vectordb: Optional[BaseVectorDB] = None,
    ) -> Dict[str, Any]:
        """
        Full classification pipeline for a single chunk.

        Requires client_id (tenant slug) — no env-based VectorDB fallback.
        """
        if not client_id or not str(client_id).strip():
            raise ValueError("client_id is required for classify_chunk()")

        slug = str(client_id).strip()
        chunk_id = str(chunk["id"])
        text = chunk.get("cleaned_text") or ""

        log_info(f"[classification_service] Classifying chunk: {chunk_id} tenant={slug}")

        await load_taxonomy(db)

        ranked = rank_taxonomy_candidates(text)
        llm_output = classify_chunk_with_llm(text, ranked)

        canonical = await canonicalize_llm_output(
            db=db,
            chunk_id=chunk_id,
            llm_output=llm_output,
        )

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

        from app.core.config.client_config_resolver import get_client_config
        from app.services.ingestion.ingestion_service_v2 import (
            _get_ingestion_pipeline_for_client,
        )

        cfg = get_client_config(slug)
        coll_name = (cfg.vectordb.collection or "").strip()
        if not coll_name:
            raise ValueError(
                f"vectordb.collection is required for client_id={slug!r}"
            )

        vdb = vectordb
        if vdb is None:
            vdb = _get_ingestion_pipeline_for_client(slug).vectordb

        tenant_meta_key = storage_uuid_str_for_vectordb_metadata(slug)
        base_meta = {
            "business_id": tenant_meta_key,
            "industry_id": canonical["industry_id"],
            "sub_industry_id": canonical["sub_industry_id"],
            "sub_sub_industry_id": canonical["sub_sub_industry_id"],
            "pending_taxonomy_id": canonical["pending_taxonomy_id"],
            "confidence": llm_output.get("confidence"),
        }

        try:
            vdb.update_metadata(
                collection=coll_name,
                ids=[chunk_id],
                metadatas=[base_meta],
            )
        except Exception as e:
            log_warning(
                f"[classification_service] VectorDB metadata update failed "
                f"(type={cfg.vectordb.type.value}): {e}"
            )
            if cfg.vectordb.type != VectorDBType.CHROMA:
                raise ValueError(
                    f"classification metadata update requires a vectordb supporting "
                    f"update_metadata; got {cfg.vectordb.type.value}"
                ) from e

        return {
            "classification_id": str(classification_id),
            "industry_id": canonical["industry_id"],
            "sub_industry_id": canonical["sub_industry_id"],
            "sub_sub_industry_id": canonical["sub_sub_industry_id"],
            "pending_taxonomy_id": canonical["pending_taxonomy_id"],
            "confidence": llm_output.get("confidence"),
        }
