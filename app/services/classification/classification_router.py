# services/classification/classification_router.py

import asyncio
from typing import Dict, Any, List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models.ingested_content import IngestedContent
from app.db.models.ingested_file_v2 import IngestedFileV2
from app.services.classification.classification_service import ClassificationService
from app.core.config.client_config_resolver import _resolve_client_id_for_config_lookup
from app.utils.logger import log_info, log_warning


class ClassificationRouter:

    @staticmethod
    async def classify_file(
        db: AsyncSession,
        file_id: str,
        parallel: bool = True,
        max_concurrency: int = 4,
    ) -> Dict[str, Any]:
        """
        Classifies ALL chunks belonging to a given ingested_file.

        Resolves tenant slug from IngestedFileV2.business_id (storage UUID).
        """
        log_info(f"[classification_router] Starting classification for file {file_id}")

        try:
            file_uuid = UUID(str(file_id))
        except ValueError as exc:
            raise ValueError(f"Invalid file_id UUID: {file_id}") from exc

        file_row = await db.execute(
            select(IngestedFileV2).where(IngestedFileV2.id == file_uuid)
        )
        ingested_file = file_row.scalar_one_or_none()
        if ingested_file is None:
            raise ValueError(f"[classification_router] File not found: {file_id}")
        if ingested_file.business_id is None:
            raise ValueError(
                f"[classification_router] File {file_id} has no business_id; "
                "cannot resolve tenant for classification."
            )

        client_id = _resolve_client_id_for_config_lookup(str(ingested_file.business_id))

        result = await db.execute(
            select(IngestedContent).where(IngestedContent.file_id == file_id)
        )
        chunks = result.scalars().all()

        if not chunks:
            raise ValueError(f"[classification_router] No chunks found for file {file_id}")

        log_info(
            f"[classification_router] Found {len(chunks)} chunks for tenant={client_id}"
        )

        async def classify_one(chunk):
            try:
                return await ClassificationService.classify_chunk(
                    db=db,
                    client_id=client_id,
                    chunk={
                        "id": str(chunk.id),
                        "cleaned_text": chunk.cleaned_text,
                        "text": chunk.text,
                        "source_type": chunk.source_type,
                    },
                )
            except Exception as e:
                log_warning(f"[classification_router] Chunk {chunk.id} failed: {e}")
                return {
                    "error": str(e),
                    "chunk_id": str(chunk.id),
                }

        results: List[Dict[str, Any]] = []

        if parallel:
            semaphore = asyncio.Semaphore(max_concurrency)

            async def sem_task(chunk):
                async with semaphore:
                    return await classify_one(chunk)

            tasks = [sem_task(c) for c in chunks]
            results = await asyncio.gather(*tasks)
        else:
            for c in chunks:
                results.append(await classify_one(c))

        pending_list = [
            r["pending_taxonomy_id"]
            for r in results
            if isinstance(r, dict) and r.get("pending_taxonomy_id") is not None
        ]

        return {
            "file_id": file_id,
            "client_id": client_id,
            "total_chunks": len(chunks),
            "classified": results,
            "pending_taxonomies": pending_list,
        }
