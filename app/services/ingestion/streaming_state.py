"""
Streaming ingestion checkpoint and L1 dedup state for bounded-window pipelines.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import ClassVar
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.ingested_file_v2 import IngestedFileV2

logger = logging.getLogger(__name__)

STREAMING_INGESTION_ENABLED: bool = (
    os.environ.get("MAI_STREAMING_INGESTION", "0") == "1"
)


@dataclass
class StreamingDedupState:
    """Best-effort L1 dedup within a single ingestion run (not persisted on resume)."""

    seen_hashes: set[str] = field(default_factory=set)
    MAX_SEEN_HASHES: ClassVar[int] = int(
        os.environ.get("MAI_MAX_SEEN_HASHES", "500000")
    )

    def add(self, h: str) -> None:
        if len(self.seen_hashes) >= self.MAX_SEEN_HASHES:
            logger.warning(
                "StreamingDedupState at capacity (%d); "
                "falling through to L2 dedup for hash %s",
                self.MAX_SEEN_HASHES,
                h[:16],
            )
            return
        self.seen_hashes.add(h)

    def is_seen(self, h: str) -> bool:
        if len(self.seen_hashes) >= self.MAX_SEEN_HASHES:
            return False
        return h in self.seen_hashes


@dataclass
class StreamingIngestionState:
    """
    Per-file streaming progress. L1 dedup (seen_hashes) is not reconstructed
    from DB on resume; L2/L3 remain authoritative across retries.
    """

    file_id: UUID
    business_id: UUID
    last_processed_chunk_index: int = 0
    last_processed_page: int = 0
    total_chunks: int = 0
    unique_chunks: int = 0
    duplicate_chunks: int = 0
    current_window_index: int = 0
    dedup: StreamingDedupState = field(default_factory=StreamingDedupState)

    @classmethod
    async def from_checkpoint(
        cls,
        db: AsyncSession,
        file_id: UUID,
        business_id: UUID,
    ) -> StreamingIngestionState:
        """
        Load checkpoint from IngestedFileV2.
        TENANT GUARD: always filters by BOTH file_id AND business_id.
        Raises ValueError if no matching record found.
        """
        result = await db.execute(
            select(IngestedFileV2).where(
                IngestedFileV2.id == file_id,
                IngestedFileV2.business_id == business_id,
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            raise ValueError(
                f"No IngestedFileV2 record for file_id={file_id} "
                f"and business_id={business_id}. "
                f"Possible tenant isolation violation or missing record."
            )
        return cls(
            file_id=file_id,
            business_id=business_id,
            last_processed_chunk_index=record.last_processed_chunk_index or 0,
            last_processed_page=record.last_processed_page or 0,
        )

    async def persist_checkpoint(self, db: AsyncSession) -> None:
        """Write current progress back to IngestedFileV2. Caller owns commit."""
        await db.execute(
            update(IngestedFileV2)
            .where(
                IngestedFileV2.id == self.file_id,
                IngestedFileV2.business_id == self.business_id,
            )
            .values(
                last_processed_chunk_index=self.last_processed_chunk_index,
                last_processed_page=self.last_processed_page,
                last_processed_at=func.now(),
            )
        )
