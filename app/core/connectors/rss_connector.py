"""
RSSConnector — pluggable RSS/Atom feed ingestor.
File: app/core/connectors/rss_connector.py

Wraps rss_ingestor_v2.parse_rss() — zero changes to that file.
Auth is injected via BaseAuthProvider — connector never builds credentials.

Supported auth strategies:
  NoAuth     → public feeds (BBC, Reuters, etc.)
  APIKeyAuth → API-key-protected feeds
"""
from __future__ import annotations

from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.connectors.base import BaseConnector, ConnectorResult
from app.core.connectors.auth.base_auth import BaseAuthProvider


class RSSConnector(BaseConnector):
    """
    RSS/Atom feed ingestor with auth support.

    Examples:
        # Public feed — no auth needed:
        connector = RSSConnector()

        # API-key-protected feed:
        connector = RSSConnector(
            auth=APIKeyAuth.from_env("FEEDBURNER_KEY", header_name="X-Api-Key")
        )
    """

    def __init__(self, auth: Optional[BaseAuthProvider] = None):
        super().__init__(auth)

    @property
    def source_type(self) -> str:
        return "rss"

    async def fetch(
        self,
        source_url: str,
        *,
        db_session: Optional[AsyncSession] = None,
        file_id: Optional[str] = None,
        business_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ConnectorResult:
        """
        Fetch and parse an RSS/Atom feed.

        Args:
            source_url:   Full RSS feed URL.
            db_session:   AsyncSession for GCI dedup during ingestion.
            file_id:      Optional file record ID (passed to row_segmenter).
            business_id:  Optional tenant scope for dedup isolation.

        Returns:
            ConnectorResult with chunks + metadata matching
            ingestion_service_v2 expected shape.
        """
        # Lazy import — avoids circular dependency at module load time
        from app.services.ingestion.rss_ingestor_v2 import parse_rss

        parsed = await parse_rss(
            source_url,
            file_id=file_id,
            business_id=business_id,
            db_session=db_session,
        )

        return ConnectorResult(
            source_type=self.source_type,
            source_url=source_url,
            chunks=parsed.get("chunks", []),      # ✅ matches rss_ingestor_v2 return shape
            metadata=parsed.get("metadata", {}),
        )
