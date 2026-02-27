from __future__ import annotations
from typing import Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.connectors.base import BaseConnector, ConnectorResult
from app.core.connectors.auth.base_auth import BaseAuthProvider


class WebConnector(BaseConnector):
    """
    Web page scraper — wraps web_scraper_v2.ingest_webpage().

    No auth (public page):
        connector = WebConnector()
    Basic auth (protected intranet):
        connector = WebConnector(auth=BasicAuth.from_env("USER", "PASS"))
    Form auth (legacy portal):
        connector = WebConnector(auth=FormBasedAuth(...))
    """

    def __init__(self, auth: Optional[BaseAuthProvider] = None):
        super().__init__(auth)

    @property
    def source_type(self) -> str:
        return "web"

    async def fetch(
        self,
        source_url: str,
        *,
        db_session: Optional[AsyncSession] = None,
        **kwargs: Any,
    ) -> ConnectorResult:
        from app.services.ingestion.web_scraper_v2 import ingest_webpage

        parsed = await ingest_webpage(source_url, db_session=db_session)

        return ConnectorResult(
            source_type=self.source_type,
            source_url=source_url,
            chunks=parsed.get("chunks", []),       # ✅ matches web_scraper_v2 return shape
            metadata=parsed.get("metadata", {}),
        )
