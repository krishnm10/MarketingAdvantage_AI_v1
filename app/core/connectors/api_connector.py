"""
APIConnector — pluggable REST API ingestor.
File: app/core/connectors/api_connector.py

Wraps api_ingestor_v2.ingest_api_data() — zero changes to that file.
Auth headers from any provider are passed directly into ingest_api_data(headers=).
The underlying ingestor never knows which auth strategy is active.

Supported auth strategies:
  NoAuth      → open public APIs
  APIKeyAuth  → NewsAPI, RapidAPI, custom APIs
  BasicAuth   → legacy APIs with username/password
  OAuth2Auth  → Salesforce, HubSpot, Google APIs
  AzureADAuth → Microsoft Graph, SharePoint REST API
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.connectors.base import BaseConnector, ConnectorResult
from app.core.connectors.auth.base_auth import BaseAuthProvider


class APIConnector(BaseConnector):
    """
    REST API ingestor — supports all auth strategies via injection.

    Examples:
        # Open API — no auth:
        connector = APIConnector()

        # NewsAPI with API key:
        connector = APIConnector(
            auth=APIKeyAuth.from_env("NEWSAPI_KEY", header_name="X-Api-Key")
        )

        # Salesforce via OAuth2:
        connector = APIConnector(
            auth=OAuth2Auth(
                token_url="https://login.salesforce.com/services/oauth2/token",
                client_id=os.environ["SF_CLIENT_ID"],
                client_secret=os.environ["SF_CLIENT_SECRET"],
            )
        )

        # Microsoft Graph via Azure AD:
        connector = APIConnector(
            auth=AzureADAuth(
                tenant_id=os.environ["AZURE_TENANT_ID"],
                client_id=os.environ["AZURE_CLIENT_ID"],
                client_secret=os.environ["AZURE_CLIENT_SECRET"],
                scope="https://graph.microsoft.com/.default",
            )
        )
    """

    def __init__(self, auth: Optional[BaseAuthProvider] = None):
        super().__init__(auth)

    @property
    def source_type(self) -> str:
        return "api"

    async def fetch(
        self,
        source_url: str,
        *,
        db_session: Optional[AsyncSession] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> ConnectorResult:
        """
        Fetch and ingest a REST API endpoint.

        Args:
            source_url:     Full API endpoint URL.
            db_session:     AsyncSession for GCI dedup during ingestion.
            extra_headers:  Any additional headers to merge (optional).
                            Auth headers from self.auth take precedence.

        Returns:
            ConnectorResult with chunks + metadata matching
            ingestion_service_v2 expected shape.
        """
        # Lazy import — avoids circular dependency at module load time
        from app.services.ingestion.api_ingestor_v2 import ingest_api_data

        # Merge extra headers first, then auth headers on top
        # Auth headers always win on key conflicts (security-correct)
        headers: Dict[str, str] = {}
        if extra_headers:
            headers.update(extra_headers)
        headers.update(self.auth.get_headers())   # ✅ auth injected transparently

        parsed = await ingest_api_data(
            source_url,
            headers=headers if headers else None,  # pass None if empty (NoAuth)
            db_session=db_session,
        )

        return ConnectorResult(
            source_type=self.source_type,
            source_url=source_url,
            chunks=parsed.get("chunks", []),       # ✅ matches api_ingestor_v2 return shape
            metadata=parsed.get("metadata", {}),
        )
