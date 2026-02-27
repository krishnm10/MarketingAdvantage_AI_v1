"""
BaseConnector — contract all external source connectors must implement.
File: app/core/connectors/base.py
"""
from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.connectors.auth.base_auth import BaseAuthProvider


@dataclass
class ConnectorResult:
    """
    Normalized output from any connector — same shape regardless of source.
    to_dict() returns exactly what ingestion_service_v2.ingest_parsed_output()
    expects: {"chunks": [...], "source_type": "...", "metadata": {...}}
    """
    source_type: str
    source_url:  str
    chunks:      List[Dict[str, Any]] = field(default_factory=list)   # ✅ chunks not entries
    metadata:    Dict[str, Any]       = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks":      self.chunks,                               # ✅ matches ingestion_service_v2
            "source_type": self.source_type,
            "metadata":    {**self.metadata, "source_url": self.source_url},
        }


class BaseConnector(abc.ABC):
    """
    All connectors receive auth via constructor injection.
    Connector only calls self.auth.get_headers() — never knows which auth is active.
    """

    def __init__(self, auth: Optional[BaseAuthProvider] = None):
        from app.core.connectors.auth.providers import NoAuth
        self.auth: BaseAuthProvider = auth or NoAuth()

    @property
    @abc.abstractmethod
    def source_type(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    async def fetch(
        self,
        source_url: str,
        *,
        db_session: Optional[AsyncSession] = None,
        **kwargs: Any,
    ) -> ConnectorResult:
        raise NotImplementedError

    def health_check(self) -> bool:
        return self.auth.is_valid()
