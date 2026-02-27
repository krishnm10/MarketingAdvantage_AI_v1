"""
BaseAuthProvider — contract for all authentication strategies.
Every connector receives an auth provider and calls .get_headers()
or .get_session() — it never knows which auth type is being used.
"""
from __future__ import annotations
import abc
from typing import Any, Dict, Optional


class BaseAuthProvider(abc.ABC):
    """
    All auth strategies implement this interface.
    Connector code ONLY calls get_headers() — never knows the auth type.
    """

    @abc.abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """Return HTTP headers needed for this auth strategy."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_valid(self) -> bool:
        """Return True if credentials are still valid (not expired)."""
        raise NotImplementedError

    def refresh(self) -> None:
        """Refresh credentials if expired. Override where needed."""
        pass
