"""
Marketing Advantage AI - Enterprise Plugin Registry
File: app/core/plugin_registry.py

Goal
----
Provide a stable, testable, enterprise-grade plugin system to make *everything*
pluggable:
- Vector DB connectors (Chroma/Qdrant/Weaviate/Pinecone/Milvus/...)
- Embedders / transformers
- LLM providers (single + chain)
- Rerankers
- Ingestion connectors

Why a new module?
-----------------
Your repo already contains an AI registry concept under app/ai (e.g., registry.py,
contracts.py). We are NOT deleting those. [cite:3]
This module is an additive, modernized registry that we can gradually adopt
across ingestion + RAG without breaking existing behavior.

Design principles
-----------------
- Explicit registration, explicit construction (no "magic" imports required).
- Strict name keys, predictable errors.
- Optional lazy singletons (for heavy models) via a provider function.
- Small surface area: register(), build(), get(), list().
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar, Union

T = TypeVar("T")


class PluginError(RuntimeError):
    """Base exception for plugin registry errors."""


class PluginNotFoundError(PluginError):
    """Raised when a plugin name is not registered."""


class PluginAlreadyRegisteredError(PluginError):
    """Raised when a plugin is registered twice (unless allow_override=True)."""


# A plugin entry can be either:
# 1) a class (callable constructor), or
# 2) a factory function returning an instance (useful for lazy singletons).
PluginFactory = Callable[..., T]
PluginEntry = Union[Type[T], PluginFactory[T]]


@dataclass(frozen=True)
class PluginSpec(Generic[T]):
    """
    Registered plugin metadata.
    Keeping this tiny makes it easy to extend later (versioning, caps, etc.).
    """
    name: str
    entry: PluginEntry[T]
    description: str = ""


class PluginRegistry(Generic[T]):
    """
    Thread-safe plugin registry.

    Examples
    --------
    registry = PluginRegistry["BaseVectorDB"]("vectordb")

    # register class
    registry.register("chroma", ChromaVectorDB, description="Local persistent ChromaDB")

    # register factory (lazy singleton or dependency-driven creation)
    registry.register("qdrant", lambda **kw: QdrantVectorDB(**kw))

    db = registry.build("chroma", persist_directory="./chroma_db")
    """

    def __init__(self, domain: str):
        self._domain = domain
        self._lock = RLock()
        self._specs: Dict[str, PluginSpec[T]] = {}

    @property
    def domain(self) -> str:
        return self._domain

    def register(
        self,
        name: str,
        entry: PluginEntry[T],
        *,
        description: str = "",
        allow_override: bool = False,
    ) -> None:
        key = (name or "").strip().lower()
        if not key:
            raise ValueError(f"[{self._domain}] Plugin name cannot be empty.")

        with self._lock:
            if key in self._specs and not allow_override:
                raise PluginAlreadyRegisteredError(
                    f"[{self._domain}] Plugin '{key}' already registered."
                )

            self._specs[key] = PluginSpec(name=key, entry=entry, description=description)

    def get(self, name: str) -> PluginSpec[T]:
        key = (name or "").strip().lower()
        with self._lock:
            spec = self._specs.get(key)
            if not spec:
                raise PluginNotFoundError(
                    f"[{self._domain}] Plugin '{key}' not found. "
                    f"Available: {sorted(self._specs.keys())}"
                )
            return spec

    def build(self, name: str, /, **kwargs: Any) -> T:
        """
        Instantiate plugin by name.

        For classes: calls cls(**kwargs)
        For factories: calls factory(**kwargs)
        """
        spec = self.get(name)
        entry = spec.entry
        return entry(**kwargs)  # type: ignore[misc]

    def list(self) -> Dict[str, str]:
        """Return {plugin_name: description}."""
        with self._lock:
            return {k: v.description for k, v in sorted(self._specs.items())}

    def has(self, name: str) -> bool:
        key = (name or "").strip().lower()
        with self._lock:
            return key in self._specs


# ---------------------------------------------------------------------
# Global registries (singletons)
# ---------------------------------------------------------------------
# Keep these in one module so every part of the system uses the same instances.
vectordb_registry: PluginRegistry[Any] = PluginRegistry(domain="vectordb")
embedder_registry: PluginRegistry[Any] = PluginRegistry(domain="embedder")
llm_registry: PluginRegistry[Any] = PluginRegistry(domain="llm")
reranker_registry: PluginRegistry[Any] = PluginRegistry(domain="reranker")
ingestor_registry: PluginRegistry[Any] = PluginRegistry(domain="ingestor")

# ---------------------------------------------------------------------
# Bootstrap built-in connectors into ingestor_registry
# ---------------------------------------------------------------------
def _bootstrap_connectors() -> None:
    from app.core.connectors.web_connector import WebConnector
    from app.core.connectors.rss_connector import RSSConnector
    from app.core.connectors.api_connector import APIConnector

    # Default registrations — NoAuth by default, overridden per client at runtime
    ingestor_registry.register("web", lambda **kw: WebConnector(**kw),
                                description="Web scraper — supports NoAuth/Basic/Form/OAuth2")
    ingestor_registry.register("rss", lambda **kw: RSSConnector(**kw),
                                description="RSS/Atom feed — supports NoAuth/APIKey")
    ingestor_registry.register("api", lambda **kw: APIConnector(**kw),
                                description="REST API — supports APIKey/Basic/OAuth2/AzureAD")

_bootstrap_connectors()

