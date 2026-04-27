"""
Pipeline Node Contracts — Phase 1
Base ABCs for all configurable pipeline nodes.
"""
from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class NodePosition(str, Enum):
    PRE_EMBEDDING = "pre_embedding"
    PRE_RETRIEVAL = "pre_retrieval"
    POST_RETRIEVAL = "post_retrieval"
    PRE_LLM = "pre_llm"
    POST_LLM = "post_llm"
    POST_FORMAT = "post_format"


@dataclass
class TrustSignal:
    """Signal emitted by nodes that affect answer quality."""
    source_node: str
    delta: float  # positive = raises trust, negative = lowers
    reason: str = ""


@dataclass
class NodeResult:
    """Generic result envelope from any pipeline node execution."""
    output: Any = None
    trust_signals: List[TrustSignal] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineNodeContract(abc.ABC):
    """Base ABC for all pipeline nodes."""

    @abc.abstractmethod
    def node_type(self) -> str:
        """Return the registered node type key (e.g. 'prompt_node')."""
        ...

    @abc.abstractmethod
    def positions(self) -> List[NodePosition]:
        """Return the pipeline positions this node operates in."""
        ...

    @abc.abstractmethod
    def execute(self, context: Dict[str, Any]) -> NodeResult:
        """Execute the node logic given pipeline context."""
        ...

    def validate_config(self) -> List[str]:
        """Return list of validation errors (empty = valid)."""
        return []
