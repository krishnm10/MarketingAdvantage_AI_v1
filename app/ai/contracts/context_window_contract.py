"""Context Window Manager Contract — Phase 1"""
from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.ai.contracts.pipeline_node_contract import PipelineNodeContract, NodePosition


@dataclass
class ContextBudgetResult:
    trimmed_chunks: List[Dict[str, Any]]
    chunks_used: int
    chunks_dropped: int
    total_tokens: int
    budget_tokens: int
    truncation_applied: bool = False
    strategy_used: str = "none"


class ContextWindowContract(PipelineNodeContract):
    """ABC for context window management nodes."""

    @abc.abstractmethod
    def apply_budget(
        self,
        chunks: List[Dict[str, Any]],
        *,
        max_context_tokens: int,
        system_prompt_tokens: int = 0,
        response_reserve_tokens: int = 1024,
    ) -> ContextBudgetResult:
        ...

    def positions(self) -> list:
        return [NodePosition.POST_RETRIEVAL]

    def node_type(self) -> str:
        return "context_window_manager"
