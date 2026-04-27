"""Output Formatter Contract — Phase 1"""
from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.ai.contracts.pipeline_node_contract import PipelineNodeContract, NodePosition


@dataclass
class FormatResult:
    formatted_output: str
    format_type: str
    blocked: bool = False
    block_reason: Optional[str] = None
    trust_gate_passed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class OutputFormatterContract(PipelineNodeContract):
    """ABC for output formatting and security gate nodes."""

    @abc.abstractmethod
    def format_output(
        self,
        raw_answer: str,
        *,
        pii_redacted: bool = False,
        pii_entities_found: Optional[List[str]] = None,
        trust_score: Optional[float] = None,
    ) -> FormatResult:
        ...

    def positions(self) -> list:
        return [NodePosition.POST_LLM]

    def node_type(self) -> str:
        return "output_formatter"
