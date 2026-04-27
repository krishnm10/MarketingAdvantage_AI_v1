"""Prompt Node Contract — Phase 1"""
from __future__ import annotations
import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.ai.contracts.pipeline_node_contract import PipelineNodeContract, NodePosition


@dataclass
class PromptRenderResult:
    rendered_prompt: str
    variables_used: Dict[str, str]
    token_count: Optional[int] = None
    warnings: List[str] = None  # use field(default_factory=list) in __post_init__
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass  
class TemplateValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]
    placeholders_found: List[str]


class PromptNodeContract(PipelineNodeContract):
    """ABC for prompt template rendering nodes."""

    @abc.abstractmethod
    def render(
        self,
        query: str,
        context: str,
        *,
        history: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> PromptRenderResult:
        ...

    @abc.abstractmethod
    def validate_template(
        self,
        template: str,
        variable_map: Dict[str, str],
    ) -> TemplateValidationResult:
        ...

    def positions(self) -> list:
        return [NodePosition.PRE_LLM]

    def node_type(self) -> str:
        return "prompt_node"
