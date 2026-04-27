"""
PipelineNodeSet — container for all configured pipeline nodes.
Injected into AssembledPipeline and RAGPipeline via constructor.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PipelineNodeSet:
    """Holds references to all optional pipeline nodes for one client."""
    pii_middleware: Optional[Any] = None  # PIIMiddlewareContract
    prompt_node: Optional[Any] = None     # PromptNodeContract
    output_formatter: Optional[Any] = None  # OutputFormatterContract
    context_window: Optional[Any] = None  # ContextWindowContract
    # Future nodes added here as Optional fields
    
    def has_pii_middleware(self) -> bool:
        return self.pii_middleware is not None
    
    def has_prompt_node(self) -> bool:
        return self.prompt_node is not None
        
    def has_output_formatter(self) -> bool:
        return self.output_formatter is not None
        
    def has_context_window(self) -> bool:
        return self.context_window is not None
