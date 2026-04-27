"""
Pipeline Node Registration — lazy factories.
Import this module once at startup to register all node types.
"""
from __future__ import annotations
from app.core.plugin_registry import pipeline_node_registry


def _pii_middleware_factory(**kwargs):
    from app.core.pipeline_nodes.pii_middleware import RegexPIIMiddleware
    return RegexPIIMiddleware(**kwargs)


def _prompt_node_factory(**kwargs):
    from app.core.pipeline_nodes.prompt_node import PromptNode
    return PromptNode(**kwargs)


def _output_formatter_factory(**kwargs):
    from app.core.pipeline_nodes.output_formatter import OutputFormatter
    return OutputFormatter(**kwargs)


def _context_window_factory(**kwargs):
    from app.core.pipeline_nodes.context_window_manager import ContextWindowManager
    return ContextWindowManager(**kwargs)


pipeline_node_registry.register(
    "pii_middleware",
    _pii_middleware_factory,
    description="PII detection and redaction middleware (regex/custom patterns)",
)

pipeline_node_registry.register(
    "prompt_node",
    _prompt_node_factory,
    description="Configurable prompt template rendering node",
)

pipeline_node_registry.register(
    "output_formatter",
    _output_formatter_factory,
    description="Output formatting and security gate node",
)

pipeline_node_registry.register(
    "context_window_manager",
    _context_window_factory,
    description="Context window budget management node",
)
