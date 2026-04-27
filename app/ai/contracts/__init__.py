# app/ai/contracts/__init__.py
#
# Unified export surface for app.ai.contracts.
#
# Two groups of contracts live here:
#   1. AI Capability ABCs  — migrated from the former app/ai/contracts.py module
#      (SpeechToText, ImageCaptioner, VideoToText, VisualExplainer,
#       LanguageDetector, MultimodalVisionEncoder)
#   2. Phase 1 Embedder contracts — TokenizerContract, EmbedderBundle, etc.
#
# All existing callers that imported from app.ai.contracts continue to work
# without modification.

# ── AI Capability ABCs (original contracts, migrated here) ───────────────────
from app.ai.contracts.ai_capability_contracts import (
    SpeechToText,
    ImageCaptioner,
    VideoToText,
    VisualExplainer,
    LanguageDetector,
    MultimodalVisionEncoder,
)

# ── Phase 1 — Tokenizer contracts ────────────────────────────────────────────
from app.ai.contracts.tokenizer_contract import (
    TokenizerFamily,
    TokenizerContract,
    TokenizerResolutionError,
)

# ── Phase 1 — Embedder contracts ─────────────────────────────────────────────
from app.ai.contracts.embedder_contract import (
    EmbedderProvider,
    DistanceMetric,
    VerificationStatus,
    EmbedderContract,
    EmbedderBundle,
    RerankerView,
)

# ── Phase A — Migration bridge ────────────────────────────────────────────────
from app.ai.contracts.chunking_bridge import ChunkingTokenizerBridge

# ── Phase 2 — Reranker contracts ──────────────────────────────────────────────
from app.ai.contracts.reranker_contract import (
    RerankerScoreSpace,
    RerankerProvider,
    RerankerCapabilities,
    RerankerCandidate,
    ScoredCandidate,
    RerankerContract,
    RerankerInputTooLongError,
    RerankerResolutionError,
)

# ── Phase 2 — Query Transform contracts ──────────────────────────────────────
from app.ai.contracts.query_transform_contract import (
    QueryTransformStrategy,
    TransformedQuery,
    QueryTransformContract,
    PassthroughTransform,
)

# ── Phase 2 — Generator contracts ────────────────────────────────────────────
from app.ai.contracts.generator_contract import (
    GeneratorProvider,
    CitationStyle,
    PromptTemplate,
    GenerationRequest,
    GenerationResult,
    GeneratorContract,
)

# ── Phase 3 — Pipeline Node contracts ────────────────────────────────────────
from app.ai.contracts.pipeline_node_contract import (
    NodePosition,
    TrustSignal,
    NodeResult,
    PipelineNodeContract,
)

from app.ai.contracts.prompt_node_contract import (
    PromptRenderResult,
    TemplateValidationResult,
    PromptNodeContract,
)

from app.ai.contracts.output_formatter_contract import (
    FormatResult,
    OutputFormatterContract,
)

from app.ai.contracts.context_window_contract import (
    ContextBudgetResult,
    ContextWindowContract,
)

from app.ai.contracts.pii_middleware_contract import (
    PIIAction,
    PIISeverity,
    PIIScanResult,
    PIIMiddlewareContract,
)

__all__ = [
    # AI Capability ABCs
    "SpeechToText",
    "ImageCaptioner",
    "VideoToText",
    "VisualExplainer",
    "LanguageDetector",
    "MultimodalVisionEncoder",
    # Phase 1 — Tokenizer
    "TokenizerFamily",
    "TokenizerContract",
    "TokenizerResolutionError",
    # Phase 1 — Embedder
    "EmbedderProvider",
    "DistanceMetric",
    "VerificationStatus",
    "EmbedderContract",
    "EmbedderBundle",
    "RerankerView",
    # Phase A — Bridge
    "ChunkingTokenizerBridge",
    # Phase 2 — Reranker
    "RerankerScoreSpace",
    "RerankerProvider",
    "RerankerCapabilities",
    "RerankerCandidate",
    "ScoredCandidate",
    "RerankerContract",
    "RerankerInputTooLongError",
    "RerankerResolutionError",
    # Phase 2 — Query Transform
    "QueryTransformStrategy",
    "TransformedQuery",
    "QueryTransformContract",
    "PassthroughTransform",
    # Phase 2 — Generator
    "GeneratorProvider",
    "CitationStyle",
    "PromptTemplate",
    "GenerationRequest",
    "GenerationResult",
    "GeneratorContract",
    # Phase 3 — Pipeline Node
    "NodePosition",
    "TrustSignal",
    "NodeResult",
    "PipelineNodeContract",
    # Phase 3 — Prompt Node
    "PromptRenderResult",
    "TemplateValidationResult",
    "PromptNodeContract",
    # Phase 3 — Output Formatter
    "FormatResult",
    "OutputFormatterContract",
    # Phase 3 — Context Window
    "ContextBudgetResult",
    "ContextWindowContract",
    # Phase 3 — PII Middleware
    "PIIAction",
    "PIISeverity",
    "PIIScanResult",
    "PIIMiddlewareContract",
]
