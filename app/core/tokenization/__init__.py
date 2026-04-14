# =============================================================================
# app.core.tokenization — Enterprise Token-Aware Tokenization Engine
#
# Production-grade, CPU-optimized tokenization with multilingual/Indic support.
# Factory pattern: switch backends via DEFAULT_TOKENIZER_BACKEND env var.
#
# Public API:
#   get_tokenizer(backend)       → BaseTokenizer instance (cached)
#   count_tokens(text, backend)  → accurate token count
#   tokenize(text, backend)      → list of token strings
#   TokenizerBackend             → enum of supported backends
# =============================================================================

from app.core.tokenization.base import BaseTokenizer, TokenizerBackend
from app.core.tokenization.factory import (
    get_tokenizer,
    count_tokens,
    tokenize,
    list_backends,
)

__all__ = [
    "BaseTokenizer",
    "TokenizerBackend",
    "get_tokenizer",
    "count_tokens",
    "tokenize",
    "list_backends",
]
