# =============================================================================
# app/ai/contracts/chunking_bridge.py
#
# ChunkingTokenizerBridge — Phase A Migration Adapter
#
# PURPOSE:
#   Wraps a TokenizerContract (model-native, F-code enforcing) in the
#   BaseTokenizer interface so any existing caller that expects BaseTokenizer
#   transparently receives accurate, model-native token counts.
#
# DIRECTION OF ADAPTATION:
#
#   TokenizerContract  (rich, model-bound, F-code safe)
#         │  wrapped by
#         ▼
#   ChunkingTokenizerBridge(BaseTokenizer)
#         │  presented to
#         ▼
#   Existing callers  (TokenChunkingService, segmenters, factory)
#
# CONTRACT MAPPING:
#   BaseTokenizer.name              → "bridge:<contract.name>"
#   BaseTokenizer.count_tokens(t)  → contract.count_tokens(t, include_special_tokens=False)
#   BaseTokenizer.tokenize(t)      → [str(id) for id in contract.tokenize(t)]
#
# SAFETY GUARANTEE:
#   Any caller holding a ChunkingTokenizerBridge is now silently using the
#   embedding model's own native tokenizer for chunk sizing.
#   This corrects the core accuracy problem in Phase A (F-02 / F-15 / R-C2).
#
# THREAD SAFETY:
#   ChunkingTokenizerBridge is thread-safe if and only if the wrapped
#   TokenizerContract is thread-safe (i.e. contract.is_thread_safe == True).
#   Callers that need guaranteed thread safety should check .is_thread_safe
#   on the underlying contract.
#
# NO CIRCULAR IMPORTS:
#   This module imports only from app.core.tokenization.base (BaseTokenizer)
#   and app.ai.contracts.tokenizer_contract (TokenizerContract).
#   Neither of those imports from this module.
# =============================================================================

from __future__ import annotations

from typing import List

from app.core.tokenization.base import BaseTokenizer
from app.ai.contracts.tokenizer_contract import TokenizerContract


class ChunkingTokenizerBridge(BaseTokenizer):
    """
    Adapts a ``TokenizerContract`` to the ``BaseTokenizer`` interface.

    Use this when you need to pass a model-native ``TokenizerContract``
    to code that was written against ``BaseTokenizer`` — for example,
    ``TokenChunkingService`` or any legacy segmenter that calls
    ``count_tokens()``.

    The bridge is intentionally thin: every call is delegated to the
    wrapped contract with no additional logic or caching.

    Parameters
    ----------
    contract : TokenizerContract
        A fully-initialised ``TokenizerContract`` instance resolved by
        ``EmbedderRegistry``.

    Raises
    ------
    TypeError
        If ``contract`` is not a ``TokenizerContract`` instance.

    Examples
    --------
    >>> from app.ai.registry import embedder_registry
    >>> from app.ai.contracts.chunking_bridge import ChunkingTokenizerBridge
    >>>
    >>> bundle = embedder_registry.get("openai/text-embedding-3-large")
    >>> bridge = ChunkingTokenizerBridge(bundle.tokenizer)
    >>>
    >>> # Pass to TokenChunkingService (or any BaseTokenizer consumer)
    >>> service = TokenChunkingService(tokenizer=bridge, chunk_size=512)
    """

    def __init__(self, contract: TokenizerContract) -> None:
        if not isinstance(contract, TokenizerContract):
            raise TypeError(
                f"ChunkingTokenizerBridge requires a TokenizerContract instance, "
                f"got {type(contract).__name__!r}. "
                "Resolve the contract via EmbedderRegistry.get(model_id).tokenizer."
            )
        self._contract = contract

    # ------------------------------------------------------------------
    # Public read-only accessors
    # ------------------------------------------------------------------

    @property
    def wrapped_contract(self) -> TokenizerContract:
        """Read-only access to the underlying ``TokenizerContract``."""
        return self._contract

    @property
    def is_thread_safe(self) -> bool:
        """
        Reflects the underlying contract's thread-safety declaration.
        Callers that need guaranteed safety must check this before sharing
        a bridge instance across threads.
        """
        return self._contract.is_thread_safe

    # ------------------------------------------------------------------
    # BaseTokenizer interface implementation
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """
        Human-readable identifier.
        Prefixed with ``"bridge:"`` so logs clearly indicate the Phase A
        migration is active for this call-site.
        """
        return f"bridge:{self._contract.name}"

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize ``text`` and return token IDs as strings.

        ``BaseTokenizer`` requires ``List[str]`` (word/sub-word strings).
        ``TokenizerContract`` returns ``List[int]`` (vocabulary IDs).
        The IDs are rendered as decimal strings to satisfy the interface
        while preserving all token information.

        Notes
        -----
        Sentinel tokens ([CLS], [SEP], BOS, EOS) are **not** added.
        Chunking code measures raw content tokens only.
        """
        if not text or not text.strip():
            return []
        ids: List[int] = self._contract.tokenize(text, add_special_tokens=False)
        return [str(token_id) for token_id in ids]

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in ``text`` without special-token overhead.

        Delegates to ``TokenizerContract.count_tokens()`` with
        ``include_special_tokens=False`` — chunkers need raw content
        token counts, not inflated counts that include sentinel tokens.

        This single delegation replaces the generic
        ``bert-base-multilingual-cased`` approximation used by the old
        factory, giving accurate counts for the actual embedding model
        (e.g. ``cl100k_base`` for OpenAI, ``mxbai``-BPE for MixedBread).
        """
        if not text or not text.strip():
            return 0
        return self._contract.count_tokens(text, include_special_tokens=False)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ChunkingTokenizerBridge("
            f"model_id={self._contract.model_id!r}, "
            f"family={self._contract.tokenizer_family.value!r}, "
            f"max_length={self._contract.max_length}, "
            f"thread_safe={self._contract.is_thread_safe})"
        )
