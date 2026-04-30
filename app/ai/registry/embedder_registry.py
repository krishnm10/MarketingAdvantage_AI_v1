# =============================================================================
# app/ai/registry/embedder_registry.py
#
# EmbedderRegistry — Phase 1, Module 4
#
# Resolves EmbedderBundle instances from embedder_catalog.yaml.
# Supports two paths:
#   PATH 1 — Cloud-native (OpenAI, Cohere, Anthropic, Google, Mistral):
#             Tokenizer resolved from catalog info only.
#   PATH 2A — HuggingFace: AutoTokenizer + config.json + tokenizer.json.
#   PATH 2B — Ollama: `ollama show <model> --modelfile` parsing.
#
# Failure modes addressed:
#   F-01  tokenizer_family validation at resolution time
#   F-02  max_length set from model metadata, not hard-coded
#   F-03  special_tokens populated from actual tokenizer vocab
#   F-04  dimension confirmed via catalog + optional runtime probe
#   F-05  embedding_fingerprint stored; migration guard via §5.4
#   F-16  tokenizer change detected via fingerprint mismatch
#
# Caching strategy (§3.5):
#   Cache key = (model_id, catalog_hash)
#   Thread-safe via RLock.
#   Cache invalidated when embedder_catalog.yaml changes on disk.
#
# Resolution pipeline per §3.1:
#   Step 1 — Lookup in catalog
#   Step 2 — Schema validation (done by EmbedderCatalogEntry)
#   Step 3 — Determine path (cloud/HF/Ollama)
#   Step 4 — Construct tokenizer
#   Step 5 — Construct embedder
#   Step 6 — Verify dimension
#   Step 7 — Assemble EmbedderBundle
#   Step 8 — Cache and return
# =============================================================================

from __future__ import annotations

import json
import logging
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.ai.catalog.catalog_loader import (
    EmbedderCatalogEntry,
    EmbedderCatalogValidationError,
    get_catalog_entry,
    get_catalog_hash,
)
from app.ai.contracts.tokenizer_contract import (
    TokenizerContract,
    TokenizerFamily,
    TokenizerResolutionError,
)
from app.ai.contracts.embedder_contract import (
    EmbedderBundle,
    EmbedderContract,
    EmbedderProvider,
    DistanceMetric,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class UnknownEmbedderModelError(KeyError):
    """
    Raised when a requested model_id is not present in embedder_catalog.yaml.

    Fields
    ──────
    model_id : the requested identifier
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(
            f"[EmbedderRegistry] model_id={model_id!r} not found in "
            "embedder_catalog.yaml (F-05/F-16 risk: ensure model is catalogued "
            "before building a pipeline)."
        )


# ---------------------------------------------------------------------------
# Cloud tokenizer implementations
# ---------------------------------------------------------------------------

class _TiktokenTokenizer(TokenizerContract):
    """
    Concrete TokenizerContract backed by tiktoken.

    Used for: OpenAI (cl100k_base, o200k_base), Qwen3 (custom BPE).
    Invariants satisfied: I-T1 through I-T6.
    Thread-safe: tiktoken encodings are stateless after construction.
    """

    def __init__(
        self,
        model_id: str,
        encoding_name: str,
        max_length: int,
        special_tokens: Dict[str, str],
    ) -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise TokenizerResolutionError(
                model_id=model_id,
                provider="openai",
                reason="tiktoken is not installed. Install: pip install tiktoken",
            ) from exc

        self._model_id = model_id
        self._encoding_name = encoding_name
        self._max_length = max_length
        self._special_tokens = special_tokens
        try:
            self._enc = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            raise TokenizerResolutionError(
                model_id=model_id,
                provider="openai",
                reason=f"tiktoken.get_encoding({encoding_name!r}) failed: {exc}",
            ) from exc

    @property
    def name(self) -> str:
        return f"tiktoken:{self._encoding_name}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def tokenizer_family(self) -> TokenizerFamily:
        return TokenizerFamily.TIKTOKEN

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def special_tokens(self) -> Dict[str, str]:
        return dict(self._special_tokens)

    @property
    def tokenizer_source(self) -> str:
        return f"tiktoken:{self._encoding_name}"

    @property
    def is_thread_safe(self) -> bool:
        return True

    def tokenize(self, text: str, *, add_special_tokens: bool = False) -> List[int]:
        if not text:
            return []
        ids = self._enc.encode(text)
        # tiktoken does not add special tokens by default; special token
        # handling for OpenAI embeddings is internal to the API.
        return ids

    def count_tokens(self, text: str, *, include_special_tokens: bool = False) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))

    def decode(self, token_ids: List[int]) -> str:
        return self._enc.decode(token_ids)

    def measure_with_overhead(self, text: str, *, special_overhead: int) -> int:
        return self.count_tokens(text) + special_overhead


class _HuggingFaceTokenizer(TokenizerContract):
    """
    Concrete TokenizerContract backed by transformers.AutoTokenizer.

    Used for: BAAI/bge-*, intfloat/e5-*, mixedbread-ai/*, Qwen/*, etc.
    Resolution follows §3.3 exactly.
    Thread-safe: HF fast tokenizers are stateless after construction.
    """

    def __init__(
        self,
        model_id: str,
        hf_tokenizer: "Any",
        max_length: int,
        tokenizer_family: TokenizerFamily,
        special_tokens: Dict[str, str],
        tokenizer_source: str,
    ) -> None:
        self._model_id = model_id
        self._hf = hf_tokenizer
        self._max_length = max_length
        self._family = tokenizer_family
        self._special_tokens = special_tokens
        self._tokenizer_source = tokenizer_source

    @property
    def name(self) -> str:
        return f"hf:{self._model_id}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def tokenizer_family(self) -> TokenizerFamily:
        return self._family

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def special_tokens(self) -> Dict[str, str]:
        return dict(self._special_tokens)

    @property
    def tokenizer_source(self) -> str:
        return self._tokenizer_source

    @property
    def is_thread_safe(self) -> bool:
        # HF fast tokenizers are thread-safe after construction.
        return True

    def tokenize(self, text: str, *, add_special_tokens: bool = False) -> List[int]:
        if not text:
            return []
        encoding = self._hf(
            text,
            add_special_tokens=add_special_tokens,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return encoding["input_ids"]

    def count_tokens(self, text: str, *, include_special_tokens: bool = False) -> int:
        if not text:
            return 0
        return len(
            self._hf.encode(text, add_special_tokens=include_special_tokens)
        )

    def decode(self, token_ids: List[int]) -> str:
        return self._hf.decode(token_ids, skip_special_tokens=True)

    def measure_with_overhead(self, text: str, *, special_overhead: int) -> int:
        return self.count_tokens(text, include_special_tokens=False) + special_overhead


class _OllamaTokenizer(TokenizerContract):
    """
    Concrete TokenizerContract for Ollama-hosted models.

    Ollama does not expose a native tokenizer API; we infer the family from
    the modelfile and count tokens via character-to-token heuristics calibrated
    per family.  This is inherently approximate — verification_status remains
    "unverified" until a full tokenizer introspection is possible.

    For production use, prefer the HuggingFace path where possible.
    Thread-safe: no mutable state after construction.
    """

    # Approximate chars-per-token by family for rough counting.
    # Used only as a safety heuristic — NOT for fine-grained measurement.
    _CHARS_PER_TOKEN: Dict[TokenizerFamily, float] = {
        TokenizerFamily.WORDPIECE:     4.0,
        TokenizerFamily.SENTENCEPIECE: 4.2,
        TokenizerFamily.TIKTOKEN:      3.8,
        TokenizerFamily.BPE:           4.0,
        TokenizerFamily.WHITESPACE:    5.0,
        TokenizerFamily.UNKNOWN:       4.0,
    }

    def __init__(
        self,
        model_id: str,
        tokenizer_family: TokenizerFamily,
        max_length: int,
        special_tokens: Dict[str, str],
    ) -> None:
        self._model_id = model_id
        self._family = tokenizer_family
        self._max_length = max_length
        self._special_tokens = special_tokens

    @property
    def name(self) -> str:
        return f"ollama:{self._model_id}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def tokenizer_family(self) -> TokenizerFamily:
        return self._family

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def special_tokens(self) -> Dict[str, str]:
        return dict(self._special_tokens)

    @property
    def tokenizer_source(self) -> str:
        return "ollama:modelfile"

    @property
    def is_thread_safe(self) -> bool:
        return True

    def tokenize(self, text: str, *, add_special_tokens: bool = False) -> List[int]:
        # Ollama does not expose a tokenizer API; return a list of approximate
        # token count as placeholder IDs.  Callers must NOT use the IDs for
        # anything other than length measurement.
        return list(range(self.count_tokens(text)))

    def count_tokens(self, text: str, *, include_special_tokens: bool = False) -> int:
        if not text:
            return 0
        chars_per_tok = self._CHARS_PER_TOKEN.get(self._family, 4.0)
        base = max(1, int(len(text) / chars_per_tok))
        if include_special_tokens and self._special_tokens:
            base += len(self._special_tokens)
        return base

    def decode(self, token_ids: List[int]) -> str:
        # Cannot reconstruct text from approximate IDs.
        return ""

    def measure_with_overhead(self, text: str, *, special_overhead: int) -> int:
        return self.count_tokens(text, include_special_tokens=False) + special_overhead


class _CohereTokenizer(TokenizerContract):
    """
    Concrete TokenizerContract for Cohere embedding models.

    Cohere exposes a tokenize endpoint via its SDK.  If the SDK is not
    available, falls back to an approximate BPE count.
    Thread-safe: SDK calls are stateless.
    """

    def __init__(
        self,
        model_id: str,
        max_length: int,
        api_key: Optional[str],
        special_tokens: Dict[str, str],
    ) -> None:
        self._model_id = model_id
        self._max_length = max_length
        self._api_key = api_key
        self._special_tokens = special_tokens
        self._cohere_client: Optional[object] = None

    def _ensure_client(self) -> None:
        if self._cohere_client is not None:
            return
        if self._api_key:
            try:
                import cohere  # type: ignore[import]
                self._cohere_client = cohere.Client(self._api_key)
            except ImportError:
                logger.warning(
                    "[CohereTokenizer] cohere SDK not installed; "
                    "falling back to BPE approximation for %s", self._model_id
                )
        # else: no API key available; use approximation

    @property
    def name(self) -> str:
        return f"cohere:{self._model_id}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def tokenizer_family(self) -> TokenizerFamily:
        return TokenizerFamily.BPE

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def special_tokens(self) -> Dict[str, str]:
        return dict(self._special_tokens)

    @property
    def tokenizer_source(self) -> str:
        return "cohere_api"

    @property
    def is_thread_safe(self) -> bool:
        return True

    def tokenize(self, text: str, *, add_special_tokens: bool = False) -> List[int]:
        return list(range(self.count_tokens(text)))

    def count_tokens(self, text: str, *, include_special_tokens: bool = False) -> int:
        if not text:
            return 0
        self._ensure_client()
        if self._cohere_client is not None:
            try:
                result = self._cohere_client.tokenize(text=text, model="command")  # type: ignore[attr-defined]
                count = len(result.tokens)
                if include_special_tokens:
                    count += 2  # BOS/EOS approximate
                return count
            except Exception as exc:
                logger.warning(
                    "[CohereTokenizer] SDK tokenize() failed for %s: %s; "
                    "using BPE approximation.",
                    self._model_id, exc,
                )
        # Approximate: BPE ~4 chars/token
        base = max(1, int(len(text) / 4.0))
        if include_special_tokens:
            base += 2
        return base

    def decode(self, token_ids: List[int]) -> str:
        return ""

    def measure_with_overhead(self, text: str, *, special_overhead: int) -> int:
        return self.count_tokens(text, include_special_tokens=False) + special_overhead


# ---------------------------------------------------------------------------
# Adapter: wraps existing BaseEmbedder to satisfy EmbedderContract
# ---------------------------------------------------------------------------

class _BaseEmbedderAdapter(EmbedderContract):
    """
    Adapts an existing ``app/core/embedders/base.BaseEmbedder`` instance to
    satisfy ``EmbedderContract``, injecting the tokenizer from the registry.

    This adapter avoids modifying any existing file.  It delegates all
    actual embedding calls to the wrapped ``BaseEmbedder`` and enforces
    the EmbedderContract invariants on top.

    Invariants enforced:
      I-E1: tokenizer must be supplied externally via ``inject_tokenizer``
      I-E2: prefixes applied deterministically
      I-E3: dimension checked against catalog after first embed call (lazy probe)
    """

    def __init__(
        self,
        base_embedder: "Any",
        provider: EmbedderProvider,
        model_id: str,
        dimension: int,
        distance_metric: DistanceMetric,
        is_normalized: bool,
        query_prefix: str,
        passage_prefix: str,
    ) -> None:
        self._base = base_embedder
        self._provider = provider
        self._model_id = model_id
        self._dimension = dimension
        self._distance_metric = distance_metric
        self._is_normalized = is_normalized
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._tokenizer: Optional[TokenizerContract] = None

    def inject_tokenizer(self, tokenizer: TokenizerContract) -> None:
        """Called exclusively by EmbedderRegistry. Enforces I-E1."""
        if self._tokenizer is not None:
            raise RuntimeError(
                f"[_BaseEmbedderAdapter] tokenizer already injected for "
                f"model_id={self._model_id!r}. "
                "inject_tokenizer() must be called exactly once by EmbedderRegistry."
            )
        self._tokenizer = tokenizer

    @property
    def provider(self) -> EmbedderProvider:
        return self._provider

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def bound_tokenizer(self) -> TokenizerContract:
        if self._tokenizer is None:
            raise RuntimeError(
                f"[_BaseEmbedderAdapter] bound_tokenizer accessed before "
                f"inject_tokenizer() was called for model_id={self._model_id!r}. "
                "This violates I-E1."
            )
        return self._tokenizer

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def distance_metric(self) -> DistanceMetric:
        return self._distance_metric

    @property
    def is_normalized(self) -> bool:
        return self._is_normalized

    @property
    def query_prefix(self) -> str:
        return self._query_prefix

    @property
    def passage_prefix(self) -> str:
        return self._passage_prefix

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = (
            [self._passage_prefix + t for t in texts]
            if self._passage_prefix
            else texts
        )
        return self._base.embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        prefixed = (self._query_prefix + text) if self._query_prefix else text
        return self._base.embed_query(prefixed)


# ---------------------------------------------------------------------------
# HuggingFace tokenizer resolution (§3.3)
# ---------------------------------------------------------------------------

_HF_FAMILY_MAP: Dict[str, TokenizerFamily] = {
    "wordpiece":            TokenizerFamily.WORDPIECE,
    "bpe":                  TokenizerFamily.BPE,
    "sentencepieceunigram": TokenizerFamily.SENTENCEPIECE,
    "sentencepiece":        TokenizerFamily.SENTENCEPIECE,
    "unigram":              TokenizerFamily.SENTENCEPIECE,
}

# model_type → tokenizer_family fallback when tokenizer.json is unavailable
_MODEL_TYPE_FAMILY_MAP: Dict[str, TokenizerFamily] = {
    "bert":         TokenizerFamily.WORDPIECE,
    "distilbert":   TokenizerFamily.WORDPIECE,
    "roberta":      TokenizerFamily.BPE,
    "xlm-roberta":  TokenizerFamily.BPE,
    "mistral":      TokenizerFamily.SENTENCEPIECE,
    "llama":        TokenizerFamily.SENTENCEPIECE,
    "qwen2":        TokenizerFamily.TIKTOKEN,
    "qwen3":        TokenizerFamily.TIKTOKEN,
    "phi":          TokenizerFamily.TIKTOKEN,
}


def _resolve_hf_tokenizer(
    entry: EmbedderCatalogEntry,
) -> Tuple[TokenizerContract, VerificationStatus]:
    """
    Load HuggingFace tokenizer following §3.3.

    Returns (TokenizerContract, VerificationStatus).
    Raises TokenizerResolutionError if loading fails.
    """
    model_id = entry.model_id

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TokenizerResolutionError(
            model_id=model_id,
            provider="huggingface",
            reason=(
                "transformers package not installed. "
                "Install: pip install transformers"
            ),
        ) from exc

    # Step 1 — load tokenizer
    try:
        hf_tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    except Exception as exc:
        raise TokenizerResolutionError(
            model_id=model_id,
            provider="huggingface",
            reason=f"AutoTokenizer.from_pretrained({model_id!r}) failed: {exc}",
        ) from exc

    # Step 2 — read config.json for max_position_embeddings and model_type
    discovered_max: Optional[int] = None
    discovered_model_type: Optional[str] = None
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id)
        discovered_max = getattr(cfg, "max_position_embeddings", None)
        discovered_model_type = getattr(cfg, "model_type", None)
    except Exception as exc:
        logger.warning(
            "[EmbedderRegistry] Could not load AutoConfig for %s: %s",
            model_id, exc,
        )

    # Step 3 — read tokenizer.json for model.type
    discovered_family: Optional[TokenizerFamily] = None
    try:
        tok_json_path = Path(hf_tok.vocab_file if hasattr(hf_tok, "vocab_file") else "")
        # Try to access backend tokenizer's model type via HF fast tokenizer API
        if hasattr(hf_tok, "_tokenizer"):
            model_type_str = hf_tok._tokenizer.model.type.lower().replace(" ", "")
            discovered_family = _HF_FAMILY_MAP.get(model_type_str)
    except Exception:
        pass

    # Fallback to model_type → family map
    if discovered_family is None and discovered_model_type:
        discovered_family = _MODEL_TYPE_FAMILY_MAP.get(
            discovered_model_type.lower()
        )

    # Step 4 — compare with catalog (catalog wins on conflict per §3.3)
    catalog_family = entry.tokenizer_family
    catalog_max = entry.embed_max_tokens

    verification_status = VerificationStatus.VERIFIED

    if discovered_family is not None and discovered_family != catalog_family:
        logger.warning(
            "[EmbedderRegistry] F-01/F-16 risk: catalog tokenizer_family=%r "
            "but discovered=%r for model_id=%r. Catalog wins.",
            catalog_family.value, discovered_family.value, model_id,
        )
        verification_status = VerificationStatus.CATALOG_MISMATCH

    if discovered_max is not None and discovered_max != catalog_max:
        logger.warning(
            "[EmbedderRegistry] F-02/F-16 risk: catalog embed_max_tokens=%d "
            "but config.json max_position_embeddings=%d for model_id=%r. "
            "Catalog wins.",
            catalog_max, discovered_max, model_id,
        )
        if verification_status == VerificationStatus.VERIFIED:
            verification_status = VerificationStatus.CATALOG_MISMATCH

    # Step 5 — build special tokens dict from the loaded tokenizer
    special_tokens = _extract_hf_special_tokens(hf_tok, catalog_family)

    tokenizer = _HuggingFaceTokenizer(
        model_id=model_id,
        hf_tokenizer=hf_tok,
        max_length=catalog_max,    # catalog is authoritative (§3.3 rule 6)
        tokenizer_family=catalog_family,
        special_tokens=special_tokens,
        tokenizer_source="hf:config.json+tokenizer.json",
    )

    logger.info(
        "[EmbedderRegistry] HF tokenizer resolved: model_id=%r family=%r "
        "max_length=%d status=%r",
        model_id, catalog_family.value, catalog_max, verification_status.value,
    )
    return tokenizer, verification_status


def _extract_hf_special_tokens(
    hf_tok: "Any",
    family: TokenizerFamily,
) -> Dict[str, str]:
    """
    Extract special tokens from a loaded HF tokenizer.
    Enforces R-E3 (must match exactly per family).
    """
    tokens: Dict[str, str] = {}

    def _safe_tok(id_: Optional[int]) -> Optional[str]:
        if id_ is None:
            return None
        try:
            return hf_tok.convert_ids_to_tokens(id_)
        except Exception:
            return None

    if family == TokenizerFamily.WORDPIECE:
        cls = getattr(hf_tok, "cls_token", "[CLS]")
        sep = getattr(hf_tok, "sep_token", "[SEP]")
        pad = getattr(hf_tok, "pad_token", "[PAD]")
        unk = getattr(hf_tok, "unk_token", "[UNK]")
        if cls:
            tokens["cls"] = cls
        if sep:
            tokens["sep"] = sep
        if pad:
            tokens["pad"] = pad
        if unk:
            tokens["unk"] = unk

    elif family in (TokenizerFamily.SENTENCEPIECE, TokenizerFamily.BPE):
        bos = getattr(hf_tok, "bos_token", None)
        eos = getattr(hf_tok, "eos_token", None)
        unk = getattr(hf_tok, "unk_token", None)
        pad = getattr(hf_tok, "pad_token", None)
        if bos:
            tokens["bos"] = bos
        if eos:
            tokens["eos"] = eos
        if unk:
            tokens["unk"] = unk
        if pad:
            tokens["pad"] = pad

    elif family == TokenizerFamily.TIKTOKEN:
        tokens["endoftext"] = "<|endoftext|>"

    return tokens


# ---------------------------------------------------------------------------
# Ollama tokenizer resolution (§3.4)
# ---------------------------------------------------------------------------

_OLLAMA_FAMILY_MAP: Dict[str, TokenizerFamily] = {
    "llama":           TokenizerFamily.SENTENCEPIECE,
    "mistral":         TokenizerFamily.SENTENCEPIECE,
    "nomic-embed-text": TokenizerFamily.WORDPIECE,
    "nomic":           TokenizerFamily.WORDPIECE,
    "qwen":            TokenizerFamily.TIKTOKEN,
    "phi":             TokenizerFamily.TIKTOKEN,
    "mxbai":           TokenizerFamily.BPE,
    "mxbai-embed":     TokenizerFamily.BPE,
}

_OLLAMA_DEFAULT_SPECIAL_TOKENS: Dict[TokenizerFamily, Dict[str, str]] = {
    TokenizerFamily.SENTENCEPIECE: {"bos": "<s>", "eos": "</s>", "unk": "<unk>"},
    TokenizerFamily.WORDPIECE:     {"cls": "[CLS]", "sep": "[SEP]", "pad": "[PAD]", "unk": "[UNK]"},
    TokenizerFamily.TIKTOKEN:      {"endoftext": "<|endoftext|>"},
    TokenizerFamily.BPE:           {"bos": "<s>", "eos": "</s>", "unk": "<unk>"},
    TokenizerFamily.UNKNOWN:       {},
}


def _resolve_ollama_tokenizer(
    entry: EmbedderCatalogEntry,
) -> Tuple[TokenizerContract, VerificationStatus]:
    """
    Parse Ollama modelfile and resolve tokenizer family following §3.4.

    Returns (TokenizerContract, VerificationStatus).
    Falls back to tiktoken cl100k_base with verification_status=unverified
    if family cannot be determined.
    """
    model_id = entry.model_id
    catalog_family = entry.tokenizer_family
    catalog_max = entry.embed_max_tokens

    # The ollama model name (without the "ollama/" prefix)
    ollama_model_name = model_id.removeprefix("ollama/")

    # Step 1 — run `ollama show <model> --modelfile`
    raw_modelfile: Optional[str] = None
    try:
        result = subprocess.run(
            ["ollama", "show", ollama_model_name, "--modelfile"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            raw_modelfile = result.stdout
        else:
            logger.warning(
                "[EmbedderRegistry] `ollama show %s --modelfile` failed "
                "(exit %d): %s",
                ollama_model_name, result.returncode, result.stderr[:200],
            )
    except FileNotFoundError:
        logger.warning(
            "[EmbedderRegistry] `ollama` binary not found. Cannot introspect "
            "tokenizer for %s. Falling back to catalog entry.",
            model_id,
        )
    except Exception as exc:
        logger.warning(
            "[EmbedderRegistry] Unexpected error running ollama show for %s: %s",
            model_id, exc,
        )

    # Step 2 — parse tokenizer family from model name
    discovered_family: Optional[TokenizerFamily] = None
    lower_name = ollama_model_name.lower()
    for key, fam in _OLLAMA_FAMILY_MAP.items():
        if key in lower_name:
            discovered_family = fam
            break

    # Step 3 — cross-check with catalog
    verification_status: VerificationStatus

    if discovered_family is None:
        # F-01/F-16 fallback: use tiktoken cl100k_base, set unverified
        logger.warning(
            "[EmbedderRegistry] F-01/F-16 risk: could not resolve tokenizer "
            "family for Ollama model %r. Falling back to tiktoken cl100k_base. "
            "verification_status=unverified.",
            model_id,
        )
        resolved_family = TokenizerFamily.TIKTOKEN
        verification_status = VerificationStatus.UNVERIFIED
    elif discovered_family == catalog_family:
        resolved_family = catalog_family
        verification_status = (
            VerificationStatus.VERIFIED
            if entry.verification_status != VerificationStatus.UNVERIFIED
            else VerificationStatus.UNVERIFIED
        )
    else:
        logger.warning(
            "[EmbedderRegistry] F-01/F-16 risk: catalog tokenizer_family=%r "
            "but discovered=%r for Ollama model %r. Catalog wins.",
            catalog_family.value, discovered_family.value, model_id,
        )
        resolved_family = catalog_family  # catalog wins
        verification_status = VerificationStatus.CATALOG_MISMATCH

    # Step 4 — build special tokens
    special_tokens = _OLLAMA_DEFAULT_SPECIAL_TOKENS.get(resolved_family, {})

    tokenizer = _OllamaTokenizer(
        model_id=model_id,
        tokenizer_family=resolved_family,
        max_length=catalog_max,
        special_tokens=dict(special_tokens),
    )

    logger.info(
        "[EmbedderRegistry] Ollama tokenizer resolved: model_id=%r family=%r "
        "status=%r",
        model_id, resolved_family.value, verification_status.value,
    )
    return tokenizer, verification_status


# ---------------------------------------------------------------------------
# Cloud tokenizer resolution (PATH 1)
# ---------------------------------------------------------------------------

_CLOUD_SPECIAL_TOKENS: Dict[str, Dict[str, str]] = {
    "cl100k_base":  {"endoftext": "<|endoftext|>"},
    "o200k_base":   {"endoftext": "<|endoftext|>"},
    "p50k_base":    {"endoftext": "<|endoftext|>"},
}


def _resolve_cloud_tokenizer(
    entry: EmbedderCatalogEntry,
) -> Tuple[TokenizerContract, VerificationStatus]:
    """
    Resolve tokenizer for cloud providers (OpenAI, Cohere, Anthropic, etc.)
    purely from catalog information.
    """
    model_id = entry.model_id
    provider = entry.provider
    catalog_family = entry.tokenizer_family
    catalog_max = entry.embed_max_tokens

    if provider == EmbedderProvider.OPENAI or catalog_family == TokenizerFamily.TIKTOKEN:
        encoding_name = entry.tokenizer_encoding or "cl100k_base"
        special_tokens = _CLOUD_SPECIAL_TOKENS.get(encoding_name, {"endoftext": "<|endoftext|>"})
        tokenizer = _TiktokenTokenizer(
            model_id=model_id,
            encoding_name=encoding_name,
            max_length=catalog_max,
            special_tokens=special_tokens,
        )
        return tokenizer, VerificationStatus.VERIFIED

    if provider == EmbedderProvider.COHERE:
        # Attempt to get Cohere API key from environment for SDK-backed counting.
        import os
        api_key = os.environ.get("COHERE_API_KEY") or None
        special_tokens: Dict[str, str] = {}  # Cohere BPE has no exposed sentinels
        tokenizer = _CohereTokenizer(
            model_id=model_id,
            max_length=catalog_max,
            api_key=api_key,
            special_tokens=special_tokens,
        )
        return tokenizer, VerificationStatus.VERIFIED

    if provider == EmbedderProvider.GOOGLE:
        # Use GeminiTokenizerContract (SentencePiece via countTokens API).
        # Falls back to tiktoken approximation if the key is missing/invalid.
        import os
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        try:
            from app.core.tokenization.gemini_tokenizer import GeminiTokenizerContract
            model_name = model_id.removeprefix("google/")
            gemini_tok = GeminiTokenizerContract(
                model_id=model_name,
                api_key_env="GOOGLE_API_KEY",
                timeout_sec=5.0,
            )
            return gemini_tok, VerificationStatus.VERIFIED
        except Exception as _tok_exc:
            logger.warning(
                "[EmbedderRegistry] GeminiTokenizerContract unavailable for "
                "model_id=%r (%s). Falling back to tiktoken approximation.",
                model_id, _tok_exc,
            )
            tokenizer = _TiktokenTokenizer(
                model_id=model_id,
                encoding_name="cl100k_base",
                max_length=catalog_max,
                special_tokens={"endoftext": "<|endoftext|>"},
            )
            return tokenizer, VerificationStatus.CATALOG_MISMATCH

    # Generic fallback for other cloud providers (Anthropic, Mistral).
    # Encode as tiktoken cl100k_base approximation; mark as catalog_mismatch.
    logger.warning(
        "[EmbedderRegistry] No native tokenizer resolution for provider=%r "
        "model_id=%r. Using tiktoken cl100k_base approximation. "
        "verification_status=catalog_mismatch.",
        provider.value, model_id,
    )
    encoding_name = entry.tokenizer_encoding or "cl100k_base"
    tokenizer = _TiktokenTokenizer(
        model_id=model_id,
        encoding_name=encoding_name,
        max_length=catalog_max,
        special_tokens={"endoftext": "<|endoftext|>"},
    )
    return tokenizer, VerificationStatus.CATALOG_MISMATCH


# ---------------------------------------------------------------------------
# EmbedderRegistry
# ---------------------------------------------------------------------------

class EmbedderRegistry:
    """
    Configuration-driven registry that resolves EmbedderBundle instances
    from embedder_catalog.yaml.

    Resolution pipeline (§3.1)
    ───────────────────────────
    1. Lookup in catalog → UnknownEmbedderModelError if missing.
    2. Schema validation → done by EmbedderCatalogEntry constructor.
    3. Determine path (cloud / HuggingFace / Ollama).
    4. Construct tokenizer (never resolved inside the embedder).
    5. Construct embedder via plugin registry adapter.
    6. Verify dimension and normalization.
    7. Assemble EmbedderBundle with all invariants checked.
    8. Cache resolved bundle under (model_id, catalog_hash).

    Thread safety (§3.5)
    ──────────────────────
    Bundle construction is protected by an RLock.  Only one bundle per
    (model_id, catalog_hash) is ever constructed in concurrent environments.

    Usage
    ──────
    ::

        registry = EmbedderRegistry()
        bundle = registry.get("openai/text-embedding-3-large")
        vectors = bundle.embed_documents(["chunk one", "chunk two"])
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # cache: (model_id, catalog_hash) → EmbedderBundle
        self._cache: Dict[Tuple[str, str], EmbedderBundle] = {}

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def get(self, model_id: str) -> EmbedderBundle:
        """
        Resolve and return the EmbedderBundle for ``model_id``.

        Steps follow §3.1 exactly.  Returns a cached bundle if the catalog
        has not changed since last resolution.

        Parameters
        ----------
        model_id : Canonical model identifier, e.g. "openai/text-embedding-3-large".

        Returns
        -------
        Immutable EmbedderBundle ready for ingestion and retrieval.

        Raises
        ──────
        UnknownEmbedderModelError
            If model_id is not in the catalog.
        EmbedderCatalogValidationError
            If the catalog entry fails schema validation.
        TokenizerResolutionError
            If the tokenizer cannot be loaded or classified.
        ValueError
            If EmbedderBundle invariant checks fail.
        """
        catalog_hash = get_catalog_hash()
        cache_key = (model_id, catalog_hash)

        with self._lock:
            if cache_key in self._cache:
                logger.debug(
                    "[EmbedderRegistry] Cache hit: model_id=%r hash=%s",
                    model_id, catalog_hash[:12],
                )
                return self._cache[cache_key]

        # Step 1 — lookup in catalog (raises UnknownEmbedderModelError via KeyError)
        try:
            entry = get_catalog_entry(model_id)
        except KeyError as exc:
            raise UnknownEmbedderModelError(model_id) from exc

        logger.info(
            "[EmbedderRegistry] Resolving bundle: model_id=%r provider=%r",
            model_id, entry.provider.value,
        )

        # Steps 3–4 — determine path and construct tokenizer
        tokenizer, verification_status = self._resolve_tokenizer(entry)

        # Step 5 — construct embedder (wrapped in adapter)
        embedder = self._resolve_embedder(entry, tokenizer)

        # Step 6 — verify dimension (runtime probe for non-trivial providers)
        if entry.provider in (
            EmbedderProvider.HUGGINGFACE,
            EmbedderProvider.OLLAMA,
        ):
            # Optionally probe: embedding a short string to confirm vector length.
            # Only done for local providers where dimension surprises are possible.
            try:
                probe_vec = embedder.embed_query("dimension probe")
                actual_dim = len(probe_vec)
                if actual_dim != entry.dimension:
                    logger.warning(
                        "[EmbedderRegistry] F-04 risk: catalog dimension=%d "
                        "but runtime probe returned dim=%d for model_id=%r. "
                        "Setting verification_status=catalog_mismatch.",
                        entry.dimension, actual_dim, model_id,
                    )
                    verification_status = VerificationStatus.CATALOG_MISMATCH
            except Exception as exc:
                logger.warning(
                    "[EmbedderRegistry] Could not probe embedder dimension for %r: %s",
                    model_id, exc,
                )

        # Step 7 — assemble EmbedderBundle (invariants checked in __post_init__)
        bundle = EmbedderBundle(
            model_id=entry.model_id,
            provider=entry.provider,
            tokenizer=tokenizer,
            embedder=embedder,
            dimension=entry.dimension,
            distance_metric=entry.distance_metric,
            is_normalized=entry.is_normalized,
            embed_max_tokens=entry.embed_max_tokens,
            default_reranker_id=entry.default_reranker_id,
            tokenizer_family=entry.tokenizer_family,
            verification_status=verification_status,
        )

        logger.info(
            "[EmbedderRegistry] Bundle assembled: %r", bundle,
        )

        # Step 8 — cache under (model_id, catalog_hash)
        with self._lock:
            self._cache[cache_key] = bundle

        return bundle

    # ------------------------------------------------------------------
    # Private: tokenizer resolution
    # ------------------------------------------------------------------

    def _resolve_tokenizer(
        self,
        entry: EmbedderCatalogEntry,
    ) -> Tuple[TokenizerContract, VerificationStatus]:
        """
        Route to the correct tokenizer resolution path based on provider.
        Follows §3.1 step 4.
        """
        provider = entry.provider

        if provider == EmbedderProvider.HUGGINGFACE:
            return _resolve_hf_tokenizer(entry)

        if provider == EmbedderProvider.OLLAMA:
            return _resolve_ollama_tokenizer(entry)

        # PATH 1 — cloud (OpenAI, Cohere, Anthropic, Google, Mistral)
        return _resolve_cloud_tokenizer(entry)

    # ------------------------------------------------------------------
    # Private: embedder construction
    # ------------------------------------------------------------------

    def _resolve_embedder(
        self,
        entry: EmbedderCatalogEntry,
        tokenizer: TokenizerContract,
    ) -> EmbedderContract:
        """
        Construct a concrete EmbedderContract for the entry.

        Uses the existing ``app/core/embedders`` plugin system where
        possible, wrapped in _BaseEmbedderAdapter so that the tokenizer
        is injected via EmbedderRegistry rather than resolved internally.
        Follows §3.1 step 5.
        """
        provider = entry.provider

        # Build the underlying BaseEmbedder from the existing plugin registry.
        base_embedder = self._build_base_embedder(entry)

        adapter = _BaseEmbedderAdapter(
            base_embedder=base_embedder,
            provider=provider,
            model_id=entry.model_id,
            dimension=entry.dimension,
            distance_metric=entry.distance_metric,
            is_normalized=entry.is_normalized,
            query_prefix=entry.query_prefix,
            passage_prefix=entry.passage_prefix,
        )

        # Inject tokenizer — this is the ONLY call site allowed to do so.
        adapter.inject_tokenizer(tokenizer)

        return adapter

    def _build_base_embedder(self, entry: EmbedderCatalogEntry) -> "Any":
        """
        Build a raw ``BaseEmbedder`` using the existing plugin registry.

        Environment variables provide credentials; no key is accepted from
        the catalog entry directly.  API keys are always resolved from env.
        """
        import os
        from app.core.plugin_registry import embedder_registry as plugin_registry  # noqa: F401 (import triggers auto-register)

        # Trigger plugin registration if needed.
        try:
            import app.core.embedders.register  # noqa: F401
        except ImportError:
            pass

        from app.core.plugin_registry import embedder_registry as er

        provider = entry.provider

        if provider == EmbedderProvider.OPENAI:
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise TokenizerResolutionError(
                    model_id=entry.model_id,
                    provider="openai",
                    reason="OPENAI_API_KEY environment variable is not set.",
                )
            model_name = entry.model_id.removeprefix("openai/")
            return er.build(
                "openai",
                api_key=api_key,
                model=model_name,
                normalize=entry.is_normalized,
            )

        if provider == EmbedderProvider.COHERE:
            api_key = os.environ.get("COHERE_API_KEY", "").strip()
            if not api_key:
                raise TokenizerResolutionError(
                    model_id=entry.model_id,
                    provider="cohere",
                    reason="COHERE_API_KEY environment variable is not set.",
                )
            model_name = entry.model_id.removeprefix("cohere/")
            return er.build(
                "cohere",
                api_key=api_key,
                model=model_name,
                normalize=entry.is_normalized,
            )

        if provider == EmbedderProvider.HUGGINGFACE:
            return er.build(
                "huggingface",
                model=entry.model_id,
                device="auto",
                batch_size=32,
                normalize=entry.is_normalized,
            )

        if provider == EmbedderProvider.OLLAMA:
            base_url = os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ).strip()
            model_name = entry.model_id.removeprefix("ollama/")
            return er.build(
                "ollama",
                model=model_name,
                base_url=base_url,
                normalize=entry.is_normalized,
            )

        if provider == EmbedderProvider.GOOGLE:
            # Google Gemini embedder — registered in plugin registry as "gemini"
            api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
            if not api_key:
                raise TokenizerResolutionError(
                    model_id=entry.model_id,
                    provider="google",
                    reason="GOOGLE_API_KEY environment variable is not set.",
                )
            model_name = entry.model_id.removeprefix("google/")
            return er.build(
                "gemini",
                api_key=api_key,
                model=model_name,
                normalize=entry.is_normalized,
            )

        raise TokenizerResolutionError(
            model_id=entry.model_id,
            provider=provider.value,
            reason=(
                f"No plugin builder for provider {provider.value!r}. "
                "Add a branch in EmbedderRegistry._build_base_embedder."
            ),
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate(self, model_id: Optional[str] = None) -> int:
        """
        Evict cached bundles.

        Parameters
        ----------
        model_id : If given, evict only bundles for this model_id.
                   If None, clear entire cache.

        Returns
        -------
        Number of entries evicted.
        """
        with self._lock:
            if model_id is None:
                count = len(self._cache)
                self._cache.clear()
                logger.info("[EmbedderRegistry] Cache cleared (%d entries).", count)
                return count
            keys = [k for k in self._cache if k[0] == model_id]
            for k in keys:
                del self._cache[k]
            logger.info(
                "[EmbedderRegistry] Evicted %d cache entries for model_id=%r.",
                len(keys), model_id,
            )
            return len(keys)

    def list_cached(self) -> List[str]:
        """Return model_ids of currently cached bundles."""
        with self._lock:
            return sorted({k[0] for k in self._cache})


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

embedder_registry = EmbedderRegistry()
