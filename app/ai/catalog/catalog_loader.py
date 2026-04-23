# =============================================================================
# app/ai/catalog/catalog_loader.py
#
# EmbedderCatalogEntry + load_catalog() — Phase 1, Module 3
#
# Loads, validates, and indexes embedder_catalog.yaml at import time.
# Provides a typed in-memory representation of every catalog entry.
#
# Failure modes addressed (via schema validation):
#   F-01  tokenizer_family enum constraint prevents vocabulary mismatches
#   F-02  embed_max_tokens > 0 constraint prevents zero/negative limits
#   F-04  dimension > 0 constraint prevents dimension=0 indexes
#   F-05  model_id uniqueness prevents silent entry override
#   F-06  is_normalized type validation prevents ambiguous normalization
#   F-08  distance_metric enum prevents unsupported metrics
#   F-16  schema_version tracks catalog compatibility across migrations
#
# All validation is EAGER — runs at load time, never at embed/query time.
# =============================================================================

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError as _yaml_err:
    raise ImportError(
        "PyYAML is required for embedder catalog loading. "
        "Install it with: pip install pyyaml"
    ) from _yaml_err

from app.ai.contracts.tokenizer_contract import TokenizerFamily
from app.ai.contracts.embedder_contract import (
    EmbedderProvider,
    DistanceMetric,
    VerificationStatus,
)

logger = logging.getLogger(__name__)

# Absolute path to the catalog file bundled with this package.
_CATALOG_PATH = Path(__file__).parent / "embedder_catalog.yaml"


# ---------------------------------------------------------------------------
# EmbedderCatalogValidationError
# ---------------------------------------------------------------------------

class EmbedderCatalogValidationError(ValueError):
    """
    Raised when a catalog entry fails schema or semantic validation.

    Raised eagerly at load time — never deferred to embedding time.

    Fields
    ──────
    model_id  : The entry's model_id (or "<unknown>" if absent).
    details   : Dict mapping field names to error descriptions.
    """

    def __init__(self, model_id: str, details: Dict[str, Any]) -> None:
        self.model_id = model_id
        self.details = details
        detail_str = "; ".join(f"{k}: {v}" for k, v in details.items())
        super().__init__(
            f"[EmbedderCatalogValidationError] model_id={model_id!r} — {detail_str}"
        )


# ---------------------------------------------------------------------------
# EmbedderCatalogEntry
# ---------------------------------------------------------------------------

class EmbedderCatalogEntry:
    """
    Fully validated, typed representation of one embedder_catalog.yaml entry.

    All fields are read-only after construction.  Construction raises
    EmbedderCatalogValidationError on any violation.

    This class is the authoritative source of configuration consumed by
    EmbedderRegistry when constructing an EmbedderBundle.
    """

    __slots__ = (
        "model_id",
        "provider",
        "tokenizer_class",
        "tokenizer_encoding",
        "tokenizer_source",
        "tokenizer_family",
        "dimension",
        "distance_metric",
        "is_normalized",
        "embed_max_tokens",
        "default_reranker_id",
        "query_prefix",
        "passage_prefix",
        "verification_status",
        "embedder_type",
        "lang_support",
        "notes",
    )

    # ------------------------------------------------------------------
    # Allowed enum values (mirrors contract enums; keeps loader independent
    # of the ABC hierarchy so there are no circular imports at load time).
    # ------------------------------------------------------------------
    _VALID_PROVIDERS: frozenset = frozenset(e.value for e in EmbedderProvider)
    _VALID_FAMILIES:  frozenset = frozenset(e.value for e in TokenizerFamily)
    _VALID_METRICS:   frozenset = frozenset(e.value for e in DistanceMetric)
    _VALID_STATUSES:  frozenset = frozenset(e.value for e in VerificationStatus)
    _VALID_TOKENIZER_CLASSES: frozenset = frozenset({
        "tiktoken",
        "transformers.AutoTokenizer",
        "ollama_modelfile",
        "cohere_sdk",
        "google-generativeai:countTokens",  # Gemini token-count API
    })

    def __init__(self, raw: Dict[str, Any]) -> None:
        errors: Dict[str, str] = {}
        model_id_raw = raw.get("model_id", "<unknown>")

        # ── model_id ─────────────────────────────────────────────────
        model_id = raw.get("model_id")
        if not model_id or not isinstance(model_id, str) or not model_id.strip():
            errors["model_id"] = "must be a non-empty string"
        else:
            model_id = model_id.strip()

        # ── provider ─────────────────────────────────────────────────
        provider_raw = raw.get("provider")
        if provider_raw not in self._VALID_PROVIDERS:
            errors["provider"] = (
                f"must be one of {sorted(self._VALID_PROVIDERS)}, "
                f"got {provider_raw!r}"
            )

        # ── tokenizer_class ──────────────────────────────────────────
        tc = raw.get("tokenizer_class")
        if tc not in self._VALID_TOKENIZER_CLASSES:
            errors["tokenizer_class"] = (
                f"must be one of {sorted(self._VALID_TOKENIZER_CLASSES)}, "
                f"got {tc!r}"
            )

        # ── tokenizer_encoding (required if tiktoken) ─────────────────
        te = raw.get("tokenizer_encoding")
        if tc == "tiktoken" and not te:
            errors["tokenizer_encoding"] = (
                "required when tokenizer_class='tiktoken' (e.g. 'cl100k_base')"
            )

        # ── tokenizer_source (required for hf/ollama/cohere; optional for google) ──
        ts_raw = raw.get("tokenizer_source")
        if tc in ("transformers.AutoTokenizer", "ollama_modelfile", "cohere_sdk"):
            if not ts_raw:
                errors["tokenizer_source"] = (
                    f"required when tokenizer_class={tc!r}"
                )
        # google-generativeai:countTokens uses tokenizer_source as a doc reference,
        # not a required model-hub path — no validation needed.

        # ── tokenizer_family ─────────────────────────────────────────
        fam = raw.get("tokenizer_family")
        if fam not in self._VALID_FAMILIES:
            errors["tokenizer_family"] = (
                f"must be one of {sorted(self._VALID_FAMILIES)}, got {fam!r}"
            )

        # Cross-check: tiktoken class must map to tiktoken family
        if tc == "tiktoken" and fam not in ("tiktoken", "bpe"):
            errors["tokenizer_family__class_mismatch"] = (
                f"tokenizer_class='tiktoken' requires family 'tiktoken' or 'bpe', "
                f"got {fam!r}"
            )

        # ── dimension ────────────────────────────────────────────────
        dim = raw.get("dimension")
        if not isinstance(dim, int) or dim <= 0:
            errors["dimension"] = f"must be a positive integer, got {dim!r}"

        # ── distance_metric ──────────────────────────────────────────
        metric = raw.get("distance_metric")
        if metric not in self._VALID_METRICS:
            errors["distance_metric"] = (
                f"must be one of {sorted(self._VALID_METRICS)}, got {metric!r}"
            )

        # ── is_normalized ────────────────────────────────────────────
        is_norm = raw.get("is_normalized")
        if not isinstance(is_norm, bool):
            errors["is_normalized"] = f"must be a boolean, got {is_norm!r}"

        # ── embed_max_tokens ─────────────────────────────────────────
        emt = raw.get("embed_max_tokens")
        if not isinstance(emt, int) or emt <= 0:
            errors["embed_max_tokens"] = f"must be a positive integer, got {emt!r}"

        # ── query_prefix / passage_prefix ────────────────────────────
        qp = raw.get("query_prefix")
        pp = raw.get("passage_prefix")
        if qp is None:
            errors["query_prefix"] = "must be present (use empty string '' if no prefix)"
        if pp is None:
            errors["passage_prefix"] = "must be present (use empty string '' if no prefix)"

        # ── verification_status ──────────────────────────────────────
        vs = raw.get("verification_status", "unverified")
        if vs not in self._VALID_STATUSES:
            errors["verification_status"] = (
                f"must be one of {sorted(self._VALID_STATUSES)}, got {vs!r}"
            )

        # ── Fail fast on any schema error ────────────────────────────
        if errors:
            raise EmbedderCatalogValidationError(
                model_id=str(model_id_raw), details=errors
            )

        # ── Assign validated fields ───────────────────────────────────
        object.__setattr__(self, "model_id",            model_id)
        object.__setattr__(self, "provider",            EmbedderProvider(provider_raw))
        object.__setattr__(self, "tokenizer_class",     tc)
        object.__setattr__(self, "tokenizer_encoding",  te or None)
        object.__setattr__(self, "tokenizer_source",    ts_raw or None)
        object.__setattr__(self, "tokenizer_family",    TokenizerFamily(fam))
        object.__setattr__(self, "dimension",           int(dim))
        object.__setattr__(self, "distance_metric",     DistanceMetric(metric))
        object.__setattr__(self, "is_normalized",       bool(is_norm))
        object.__setattr__(self, "embed_max_tokens",    int(emt))
        object.__setattr__(self, "default_reranker_id", raw.get("default_reranker_id") or None)
        object.__setattr__(self, "query_prefix",        str(qp) if qp is not None else "")
        object.__setattr__(self, "passage_prefix",      str(pp) if pp is not None else "")
        object.__setattr__(self, "verification_status", VerificationStatus(vs))
        object.__setattr__(self, "embedder_type",       raw.get("embedder_type") or None)
        object.__setattr__(self, "lang_support",        raw.get("lang_support") or "en")
        object.__setattr__(self, "notes",               raw.get("notes") or None)

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError(
            f"EmbedderCatalogEntry is immutable — cannot set {key!r}"
        )

    def __repr__(self) -> str:
        return (
            f"EmbedderCatalogEntry("
            f"model_id={self.model_id!r}, "
            f"provider={self.provider.value!r}, "
            f"dim={self.dimension}, "
            f"family={self.tokenizer_family.value!r}, "
            f"status={self.verification_status.value!r})"
        )


# ---------------------------------------------------------------------------
# _LoadedCatalog — internal cache
# ---------------------------------------------------------------------------

class _LoadedCatalog:
    """
    Internal container for a fully loaded and validated catalog.
    Holds:
      - entries      : dict from model_id → EmbedderCatalogEntry
      - catalog_hash : SHA-256 of the raw YAML bytes
      - schema_version : from YAML top-level field
    """
    __slots__ = ("entries", "catalog_hash", "schema_version")

    def __init__(
        self,
        entries: Dict[str, EmbedderCatalogEntry],
        catalog_hash: str,
        schema_version: str,
    ) -> None:
        self.entries = entries
        self.catalog_hash = catalog_hash
        self.schema_version = schema_version


_cache: Optional[_LoadedCatalog] = None
_catalog_file: Path = _CATALOG_PATH


def _compute_file_hash(path: Path) -> str:
    """SHA-256 of the raw YAML bytes — used as cache key."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def set_catalog_path(path: "str | Path") -> None:
    """
    Override the default catalog file path.
    Call this before the first ``load_catalog()`` invocation, e.g. in tests.
    """
    global _catalog_file, _cache
    _catalog_file = Path(path)
    _cache = None  # invalidate cache when path changes


def load_catalog(*, force_reload: bool = False) -> _LoadedCatalog:
    """
    Load, validate, and cache the embedder catalog.

    Caching strategy
    ─────────────────
    The catalog YAML is hashed on every call.  If the hash matches the
    currently cached catalog, the in-memory representation is returned
    immediately without re-parsing.  If the hash differs (because the
    YAML was edited on disk), the cache is cleared and the file is
    re-parsed and re-validated.

    Parameters
    ----------
    force_reload : If True, bypass the hash check and always reload.

    Returns
    -------
    _LoadedCatalog with ``entries`` dict and ``catalog_hash``.

    Raises
    ------
    EmbedderCatalogValidationError
        If any entry fails schema validation.
    FileNotFoundError
        If the catalog YAML file does not exist.
    """
    global _cache

    path = _catalog_file
    if not path.exists():
        raise FileNotFoundError(
            f"embedder_catalog.yaml not found at {path}. "
            "Ensure app/ai/catalog/embedder_catalog.yaml is present."
        )

    current_hash = _compute_file_hash(path)

    if not force_reload and _cache is not None and _cache.catalog_hash == current_hash:
        return _cache

    logger.info(
        "[CatalogLoader] Loading embedder_catalog.yaml (hash=%s)", current_hash[:12]
    )

    raw_bytes = path.read_bytes()
    try:
        raw_yaml = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # YAML was saved with a Windows code-page (e.g. cp1252).
        # Decode permissively and replace any unparseable bytes with the
        # closest Unicode equivalent so the YAML can still be loaded.
        # The offending characters are cosmetic (notes/dashes) and do
        # not affect validated required fields.
        raw_yaml = raw_bytes.decode("cp1252")
        logger.warning(
            "[CatalogLoader] embedder_catalog.yaml is not valid UTF-8 — "
            "decoded as cp1252. Re-save the file as UTF-8 to remove this warning."
        )
    data = yaml.safe_load(raw_yaml)

    if not isinstance(data, dict):
        raise EmbedderCatalogValidationError(
            "<root>", {"structure": "catalog YAML must be a mapping at the top level"}
        )

    schema_version = str(data.get("schema_version", "unknown"))
    raw_entries = data.get("embedders", [])

    if not isinstance(raw_entries, list):
        raise EmbedderCatalogValidationError(
            "<root>",
            {"embedders": "must be a list of entry mappings"},
        )

    entries: Dict[str, EmbedderCatalogEntry] = {}
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise EmbedderCatalogValidationError(
                "<unknown>", {"entry": "each embedder entry must be a YAML mapping"}
            )
        entry = EmbedderCatalogEntry(raw)

        # model_id uniqueness check (F-05 — prevents silent space drift)
        if entry.model_id in entries:
            raise EmbedderCatalogValidationError(
                entry.model_id,
                {
                    "model_id": (
                        f"duplicate model_id {entry.model_id!r} detected in catalog. "
                        "Duplicate entries risk F-05 (embedding space drift)."
                    )
                },
            )

        entries[entry.model_id] = entry
        logger.debug(
            "[CatalogLoader] Registered entry: model_id=%r status=%r",
            entry.model_id,
            entry.verification_status.value,
        )

    catalog = _LoadedCatalog(
        entries=entries,
        catalog_hash=current_hash,
        schema_version=schema_version,
    )
    _cache = catalog

    logger.info(
        "[CatalogLoader] Loaded %d entries | schema_version=%s | hash=%s",
        len(entries),
        schema_version,
        current_hash[:12],
    )
    return catalog


def get_catalog_entry(model_id: str) -> EmbedderCatalogEntry:
    """
    Return the validated catalog entry for ``model_id``.

    Raises
    ──────
    KeyError
        If no entry with the given model_id exists in the catalog.
    """
    catalog = load_catalog()
    entry = catalog.entries.get(model_id)
    if entry is None:
        available = sorted(catalog.entries.keys())
        raise KeyError(
            f"[CatalogLoader] model_id={model_id!r} not found in catalog. "
            f"Available models: {available}"
        )
    return entry


def list_model_ids() -> List[str]:
    """Return all model IDs registered in the catalog, sorted."""
    return sorted(load_catalog().entries.keys())


def get_catalog_hash() -> str:
    """Return the SHA-256 hash of the currently loaded catalog YAML."""
    return load_catalog().catalog_hash
