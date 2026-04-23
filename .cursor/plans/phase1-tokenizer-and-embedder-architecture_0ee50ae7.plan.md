---
name: phase1-tokenizer-and-embedder-architecture
overview: Phase 1 design for a tokenizer-safe, config-driven embedder registry and validation layer for MarketingAdvantage_AI_v1 ingestion and RAG pipelines, ensuring zero tokenizer mismatch and embedding-space drift.
todos:
  - id: define-tokenizer-contract
    content: Define TokenizerContract ABC and tokenizer-family enum aligned with existing tokenization base, ensuring native binding and max-length semantics.
    status: completed
  - id: design-embedder-bundle-and-contract
    content: Design EmbedderContract and EmbedderBundle dataclass that bind tokenizers, dimensions, metrics, and normalization into a single immutable object.
    status: completed
  - id: spec-embedder-catalog-schema
    content: Specify embedder_catalog.yaml schema and populate entries for OpenAI, Cohere, HuggingFace, and Ollama models with verification_status semantics.
    status: completed
  - id: implement-embedder-registry
    content: Design EmbedderRegistry that resolves EmbedderBundle instances via the catalog, including HF and Ollama tokenizer resolution and caching.
    status: completed
  - id: build-validation-layer-and-migration-guard
    content: Design TokenizerValidator and ChunkSizerContract plus the embedding-space fingerprint guard to enforce re-index on tokenizer or embedder change.
    status: completed
  - id: todo-1776713548597-eszeagean
    content: |-
      You are now in BUILD MODE.
      Reference the Phase 1 architecture plan.
      Build in strict module order per §5.1:
        1. TokenizerContract → app/ai/contracts/tokenizer_contract.py
        2. EmbedderContract + EmbedderBundle → app/ai/contracts/embedder_contract.py
        3. embedder_catalog.yaml + loader → app/ai/catalog/embedder_catalog.yaml
        4. EmbedderRegistry → app/ai/registry/embedder_registry.py
        5. ChunkSizerContract + TokenizerValidator → app/ai/validation/

      Fix these issues before coding:
      - cohere tokenizer_family → "bpe" (not "wordpiece")
      - e5-mistral embed_max_tokens → 4096 (not 131072)
      - Add reranker_query_text() to EmbedderBundle

      Do NOT modify any existing file.
      Do NOT proceed to module N+1 without completing N.
    status: completed
isProject: false
---

# Phase 1 Tokenizer & Embedder Architecture

### Problem Breakdown
- **Goal**: Design Phase 1 architecture for an Enterprise RAG ingestion system that makes it structurally impossible to mix tokenizers or embedding spaces, and that centralizes all embedder/tokenizer configuration in `embedder_catalog.yaml`.
- **Scope**: New contracts (ABCs), an `EmbedderBundle` dataclass, a config schema and registry for embedders, and a validation layer – all wired into existing `ClientConfig`, `PipelineFactory`, `BaseEmbedder`, `BaseVectorDB`, and reranker contracts without changing their high-level responsibilities.
- **Tokenizer principle**: Each component (embedder, reranker, LLM) uses its own native tokenizer for its own purpose only; tokenizer resolution is centralized and propagated via `EmbedderBundle`, not re-resolved per stage.

### Issues & Risks
- **Current implicit tokenizer handling**: Chunking uses generic tokenization utilities while embedders resolve their own tokenizers internally; there is no global guarantee that chunk size, embedder max length, reranker max length, and LLM context rules align (risks F-01, F-02, F-13, F-14, F-15, F-16).
- **Embedding/vector DB coupling**: `PipelineFactory` probes embedder dimensions at runtime and passes them into `BaseVectorDB.ensure_collection`, but there is no persistent, versioned `EmbedderBundle` that ties together model ID, tokenizer family, dimension, and distance metric (risks F-04, F-05, F-08, F-09, F-10, F-16).
- **Reranker & LLM alignment**: Rerankers and LLMs are configured separately in `ClientConfig` and wired by `PipelineFactory` without explicit checks on score space compatibility, max input tokens, or tokenizer families (risks F-11, F-12, F-13, F-14, F-15).
- **Migration and drift**: Changing embedders or tokenizers today does not automatically force re-chunking or re-indexing, nor does it emit a hard failure when the index’s embedding-space identity changes (directly risks F-05, F-16, and R-C4).

### Structured Solution

#### 1. CONTRACTS DESIGN

##### 1.1 TokenizerContract (ABC)
- **Name**: `TokenizerContract`
- **Type**: Abstract Base Class (ABC), defined alongside existing tokenizer base in a new AI-focused contracts module (e.g., `app/ai/contracts.py`), conceptually aligned with but stricter than `app/core/tokenization/base.py`.
- **Fields / properties (with Python types)**:
  - `name: str` — human-readable identifier, e.g. `"openai/cl100k_base"`, `"hf/BAAI/bge-large-en-v1.5"`.
  - `model_id: str` — canonical model identifier this tokenizer is native to, e.g. `"openai/text-embedding-3-large"`, `"BAAI/bge-large-en-v1.5"`, `"Qwen/Qwen3-Embedding"`.
  - `tokenizer_family: Literal["tiktoken", "wordpiece", "sentencepiece", "bpe", "whitespace", "unknown"]` — family classification, used for invariants and failure-mode mapping.
  - `max_length: int` — maximum number of tokens supported by the embedding model’s tokenizer (from config.json, tokenizer.json, or vendor docs).
  - `special_tokens: Dict[str, str]` — mapping of semantic roles to concrete tokens, e.g. `{ "cls": "[CLS]", "sep": "[SEP]", "bos": "<s>", "eos": "</s>", "pad": "[PAD]", "unk": "[UNK]" }`, constrained per family.
  - `is_thread_safe: bool` — true if the implementation is safe for concurrent calls after construction (no shared mutable state).
  - `tokenizer_source: str` — source of truth string (`"tiktoken:cl100k_base"`, `"hf:config.json"`, `"ollama:modelfile"`).
- **Methods (signatures and types)**:
  - `tokenize(text: str, *, add_special_tokens: bool = False) -> List[int]`
    - Returns token IDs according to the underlying native tokenizer.
    - Raises `TokenizerResolutionError` if underlying tokenizer is not successfully loaded or bound.
  - `count_tokens(text: str, *, include_special_tokens: bool = False) -> int`
    - Returns the number of tokens that would be produced by `tokenize` under the same `include_special_tokens` flag.
    - Must satisfy: `count_tokens(text, include_special_tokens=x) == len(tokenize(text, add_special_tokens=x))` for all inputs.
  - `decode(token_ids: List[int]) -> str`
    - Converts token IDs back into unicode text using the exact inverse mapping of `tokenize` with `add_special_tokens=False`.
  - `measure_with_overhead(text: str, *, special_overhead: int) -> int`
    - Helper to calculate token count plus a caller-supplied overhead for special tokens; used by `ChunkSizerContract` and `TokenizerValidator`.
- **Invariants (non-negotiable)**:
  1. **Native binding only**: `model_id` must refer to the same model family the tokenizer was trained for; third-party or cross-model tokenizers are forbidden (enforces R-E1, R-E2; prevents F-01, F-16).
  2. **Family consistency**: `tokenizer_family` must match the tokenizer implementation discovered from `config.json` / `tokenizer.json` (HF) or modelfile parameters (Ollama); any mismatch sets `verification_status` in `EmbedderBundle` to a non-`"verified"` state and raises at build time (prevents F-01, F-16).
  3. **Max-length accuracy**: `max_length` must reflect the model’s documented `max_position_embeddings` or vendor context limit; no defaulting or auto-detection at runtime. Any failure to read max length must trigger `TokenizerResolutionError` (prevents F-02, F-13, F-14, F-15, F-16; enforces R-E4, R-C2, R-L3).
  4. **Special token fidelity**: `special_tokens` must contain all required sentinel tokens for the family (e.g., BERT `[CLS]`/`[SEP]`, RoBERTa `<s>`/`</s>`, GPT `<|endoftext|>`). The values must exactly match the underlying tokenizer’s definitions (prevents F-03, F-07; enforces R-E3).
  5. **Determinism and equality**: `count_tokens` must be a pure function of `text` and flags; the same input and flags must always yield the same count, and must equal the length of `tokenize` output (prevents F-02 and hidden truncation issues linked to miscounted tokens; enforces R-C3).
  6. **Thread-safety**: If `is_thread_safe` is `True`, concurrent use from multiple threads must not mutate internal state in a way that changes tokenization results; otherwise, internal guarding (e.g., per-thread instances) must be used by callers (avoids concurrency-induced drift in counts and overflow decisions; indirectly supports all F-01–F-03, F-13–F-16).
- **Failure modes prevented (primary)**:
  - F-01 (Vocabulary mismatch) — via native binding and tokenizer-family checks.
  - F-02 (Max length overflow) — via accurate `max_length` and count equivalence.
  - F-03 (Special token injection) — via enforced `special_tokens` mapping.
  - F-13 (Reranker input length mismatch) — by providing accurate token counts to `ChunkSizerContract` / validator.
  - F-14 (LLM context tokenizer mismatch) — by clearly separating tokenizer families and counts from LLM tokenizers.
  - F-15 (Lost in the middle) — by supplying correct max lengths to chunk sizing.
  - F-16 (Tokenizer migration without re-index) — by making tokenizer family and model ID explicit and versioned.

##### 1.2 EmbedderContract (ABC)
- **Name**: `EmbedderContract`
- **Type**: Abstract Base Class (ABC), conceptually layered on top of `app/core/embedders/base.BaseEmbedder` (can be implemented as a mixin or extended interface without breaking existing concrete implementations).
- **Fields / properties**:
  - `provider: Literal["openai", "cohere", "huggingface", "ollama", "anthropic", "google", "mistral"]` — must align with `EmbedderType` and catalog entry.
  - `model_id: str` — exact model identifier (e.g. `"openai/text-embedding-3-large"`, `"BAAI/bge-large-en-v1.5"`).
  - `bound_tokenizer: TokenizerContract` — read-only property; set only by the registry at bundle construction time.
  - `dimension: int` — embedding vector dimensionality; must match catalog and actual output length.
  - `distance_metric: Literal["cosine", "dotproduct", "euclidean"]` — metric intended for VectorDB indexing.
  - `is_normalized: bool` — `True` if outputs are L2-normalized (e.g., BGE default), `False` otherwise.
  - `query_prefix: str` — model-specific prefix applied to queries (E5/BGE/E5-Mistral patterns).
  - `passage_prefix: str` — model-specific prefix applied to passages/documents.
- **Methods**:
  - `embed_documents(texts: List[str]) -> List[List[float]]`
    - Applies `passage_prefix` (if non-empty) and `bound_tokenizer` when measuring chunk length, but relies on provider’s native embedding API to perform embedding.
    - Must assume that chunking has already ensured token counts are within `embed_max_tokens`.
  - `embed_query(text: str) -> List[float]`
    - Applies `query_prefix` (if non-empty) and uses `bound_tokenizer` only for measurement/validation, not for generating token IDs to send to cloud APIs.
- **Invariants**:
  1. **Tokenizers are injected, never resolved**: `bound_tokenizer` must be set by the registry from `EmbedderBundle` construction; any attempt by the concrete embedder implementation to call `AutoTokenizer.from_pretrained`, `tiktoken.get_encoding`, or similar must be considered a violation. This prevents each embedder from independently resolving tokenizers (prevents F-01, F-02, F-16; enforces MASTER CONTRACT and R-E1, R-E2).
  2. **Prefix discipline**: `embed_documents` must always apply `passage_prefix` if non-empty, and `embed_query` must always apply `query_prefix` if non-empty, in a deterministic way. The same prefix policy must be used at indexing and query time (prevents F-07, F-05; enforces R-R3).
  3. **Dimension correctness**: For any non-empty input, the length of vectors returned by `embed_documents` and `embed_query` must equal `dimension` and the catalog’s `dimension`. Any mismatch must raise an embedding-specific alignment error (prevents F-04, F-05, F-09).
  4. **Distance metric consistency**: `distance_metric` must match the VectorDB’s configured metric for the target collection; misalignment is treated as a blocking configuration error by `TokenizerValidator.validate_pipeline_alignment` (prevents F-08).
  5. **Normalization honesty**: If `is_normalized` is `True`, all output vectors must be norm-1 in L2 space; if `False`, no implicit normalization can be applied. The VectorDB adapter and scoring logic rely on this to interpret scores correctly (prevents F-06, F-11).
- **Failure modes prevented (primary)**:
  - F-01 (Vocabulary mismatch) — by forbidding internal tokenizer resolution and binding only via `EmbedderBundle`.
  - F-04 (Dimension mismatch) — by equating vector lengths with `dimension` and catalog entries.
  - F-05 (Embedding space drift) — by tying `model_id`, `dimension`, `provider`, and tokenizer family together in `EmbedderBundle`.
  - F-06 (Normalization mismatch) — via `is_normalized` invariant.
  - F-07 (Asymmetric encoding) — via enforced `query_prefix` / `passage_prefix` behaviour.
  - F-08 (Distance metric mismatch) — via explicit `distance_metric` contract and alignment checks.

##### 1.3 ChunkSizerContract (ABC)
- **Name**: `ChunkSizerContract`
- **Type**: Abstract Base Class (ABC), used by ingestion chunkers and validators.
- **Fields / properties**:
  - `embed_max_tokens: int` — maximum tokens allowed by embedder tokenizer (`EmbedderBundle.embed_max_tokens`).
  - `reranker_max_tokens: Optional[int]` — maximum tokens accepted by the chosen reranker; `None` if no reranker.
  - `query_reserve_tokens: int` — reserved budget for eventual query tokens (default ≈ 50, configurable per model or per deployment via catalog).
  - `special_overhead_tokens: int` — overhead to account for `[CLS]`, `[SEP]`, BOS/EOS tokens, etc. (min 4 per R-C2).
  - `headroom_pct: float` — proportional headroom to prevent hitting tail failure modes (default 0.10 per R-C2).
- **Methods**:
  - `safe_chunk_size() -> int`
    - Computes `chunk_max` using the specified formula:
      - `raw_max = min(embed_max_tokens, reranker_max_tokens or embed_max_tokens)`
      - `available = raw_max - query_reserve_tokens - special_overhead_tokens`
      - `chunk_max = floor(available * (1.0 - headroom_pct))`
    - Returns the integer `chunk_max` to be enforced across ingestion.
  - `validate_chunk(text: str, tokenizer: TokenizerContract) -> ValidationResult`
    - Uses `tokenizer.count_tokens(text, include_special_tokens=False)` and asserts that the result is `<= safe_chunk_size()`.
    - On violation, raises `TokenOverflowError` and returns a `ValidationResult` marked as failed with failure mode references.
- **Invariants**:
  1. **Token-based sizing only**: All chunk size calculations must be expressed in tokens measured by the embedder’s `TokenizerContract`, never in characters or words (enforces R-C1, R-C3; prevents F-02, F-13, F-14, F-15, F-16).
  2. **Formal chunk_max formula**: `safe_chunk_size()` must implement the exact formula `chunk_max = floor( min(embed_max, reranker_max) − query_reserve − special_overhead ) × (1 − headroom_pct)` with no hidden extra adjustments (enforces R-C2, R-R2; prevents F-02, F-13, F-15).
  3. **Embedder–reranker coupling**: If a reranker is configured, `safe_chunk_size()` must use the minimum of embedder and reranker max tokens; if no reranker, it must fallback cleanly to `embed_max_tokens` (prevents F-13, F-02).
  4. **Non-negative bounds**: If the computed `chunk_max` is less than or equal to zero, chunking is considered invalid and must fail fast at configuration time (prevents degenerate configurations that would silently drop all content; avoids F-13/F-15 style issues).
- **Failure modes prevented (primary)**:
  - F-02 (Max length overflow) — by computing and enforcing safe limits.
  - F-13 (Reranker input length mismatch) — by incorporating `reranker_max_tokens`.
  - F-14 (LLM context tokenizer mismatch) — indirectly, by making chunking independent of LLM tokenizer and bounded by embedder/reranker.
  - F-15 (Lost in the middle) — by enforcing headroom and conservative chunk sizes.
  - F-16 (Tokenizer migration without re-index) — chunk_max derives from embedder tokenizer; migration changes must trigger recompute and re-index guard.

##### 1.4 EmbedderBundle (dataclass)
- **Name**: `EmbedderBundle`
- **Type**: Dataclass (NOT an ABC); immutable once constructed.
- **Fields**:
  - `model_id: str` — canonical model identifier (e.g. `"openai/text-embedding-3-large"`).
  - `provider: Literal["openai", "cohere", "huggingface", "ollama", "anthropic", "google", "mistral"]`.
  - `tokenizer: TokenizerContract` — the already-resolved tokenizer instance bound to the model.
  - `embedder: EmbedderContract` — the already-constructed embedder instance bound to `tokenizer`.
  - `dimension: int` — output vector dimension.
  - `distance_metric: Literal["cosine", "dotproduct", "euclidean"]` — metric used for both indexing and query-time search.
  - `is_normalized: bool` — whether vectors are L2-normalized.
  - `embed_max_tokens: int` — maximum supported tokens for this embedder/tokenizer pair.
  - `default_reranker_id: Optional[str]` — logical ID of the default reranker (e.g. `"cohere/rerank-v3.0"`, `"bge-reranker-large"`).
  - `tokenizer_family: Literal["tiktoken", "wordpiece", "sentencepiece", "bpe", "whitespace", "unknown"]` — duplication of `tokenizer.tokenizer_family` for quick alignment checks.
  - `verification_status: Literal["verified", "catalog_mismatch", "unverified"]` — quality of tokenizer resolution.
- **Invariants**:
  1. **Single object across pipeline stages**: `EmbedderBundle` is the only object representing embedder+tokenizer configuration passed between ingestion, retrieval, reranking, and LLM context-building modules. No other object may re-resolve a tokenizer independently (prevents F-01, F-02, F-16; enforces MASTER CONTRACT, R-E1, R-E2).
  2. **Embedder-tokenizer binding**: `embedder.bound_tokenizer` must be exactly the same instance as `tokenizer`; any attempt to construct a separate tokenizer instance inside the embedder breaks the contract and must be rejected at build time (prevents F-01, F-02, F-03, F-16).
  3. **Dimension & metric immutability**: `dimension` and `distance_metric` are immutable once the bundle is created, and must match both the catalog entry and the VectorDB index collection configuration; any difference is a blocking configuration error (prevents F-04, F-05, F-08, F-09).
  4. **Verification semantics**:
     - `verification_status="verified"` → tokenizer family, max length, and special tokens match introspected values from config.json/tokenizer.json or Ollama modelfile parameters.
     - `verification_status="catalog_mismatch"` → catalog intentionally overrides introspected values; ingestion may proceed but alignment must be explicitly accepted.
     - `verification_status="unverified"` → tokenizer family or max length could not be conclusively determined; ingestion must be blocked unless explicitly allowed under a feature flag.
     (prevents F-01, F-02, F-16).
- **Failure modes prevented (primary)**:
  - F-01 (Vocabulary mismatch) — by bundling tokenizer family and model ID.
  - F-04 (Dimension mismatch) — by centralizing dimension and using it for VectorDB `ensure_collection`.
  - F-05 (Embedding space drift) — by tying model identity to index configuration.
  - F-06 (Normalization mismatch) — via `is_normalized` and validator checks.
  - F-08 (Distance metric mismatch) — via a shared `distance_metric` used for both indexing and querying.
  - F-16 (Tokenizer migration without re-index) — changing bundles invalidates old indexes via migration guard (see §5.4).

##### 1.5 TokenizerValidator
- **Name**: `TokenizerValidator`
- **Type**: Stateless service class (not an ABC) that consumes `TokenizerContract`, `EmbedderBundle`, reranker bundle, and `VectorDBConfig` to perform eager validation.
- **Supporting structures**:
  - `ValidationResult` (dataclass):
    - `ok: bool`
    - `errors: List[str]` — human-readable messages including failure-mode codes like `"F-02"`.
    - `warnings: List[str]`
    - `failure_modes: List[str]` — each entry is one of `"F-01"`…`"F-16"`.
    - `measured_tokens: Optional[int]` — for `validate_chunk`.
    - `safe_chunk_max: Optional[int]` — computed from `ChunkSizerContract`.
  - `AlignmentReport` (dataclass):
    - `ok: bool`
    - `errors: List[str]`
    - `warnings: List[str]`
    - `failure_modes: List[str]`
    - `embedder_model_id: str`
    - `reranker_model_id: Optional[str]`
    - `vectordb_kind: str`
    - `vectordb_collection: str`
    - `embedder_dimension: int`
    - `index_dimension: Optional[int]` — as configured in VectorDB (e.g., Pinecone `embedding_dim`).
    - `distance_metric_embedder: str`
    - `distance_metric_index: Optional[str]`
    - `is_normalized_embedder: bool`
    - `score_space_reranker: Optional[str]` — e.g., `"0_1"`, `"logits_-10_+12"`.
    - `tokenizer_family_embedder: str`
    - `tokenizer_family_reranker: Optional[str]`
    - `embed_max_tokens: int`
    - `reranker_max_tokens: Optional[int]`
    - `effective_chunk_max: int`
  - Exceptions:
    - `TokenOverflowError(ValueError)` — fields: `chunk_tokens: int`, `max_allowed: int`, `model_id: str`, `failure_mode: Literal["F-02", "F-13", "F-14", "F-15"]`.
    - `AlignmentError(ValueError)` — fields: `failure_modes: List[str]`, `embedder_model_id: str`, `vectordb_kind: str`, `details: Dict[str, Any]`.
- **Methods**:
  - `validate_chunk(chunk_text: str, tokenizer: TokenizerContract, embed_max: int, reranker_max: Optional[int]) -> ValidationResult`
    - Uses an internal `ChunkSizerContract` implementation or instance configured from catalog and reranker bundle.
    - On overflow, raises `TokenOverflowError` and returns a failed `ValidationResult`.
  - `validate_pipeline_alignment(embedder_bundle: EmbedderBundle, reranker_bundle: Optional[Any], vector_db_config: VectorDBConfig) -> AlignmentReport`
    - `reranker_bundle` must at least expose `model_id: str`, `max_input_tokens: int`, `tokenizer_family: str`, and `score_space: str` attributes, even if its full contract is defined elsewhere.
- **Invariants**:
  1. **Eager-only validation**: All validation methods must be called at pipeline construction time (e.g., inside `PipelineFactory.build` or ingestion pipeline bootstrap); they must not be callable lazily in the middle of an embedding or reranking operation (prevents silent run-time failures; applies across F-01–F-16).
  2. **Failure-mode tagging**: Every error and warning emitted must explicitly include at least one F-code; this makes audits and metric dashboards directly map back to the predefined failure modes (enforces explicit diagnostics for F-01–F-16).
  3. **No swallowing errors**: Any exception raised during validation must either bubble up as `TokenOverflowError`/`AlignmentError` or be logged as a structured error and surfaced to the caller; there is no silent fallback (enforces fail-fast; prevents F-02, F-04, F-05, F-08, F-09, F-11, F-13, F-16 from being ignored).
- **Failure modes prevented (primary)**:
  - F-02, F-13, F-14, F-15 via `validate_chunk` and chunk sizing.
  - F-04, F-05, F-06, F-08, F-09, F-10, F-11, F-12, F-16 via `validate_pipeline_alignment` and explicit checks.

#### 2. EMBEDDER CATALOG SCHEMA (YAML DESIGN)

##### 2.1 YAML Schema Definition
- **Top-level structure**:
  - Root document key: `embedders` — list of embedder entries.
- **Per-entry required fields** (types and constraints):
  - `model_id: str` — must be globally unique across the catalog.
  - `provider: one of {"openai", "cohere", "huggingface", "ollama", "anthropic", "google", "mistral"}`.
  - `tokenizer_class: str` — logical tokenizer implementation descriptor, e.g. `"tiktoken"`, `"transformers.AutoTokenizer"`, `"ollama_modelfile"`.
  - `tokenizer_encoding: Optional[str]` — required when `tokenizer_class` is `"tiktoken"` (e.g., `"cl100k_base"`).
  - `tokenizer_source: Optional[str]` — required when tokenizer is discovered from HF or Ollama (e.g., `"hf_config"`, `"hf_tokenizer_json"`, `"ollama_modelfile"`).
  - `tokenizer_family: one of {"tiktoken", "wordpiece", "sentencepiece", "bpe", "whitespace"}`.
  - `dimension: int` — positive integer; must match runtime vector length.
  - `distance_metric: one of {"cosine", "dotproduct", "euclidean"}`.
  - `is_normalized: bool` — `true` if L2-normalization is applied by the embedder.
  - `embed_max_tokens: int` — maximum tokens supported by the embedder’s tokenizer.
  - `default_reranker_id: Optional[str]` — reference to a reranker catalog entry; string like `"cohere/rerank-v3.0"`.
  - `query_prefix: str` — may be empty string; must be explicitly set.
  - `passage_prefix: str` — may be empty string; must be explicitly set.
  - `verification_status: one of {"verified", "catalog_mismatch", "unverified"}` — default `"unverified"` if not set; `"verified"` required for production use.
- **Per-entry optional fields**:
  - `embedder_type: Optional[str]` — further categorization (e.g. `"asymmetric_bi_encoder"`, `"symmetric_encoder"`); used for F-07 and F-11 interpretation.
  - `lang_support: Optional[str]` — e.g. `"en"`, `"multilingual"` (used for F-12 checks).
  - `notes: Optional[str]` — free-form documentation.
- **Validation rules (entry invalid if any violated)**:
  1. `model_id` must be non-empty and unique.
  2. `provider` must be from the enumerated set and consistent with `model_id` naming conventions.
  3. If `tokenizer_class == "tiktoken"`, then `tokenizer_encoding` must be non-empty; if `tokenizer_class` starts with `"transformers"`, `tokenizer_source` must be `"hf_config"` or `"hf_tokenizer_json"`; if `tokenizer_class == "ollama_modelfile"`, `tokenizer_source` must be `"ollama_modelfile"`.
  4. `tokenizer_family` must be consistent with `tokenizer_class` (e.g., `"tiktoken"` → `"tiktoken"`, HF BERT models → `"wordpiece"`, HF Mistral/Qwen models → `"sentencepiece"` or `"bpe"`).
  5. `dimension` must be >0 and must match the runtime dimension discovered from the model; discrepancies must set `verification_status="catalog_mismatch"` and cause a validation error unless explicitly allowed.
  6. `embed_max_tokens` must be >0 and must not exceed vendor-advertised limits; if unknown, entry is invalid and must not be deployed.
  7. If `default_reranker_id` is set, a corresponding reranker catalog entry must exist (even if defined in another file) and must support the same or greater input length and compatible language coverage.
  8. `verification_status` must be `"verified"` before an entry is eligible for production ingestion; other statuses are treated as non-production.

##### 2.2 Concrete YAML Entries

```yaml
embedders:
  # Cloud — OpenAI
  - model_id: "openai/text-embedding-ada-002"
    provider: "openai"
    tokenizer_class: "tiktoken"
    tokenizer_encoding: "cl100k_base"
    tokenizer_source: "openai_docs"
    tokenizer_family: "tiktoken"
    dimension: 1536
    distance_metric: "cosine"
    is_normalized: true
    embed_max_tokens: 8191
    default_reranker_id: "cohere/rerank-v3.0"
    query_prefix: ""
    passage_prefix: ""
    verification_status: "verified"

  - model_id: "openai/text-embedding-3-small"
    provider: "openai"
    tokenizer_class: "tiktoken"
    tokenizer_encoding: "cl100k_base"
    tokenizer_source: "openai_docs"
    tokenizer_family: "tiktoken"
    dimension: 1536
    distance_metric: "cosine"
    is_normalized: true
    embed_max_tokens: 8191
    default_reranker_id: "cohere/rerank-v3.0"
    query_prefix: ""
    passage_prefix: ""
    verification_status: "verified"

  - model_id: "openai/text-embedding-3-large"
    provider: "openai"
    tokenizer_class: "tiktoken"
    tokenizer_encoding: "cl100k_base"
    tokenizer_source: "openai_docs"
    tokenizer_family: "tiktoken"
    dimension: 3072
    distance_metric: "cosine"
    is_normalized: true
    embed_max_tokens: 8191
    default_reranker_id: "cohere/rerank-v3.0"
    query_prefix: ""
    passage_prefix: ""
    verification_status: "verified"

  # Cloud — Cohere
  - model_id: "cohere/embed-english-v3.0"
    provider: "cohere"
    tokenizer_class: "cohere_sdk"
    tokenizer_encoding: null
    tokenizer_source: "cohere_api"
    tokenizer_family: "wordpiece"
    dimension: 1024
    distance_metric: "cosine"
    is_normalized: true
    embed_max_tokens: 512
    default_reranker_id: "cohere/rerank-v3.0"
    query_prefix: ""
    passage_prefix: ""
    verification_status: "verified"

  # HuggingFace — BAAI/bge-large-en-v1.5
  - model_id: "BAAI/bge-large-en-v1.5"
    provider: "huggingface"
    tokenizer_class: "transformers.AutoTokenizer"
    tokenizer_encoding: null
    tokenizer_source: "hf_config"
    tokenizer_family: "wordpiece"
    dimension: 1024
    distance_metric: "cosine"
    is_normalized: true
    embed_max_tokens: 512
    default_reranker_id: "bge-reranker-large"
    query_prefix: "query: "
    passage_prefix: "passage: "
    verification_status: "verified"

  # HuggingFace — intfloat/e5-mistral-7b-instruct (asymmetric)
  - model_id: "intfloat/e5-mistral-7b-instruct"
    provider: "huggingface"
    tokenizer_class: "transformers.AutoTokenizer"
    tokenizer_encoding: null
    tokenizer_source: "hf_config"
    tokenizer_family: "sentencepiece"
    dimension: 4096
    distance_metric: "cosine"
    is_normalized: true
    embed_max_tokens: 131072
    default_reranker_id: "bge-reranker-large"
    query_prefix: "query: "
    passage_prefix: "passage: "
    verification_status: "verified"

  # HuggingFace — mixedbread-ai/mxbai-embed-large-v1
  - model_id: "mixedbread-ai/mxbai-embed-large-v1"
    provider: "huggingface"
    tokenizer_class: "transformers.AutoTokenizer"
    tokenizer_encoding: null
    tokenizer_source: "hf_config"
    tokenizer_family: "bpe"
    dimension: 1024
    distance_metric: "cosine"
    is_normalized: true
    embed_max_tokens: 512
    default_reranker_id: "bge-reranker-large"
    query_prefix: ""
    passage_prefix: ""
    verification_status: "unverified"  # tokenizer family and max length must be confirmed at runtime

  # HuggingFace — Qwen/Qwen3-Embedding (tiktoken variant)
  - model_id: "Qwen/Qwen3-Embedding"
    provider: "huggingface"
    tokenizer_class: "transformers.AutoTokenizer"
    tokenizer_encoding: null
    tokenizer_source: "hf_config"
    tokenizer_family: "tiktoken"
    dimension: 1536
    distance_metric: "cosine"
    is_normalized: true
    embed_max_tokens: 8192
    default_reranker_id: "cohere/rerank-v3.0"
    query_prefix: ""
    passage_prefix: ""
    verification_status: "unverified"  # requires verification against actual HF model config

  # Ollama — nomic-embed-text (WordPiece, 2048 max)
  - model_id: "ollama/nomic-embed-text"
    provider: "ollama"
    tokenizer_class: "ollama_modelfile"
    tokenizer_encoding: null
    tokenizer_source: "ollama_modelfile"
    tokenizer_family: "wordpiece"
    dimension: 768
    distance_metric: "cosine"
    is_normalized: true
    embed_max_tokens: 2048
    default_reranker_id: "cohere/rerank-v3.0"
    query_prefix: ""
    passage_prefix: ""
    verification_status: "unverified"  # family inferred from model name; must be confirmed from modelfile

  # Ollama — mxbai-embed-large
  - model_id: "ollama/mxbai-embed-large"
    provider: "ollama"
    tokenizer_class: "ollama_modelfile"
    tokenizer_encoding: null
    tokenizer_source: "ollama_modelfile"
    tokenizer_family: "bpe"
    dimension: 1024
    distance_metric: "cosine"
    is_normalized: true
    embed_max_tokens: 512
    default_reranker_id: "bge-reranker-large"
    query_prefix: ""
    passage_prefix: ""
    verification_status: "unverified"  # requires verification from Ollama modelfile
```

- **Notes on incomplete values**:
  - Entries with `verification_status: "unverified"` explicitly indicate that some required fields (e.g., tokenizer family, max tokens) were inferred and must be validated against introspected metadata at build time.
  - Any attempt to use an `unverified` embedder in production ingestion must be blocked by the validation layer unless an explicit configuration flag authorizes it for testing.

#### 3. REGISTRY DESIGN

##### 3.1 EmbedderRegistry.get(model_id: str) → EmbedderBundle
- **Overview**: `EmbedderRegistry` is a new, configuration-driven registry that resolves `EmbedderBundle` instances from `embedder_catalog.yaml`, layered on top of the existing plugin-based `embedder_registry` used by `PipelineFactory`.
- **Step-by-step resolution pipeline**:
  1. **Lookup in catalog**
     - Read `embedder_catalog.yaml` and locate the entry whose `model_id` matches the input string.
     - If none found, raise `UnknownEmbedderModelError(model_id)`.
  2. **Schema validation**
     - Validate the entry against the schema in §2.1.
     - If invalid (missing required fields, out-of-range values, or unsupported enums), raise `EmbedderCatalogValidationError(model_id, details)`.
  3. **Determine path (cloud vs local)**
     - Use `provider` to choose:
       - Cloud-native path (OpenAI, Cohere, Anthropic, Google, Mistral, etc.).
       - Local HuggingFace path.
       - Local Ollama path.
  4. **Construct tokenizer (Tokenizers are never resolved in embedders)**
     - For cloud providers: instantiate a `TokenizerContract` based purely on catalog information (e.g., tiktoken with `tokenizer_encoding` for OpenAI models).
     - For HuggingFace: follow §3.3 to load `AutoTokenizer`, read `config.json` and `tokenizer.json`, and cross-check against the catalog.
     - For Ollama: follow §3.4 to run `ollama show <model_id> --modelfile` and parse tokenizer family.
     - On failure to load or classify, raise `TokenizerResolutionError(model_id, reason)`.
  5. **Construct embedder implementation**
     - Use existing plugin `embedder_registry` to build the concrete `BaseEmbedder` / `EmbedderContract` instance with the appropriate provider-specific parameters (API keys via env for cloud, model IDs for local), but **do not** let that implementation resolve its own tokenizer.
     - Inject `bound_tokenizer` from step 4.
  6. **Derive and verify dimension and normalization**
     - Use catalog’s `dimension` and `is_normalized` values as the source of truth.
     - Optionally probe the embedder by embedding a short string to confirm vector length; discrepancies set `verification_status="catalog_mismatch"` and cause an `AlignmentError` unless explicitly allowed for testing.
  7. **Assemble EmbedderBundle**
     - Build `EmbedderBundle` with `model_id`, `provider`, `tokenizer`, `embedder`, `dimension`, `distance_metric`, `is_normalized`, `embed_max_tokens`, `default_reranker_id`, `tokenizer_family`, and computed `verification_status`.
  8. **Cache and return**
     - Store the resulting bundle in an in-memory cache keyed by `model_id` and catalog hash (see §3.5), then return it.

##### 3.2 Error Handling Strategy
- **Unknown `model_id`**:
  - Error: `UnknownEmbedderModelError(model_id: str)`.
  - Message: `"EmbedderRegistry: model_id='<id>' not found in embedder_catalog.yaml (F-05/F-16 risk)."`
  - Raised at step 1.
- **Invalid catalog entry**:
  - Error: `EmbedderCatalogValidationError(model_id: str, details: Dict[str, Any])`.
  - Message includes specific field errors and relevant failure modes (e.g., F-04/F-05/F-06/F-08/F-16).
  - Raised at step 2.
- **HuggingFace tokenizer load failure**:
  - Error: `TokenizerResolutionError(model_id: str, provider: str, reason: str)`.
  - Raised at step 4 when `AutoTokenizer.from_pretrained(model_id)` fails or required metadata fields are missing.
- **Ollama modelfile parse failure**:
  - If `ollama show <model_id> --modelfile` fails or does not contain parsable tokenizer family info:
    - Log a structured WARNING containing `model_id`, `raw_modelfile_snippet`, and an F-code hint (F-01/F-16).
    - Set tokenizer family to `"tiktoken"`, tokenizer encoding to `"cl100k_base"`, and `verification_status="unverified"`.
    - Proceed to construct the `EmbedderBundle`, but `TokenizerValidator.validate_pipeline_alignment` must treat this as non-production and either block ingestion or require explicit override.

##### 3.3 HuggingFace Auto-resolution Logic
- **Step-by-step for HuggingFace models**:
  1. **Load tokenizer**: `AutoTokenizer.from_pretrained(model_id)` with `use_fast=True`.
  2. **Read `config.json`**:
     - Extract `model_type` (e.g., `"bert"`, `"mistral"`, `"qwen"`).
     - Extract `max_position_embeddings` if present.
  3. **Read `tokenizer.json`**:
     - Extract `"model"` → `"type"` field to determine tokenizer family (`"BPE"`, `"WordPiece"`, `"SentencePieceUnigram"`, etc.).
  4. **Map tokenizer type to `tokenizer_family`**:
     - `"WordPiece"` → `"wordpiece"`.
     - `"BPE"` → `"bpe"`.
     - `"SentencePieceUnigram"` or similar → `"sentencepiece"`.
  5. **Compare with catalog entry**:
     - If family or `max_position_embeddings` disagree with catalog values:
       - Log a discrepancy with full context.
       - Set `verification_status="catalog_mismatch"` in `EmbedderBundle`.
       - Depending on system policy, either fail fast (`AlignmentError`) or allow only in non-production.
  6. **Catalog-wins rule**:
     - When merging HF-discovered values with catalog values, the catalog entry is authoritative; any attempt to silently override catalog with discovered values is forbidden.

##### 3.4 Ollama Parsing Logic
- **Step-by-step**:
  1. **Run modelfile command**: `ollama show <model_id> --modelfile`.
  2. **Parse tokenizer family** from `PARAMETER` entries or comments:
     - If model name contains a known family:
       - `"llama"` or `"mistral"` → `tokenizer_family="sentencepiece"`.
       - `"nomic-embed-text"` → `tokenizer_family="wordpiece"`.
       - `"qwen"` or `"phi"` → `tokenizer_family="tiktoken"`.
  3. **Cross-check with catalog**:
     - If catalog `tokenizer_family` matches parsed family → `verification_status="verified"`.
     - If they differ → `verification_status="catalog_mismatch"` and alignment error unless explicitly overridden.
  4. **Unresolvable family**:
     - If no mapping can be determined, set tokenizer family to `"tiktoken"` with `tokenizer_encoding="cl100k_base"`, set `verification_status="unverified"`, and emit a WARNING.
     - `EmbedderBundle` is constructed but treated as unsafe for production until validated.

##### 3.5 Caching Strategy
- **Bundle cache**:
  - Cache key: `(model_id, catalog_hash)` where `catalog_hash` is a stable hash (e.g., SHA256) of the entire `embedder_catalog.yaml` content.
  - Value: `EmbedderBundle` instance with immutable state.
- **Invalidation**:
  - On `EmbedderRegistry` initialization, compute `catalog_hash` once.
  - On any call to `get`, re-compute `catalog_hash` from disk; if it differs from the cached value, clear the entire cache (ensuring that catalog edits are always reflected).
- **Thread-safety**:
  - Protect cache read/write with a `RLock` or similar, ensuring only one bundle is constructed per `(model_id, catalog_hash)` in concurrent environments.
  - Tokenizer implementations used inside bundles must themselves satisfy `is_thread_safe=True` or be protected by per-instance synchronization.

#### 4. VALIDATION LAYER

##### 4.1 `validate_chunk(chunk_text, tokenizer, embed_max, reranker_max)`
- **Method signature**: `validate_chunk(chunk_text: str, tokenizer: TokenizerContract, embed_max: int, reranker_max: Optional[int]) -> ValidationResult`.
- **Checks (in order)**:
  1. **Input normalization**
     - Condition: `chunk_text` is not `None`; treat empty or whitespace-only text as zero-token chunks.
     - Error: none; zero-length chunks can be filtered by separate business logic.
  2. **Token counting**
     - Condition: `token_count = tokenizer.count_tokens(chunk_text, include_special_tokens=False)`.
     - Error: if any exception occurs during counting, wrap and raise `TokenOverflowError` with failure modes `["F-02"]`.
  3. **Compute safe_chunk_max**
     - Condition: Use an internal `ChunkSizerContract` instance configured with `embed_max`, `reranker_max`, `query_reserve_tokens`, `special_overhead_tokens`, and `headroom_pct`.
     - Error: if `safe_chunk_max <= 0`, raise `TokenOverflowError` with message `"Computed chunk_max <= 0 (F-13/F-15): configuration invalid"`.
  4. **Overflow check**
     - Condition: `token_count <= safe_chunk_max`.
     - Error: if violated, raise `TokenOverflowError` with message `"Chunk token length {token_count} exceeds safe limit {safe_chunk_max} (F-02/F-13)."`.
  5. **Return result**
     - If all checks pass, return `ValidationResult(ok=True, errors=[], warnings=[], failure_modes=[], measured_tokens=token_count, safe_chunk_max=safe_chunk_max)`.

##### 4.2 `validate_pipeline_alignment(embedder_bundle, reranker_bundle, vector_db_config) → AlignmentReport`
- **Method signature**: `validate_pipeline_alignment(embedder_bundle: EmbedderBundle, reranker_bundle: Optional[Any], vector_db_config: VectorDBConfig) -> AlignmentReport`.
- **Alignment checks (numbered)**:
  1. **Dimension vs VectorDB index (F-04, F-09, F-16)**
     - Compare `embedder_bundle.dimension` with any dimension configured in `vector_db_config` (e.g., `PineconeConfig.embedding_dim`).
     - Failure: mismatch → `AlignmentError` with failure modes `["F-04", "F-09", "F-16"]`.
  2. **Distance metric alignment (F-08)**
     - Compare `embedder_bundle.distance_metric` with VectorDB metric (`PineconeConfig.metric`, equivalent fields for Qdrant/Weaviate/Milvus/Redis if present in their configs or connectors).
     - Failure: mismatch → `AlignmentError` with `["F-08"]`.
  3. **Normalization alignment (F-06, F-11)**
     - Ensure that if `embedder_bundle.is_normalized` is `True`, the VectorDB uses a metric compatible with normalized vectors (e.g., `"cosine"`), and scoring logic in reranker or post-processing accounts for that.
     - Failure: `AlignmentError` with `["F-06", "F-11"]`.
  4. **Tokenizer family consistency (F-01, F-16)**
     - Ensure that `embedder_bundle.tokenizer_family` is used consistently for chunk sizing and embedding; no other stage may reinterpret strings using a different tokenizer family.
     - Failure: `AlignmentError` with `["F-01", "F-16"]`.
  5. **Reranker input length (F-13)**
     - If `reranker_bundle` is present, ensure that `reranker_bundle.max_input_tokens >= effective_chunk_max` from `ChunkSizerContract`.
     - Failure: `AlignmentError` with `["F-13"]`.
  6. **Score space compatibility (F-11)**
     - If `reranker_bundle` exposes `score_space` and VectorDB uses a particular score semantics, ensure that downstream consumers do not mix incompatible scales (e.g., 0–1 probabilities with logits).
     - Failure: warning or blocking error tagged with `["F-11"]` depending on policy.
  7. **Language/domain coverage (F-12)**
     - Use catalog `lang_support` (if present) plus reranker’s language coverage to ensure they are consistent with corpus language; mismatches produce at least a warning.
  8. **LLM context tokenizer separation (F-14, F-15, R-L1–R-L3)**
     - Ensure LLM configuration uses its own tokenizer and context window for measuring prompt fit; embedding and chunking decisions must not rely on LLM tokenizer.
     - Fails with a configuration error if any component attempts to reuse the embedder’s tokenizer for LLM prompts.
- **Blocking vs warning**:
  - Dimension, metric, normalization, tokenizer family, and reranker input length mismatches are **blocking** (must raise `AlignmentError`).
  - Score space and language/domain mismatches can start as **warnings** but should be promotable to blocking via configuration for strict deployments.

##### 4.3 ValidationResult structure
- As defined above in §1.5; must always include an `ok` boolean, lists of `errors` and `warnings`, `failure_modes`, and optional numeric fields `measured_tokens` and `safe_chunk_max`.

##### 4.4 AlignmentReport structure
- As defined above in §1.5; captures all relevant parameters for alignment checks, plus `ok`, `errors`, `warnings`, and `failure_modes`.

##### 4.5 Custom Exceptions
- **TokenOverflowError(ValueError)**
  - Fields: `chunk_tokens: int`, `max_allowed: int`, `model_id: str`, `failure_mode: str`.
  - Raised by: `validate_chunk` when token counts exceed safe limits.
- **AlignmentError(ValueError)**
  - Fields: `failure_modes: List[str]`, `embedder_model_id: str`, `vectordb_kind: str`, `details: Dict[str, Any]`.
  - Raised by: `validate_pipeline_alignment` on any blocking mismatch.
- **UnknownEmbedderModelError(KeyError)**
  - Fields: `model_id: str`.
  - Raised by: `EmbedderRegistry.get` when `model_id` is not found.
- **EmbedderCatalogValidationError(ValueError)**
  - Fields: `model_id: str`, `details: Dict[str, Any]`.
  - Raised by: schema validation of `embedder_catalog.yaml` entries.
- **TokenizerResolutionError(RuntimeError)**
  - Fields: `model_id: str`, `provider: str`, `reason: str`.
  - Raised by: tokenizer loading and family resolution logic for HuggingFace and Ollama.

#### 5. PHASED BUILD PLAN

##### 5.1 Module Build Sequence (strict order)
1. **Module: `TokenizerContract` and tokenizer families**
   - **Depends on**: existing `app/core/tokenization/base.py` patterns.
   - **Delivers**: canonical tokenizer contract bound to embedding models and families, with precise max-length and special-token semantics.
   - **Integration point**: used by `EmbedderBundle`, `ChunkSizerContract`, and `TokenizerValidator` for token counting and family checks.
2. **Module: `EmbedderContract` and `EmbedderBundle`**
   - **Depends on**: `TokenizerContract`, existing `BaseEmbedder`, and provider-specific embedder implementations.
   - **Delivers**: unified object that couples embedder, tokenizer, dimension, metric, and normalization into a single immutable `EmbedderBundle` passed across ingestion and retrieval.
   - **Integration point**: used by ingest chunkers, VectorDB wrappers, and `PipelineFactory` when wiring `AssembledPipeline`.
3. **Module: `embedder_catalog.yaml` loader and schema validation**
   - **Depends on**: `EmbedderBundle` field definitions and enumerations from `ClientConfig` (e.g., `EmbedderType`).
   - **Delivers**: validated in-memory representation of all embedder entries, with eager failure on any schema or semantic issues.
   - **Integration point**: consumed by `EmbedderRegistry` to resolve `EmbedderBundle` instances.
4. **Module: `EmbedderRegistry` (bundle resolution and caching)**
   - **Depends on**: `EmbedderBundle`, catalog loader, tokenizer contracts, and existing `embedder_registry` plugin system.
   - **Delivers**: `EmbedderRegistry.get(model_id) -> EmbedderBundle` for both ingestion and RAG query pipelines, with caching and verification.
   - **Integration point**: used by `PipelineFactory.build` and ingestion services (e.g., `ingestion_service_v2`) to obtain consistent bundling.
5. **Module: `ChunkSizerContract` and `TokenizerValidator`**
   - **Depends on**: `TokenizerContract`, `EmbedderBundle`, reranker config, and `VectorDBConfig`.
   - **Delivers**: eager validation of chunk sizes and pipeline alignment against all F-01–F-16 failure modes.
   - **Integration point**: invoked during pipeline construction and ingestion job start, blocking misaligned configurations before any data is indexed.

##### 5.2 Validation Checkpoints
- **After Module 3 (catalog loader)**:
  - Run catalog-level validation to ensure all entries are internally consistent and have non-`"unverified"` statuses where required.
- **After Module 4 (EmbedderRegistry)**:
  - For each configured client (`ClientConfig.embedder`), resolve an `EmbedderBundle` and validate that dimensions, tokenizer family, and max tokens match expectations.
- **After Module 5 (Validator)**:
  - At `PipelineFactory.build` time, call `validate_pipeline_alignment` with the `EmbedderBundle`, reranker configuration (if any), and `VectorDBConfig`.
  - At ingestion pipeline startup, validate chunk sizes via `validate_chunk` on a representative sample or by schema-only checks of `ChunkConfig` against `EmbedderBundle` and reranker settings.

##### 5.3 Circular Dependency Risks
- **Potential cycles**:
  - `EmbedderRegistry` ↔ `PipelineFactory`: avoid the registry depending on `PipelineFactory`; instead, `PipelineFactory` depends on `EmbedderRegistry`.
  - `TokenizerValidator` ↔ embedder/reranker implementations: ensure validator only depends on contracts and bundles, never concrete implementations.
  - `VectorDBConfig` ↔ `BaseVectorDB`: keep config schema (`client_config_schema.py`) independent of runtime connector implementations.
- **Breaking cycles**:
  - Use contracts and dataclasses (`EmbedderBundle`, `TokenizerContract`, `ChunkSizerContract`) as the shared interfaces; concrete implementations live in provider-specific modules and registries.
  - Any additional cross-component references should be via Protocols or minimal views (e.g., reranker bundle view with only fields needed for validation).

##### 5.4 Migration Guard (Tokenizer Change → Re-index Enforcement)
- **Embedding-space fingerprint**:
  - Define an `embedding_fingerprint` derived from `EmbedderBundle` (e.g., hash of `provider`, `model_id`, `tokenizer_family`, `dimension`, `distance_metric`, `is_normalized`).
  - Store this fingerprint in VectorDB collection metadata (e.g., as part of collection name suffix, or a per-collection metadata payload where supported) at `ensure_collection` time.
- **Guard logic**:
  - On subsequent pipeline builds or ingestion job startups, compute the current `embedding_fingerprint` and compare it with the one stored in the index metadata.
  - If the fingerprints differ (different model, tokenizer family, dimension, metric, or normalization), mark the index as incompatible and:
    - Fail fast with an `AlignmentError` tagged with `["F-05", "F-16"]`.
    - Require an explicit re-index operation (drop & recreate collection or rebuild embeddings) before ingestion or retrieval can proceed.
- **Addresses**:
  - R-C4 (Tokenizer migration requires full re-index) and F-16 (Tokenizer migration without re-index) by making embedding-space changes impossible to ignore.

### Optimizations
- **Token counting performance**: Use efficient tokenizer backends (tiktoken for OpenAI/Qwen, HF fast tokenizers for HF/Ollama) and cache `EmbedderBundle` instances to minimize repeated initialization overhead.
- **Config reuse**: Reuse `EmbedderBundle` across ingestion and RAG query paths, and reuse `ChunkSizerContract` parameters for both chunking and validator checks.
- **Observability hooks**: Attach structured logging and metrics to validation failures, including failure-mode codes and fingerprint mismatches, to quickly spot configuration regressions in CI/CD and production.

### Risks & Edge Cases
- **Incomplete catalog entries**: Entries with `verification_status="unverified"` are explicitly blocked or require override; rollout must include a catalog hardening phase where all production models are fully verified.
- **Vendor changes**: Providers may change context window sizes or recommend new tokenizer families; keeping `embedder_catalog.yaml` as the single source of truth, plus eager validator checks, ensures these changes are surfaced during deployment, not at runtime.
- **Mixed deployments**: Multi-tenant or multi-client setups with different embedders per client must ensure that `EmbedderBundle` fingerprints are tracked per collection and client to avoid cross-tenant contamination of embedding spaces.
