# =============================================
# MOVED to app/core/chunking_stratagies/segmenter_v2.py
# This shim exists for backward compatibility with any code that
# still imports from the old path. Remove once all consumers verified.
# =============================================
from app.core.chunking_stratagies.segmenter_v2 import (  # noqa: F401
    DEFAULT_MAX_CHUNK_LEN,
    DEFAULT_MIN_CHUNK_LEN,
    MAX_SPLIT_DEPTH,
    count_tokens,
    make_semantic_hash,
    build_reasoning_ingestion_metadata,
    merge_small_chunks,
    make_chunk_dict,
    recursive_semantic_chunk,
)
