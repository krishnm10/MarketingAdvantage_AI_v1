# =============================================
# MOVED to app/core/chunking_stratagies/chunking_registry.py
# This shim exists for backward compatibility with any code that
# still imports from the old path. Remove once all consumers verified.
# =============================================
from app.core.chunking_stratagies.chunking_registry import (  # noqa: F401
    Chunker,
    register_chunker,
    get_chunker,
    list_chunking_strategies,
    clear_chunker_cache,
)

