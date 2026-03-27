# =============================================
# app.core.chunking_stratagies — Enterprise Chunking Engine
#
# All text chunking strategies for the ingestion pipeline.
# Registry pattern: each segmenter_*.py auto-registers via
# @register_chunker("name") decorator on import.
#
# Public API:
#   get_chunker(strategy)        → Chunker instance
#   list_chunking_strategies()   → sorted list of registered names
#   clear_chunker_cache()        → invalidate LRU cache
# =============================================
