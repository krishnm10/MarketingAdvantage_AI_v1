# =============================================
# ingestion_settings.py — Centralized Ingestion Configuration
# Controls parser-wide feature toggles and LLM integration
# =============================================

from typing import Final



# -------------------------------------------------------------------
# 🧠 EMBEDDING MODEL CONFIGURATION (CRITICAL)
# -------------------------------------------------------------------
# The source of truth for all ingestion and retrieval. 
# Changing this requires re-indexing the entire database and ChromaDB.
EMBEDDING_MODEL_NAME: Final[str] = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSION: Final[int] = 1024  # bge-large models use 1024 dims

# -------------------------------------------------------------------
# 🧠 LLM MODE SELECTION (Dynamic toggle for factual vs creative)
# -------------------------------------------------------------------
# factual  → Strict factual rewrite (no hallucination, no paraphrase)
# creative → Natural human-style rewrite (fluency & readability)
LLM_MODE: str = "factual"  # Options: "factual" | "creative"

# -------------------------------------------------------------------
# 🚀 GLOBAL INGESTION CONFIGURATION
# -------------------------------------------------------------------
# Master flag controlling whether LLM normalization is used across parsers.
# -------------------------------------------------------
# True  → Cleaned + normalized by LLM (Ollama/OpenAI)
# False → Only cleaned text (no LLM processing)
ENABLE_LLM_NORMALIZATION: bool = False

# -------------------------------------------------------------------
# ⚙️ LLM SETTINGS (used by llm_rewriter.py and all parsers)
# -------------------------------------------------------------------
# Shared across csv_parser_v2, excel_parser_v2, json_parser_v2,
# xml_parser_v2, rss_ingestor_v2, web_scraper_v2, etc.

LLM_PROVIDER: Final[str] = "ollama"        # Options: "ollama", "openai", "none"
OLLAMA_API_URL: Final[str] = "http://localhost:11434/api/generate"
OLLAMA_MODEL: Final[str] = "llama3.1:8b"

# Retry/backoff strategy for the rewrite queue (llm_rewriter.py)
LLM_MAX_RETRIES: Final[int] = 3
LLM_BACKOFF_SECONDS: Final[int] = 2

# -------------------------------------------------------------------
# ⚙️ PARSER BEHAVIOR FLAGS
# -------------------------------------------------------------------
# Used to control ingestion performance and safety behaviors.

# Deduplication and smart chunk segmentation
ENABLE_DEDUP_AWARE_SEGMENTATION: Final[bool] = True

# Auto-start background ingestion for newly uploaded files
ENABLE_WATCHER_INGESTION: Final[bool] = True

# Perform lightweight text cleaning even when LLM is disabled
ENABLE_BASELINE_TEXT_CLEANING: Final[bool] = True

# -------------------------------------------------------------------
# 🧩 FUTURE FEATURE FLAGS (Taxonomy + Classification Modules)
# -------------------------------------------------------------------
# Reserved for Phase 2 enhancements — business taxonomy and AI classification
ENABLE_AUTO_BUSINESS_CLASSIFICATION: Final[bool] = False
ENABLE_TAXONOMY_DISCOVERY: Final[bool] = False

# -------------------------------------------------------------------
# ✅ CONFIG SUMMARY HELPER
# -------------------------------------------------------------------
def summarize_ingestion_config() -> None:
    """Logs current ingestion settings for debugging or startup verification."""
    print("==== INGESTION CONFIGURATION ====")
    print(f"LLM_MODE: {LLM_MODE}")
    print(f"LLM Normalization Enabled: {ENABLE_LLM_NORMALIZATION}")
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"Ollama API URL: {OLLAMA_API_URL}")
    print(f"Ollama Model: {OLLAMA_MODEL}")
    print(f"Dedup-Aware Segmentation: {ENABLE_DEDUP_AWARE_SEGMENTATION}")
    print(f"Watcher Ingestion: {ENABLE_WATCHER_INGESTION}")
    print(f"Baseline Cleaning: {ENABLE_BASELINE_TEXT_CLEANING}")
    print(f"Auto Business Classification: {ENABLE_AUTO_BUSINESS_CLASSIFICATION}")
    print(f"Taxonomy Discovery: {ENABLE_TAXONOMY_DISCOVERY}")
    print("=================================")
