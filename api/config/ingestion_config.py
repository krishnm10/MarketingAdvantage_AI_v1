# ==========================================================
# 🧠 ingestion_config.py — Centralized Configuration Loader (v1.5)
# ==========================================================
"""
Loads and validates all ingestion-related settings from:
    api/config/ingestion_config.yaml

Used by:
  - intelligent_*_ingest.py modules
  - unified_ingest_helper.py
  - llm_classifier.py
  - intelligent_auto_ingest_router.py
  - main.py

Purpose:
  ✅ Centralize model, chunking, and ChromaDB settings
  ✅ Allow flexible tuning without editing code
  ✅ Ensure consistent ingestion behavior across all data sources
"""

import os
import yaml
import logging
from typing import Any, Dict

# ==========================================================
# ⚙️ PATHS
# ==========================================================
CONFIG_PATH = os.path.join(os.getcwd(), "api", "config", "ingestion_config.yaml")

# ==========================================================
# 🧾 LOGGING
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Config] %(message)s"
)
logger = logging.getLogger("ingestion_config")

# ==========================================================
# 🧠 DEFAULT CONFIG (Auto-Fallback)
# ==========================================================
DEFAULT_CONFIG = {
    "version": "1.0",
    "llm": {
        "model_name": "llama3.1:8b",
        "prompt_version": "4.0-Llama-Hyper-Pro",
        "temperature": 0.2,
        "max_tokens": 4096
    },
    "chunking": {
        "method": "semantic_recursive",
        "max_chunk_size": 1800,
        "overlap_ratio": 0.15,
        "min_sentence_length": 30
    },
    "chromadb": {
        "persist_directory": "data/rag_db",
        "collection_name": "enterprise_knowledge",
        "distance_metric": "cosine"
    },
    "paths": {
        "upload_dir": "data/uploads",
        "logs_dir": "logs"
    },
    "file_watcher": {
        "auto_start": True
    },
    "modules": {
        "enabled": ["excel", "pdf", "docx", "txt", "json", "rss"]
    }
}

# ==========================================================
# 🧩 CONFIG CACHE
# ==========================================================
_cached_config: Dict[str, Any] = {}

# ==========================================================
# 🚀 CORE LOADER FUNCTION
# ==========================================================
def get_ingestion_config() -> Dict[str, Any]:
    """
    Load and merge ingestion configuration.
    Returns DEFAULT_CONFIG if YAML missing or invalid.
    """
    global _cached_config

    if _cached_config:
        return _cached_config

    if not os.path.exists(CONFIG_PATH):
        logger.warning(f"⚠️ ingestion_config.yaml not found. Using defaults.")
        _cached_config = DEFAULT_CONFIG
        return _cached_config

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}

        # Deep-merge user config with defaults
        def merge_dicts(default: dict, user: dict) -> dict:
            merged = default.copy()
            for k, v in user.items():
                if isinstance(v, dict) and k in merged:
                    merged[k] = merge_dicts(merged[k], v)
                else:
                    merged[k] = v
            return merged

        _cached_config = merge_dicts(DEFAULT_CONFIG, user_config)
        logger.info(f"✅ Loaded ingestion config (version {_cached_config.get('version', 'N/A')})")
        return _cached_config

    except Exception as e:
        logger.error(f"❌ Failed to load ingestion configuration: {e}")
        _cached_config = DEFAULT_CONFIG
        return _cached_config

# ==========================================================
# 🔍 ACCESSORS
# ==========================================================
def get_model_settings() -> Dict[str, Any]:
    """Return LLM model configuration."""
    return get_ingestion_config().get("llm", {})

def get_chunking_settings() -> Dict[str, Any]:
    """Return semantic chunking configuration."""
    return get_ingestion_config().get("chunking", {})

def get_chroma_settings() -> Dict[str, Any]:
    """Return ChromaDB settings."""
    return get_ingestion_config().get("chromadb", {})

def get_enabled_modules() -> list:
    """Return list of enabled ingestion modules."""
    return get_ingestion_config().get("modules", {}).get("enabled", [])

def get_path_settings() -> Dict[str, str]:
    """Return file path settings."""
    return get_ingestion_config().get("paths", {})

def is_file_watcher_enabled() -> bool:
    """Check if file watcher should auto-start."""
    return get_ingestion_config().get("file_watcher", {}).get("auto_start", True)

# ==========================================================
# 🧪 SELF-TEST
# ==========================================================
if __name__ == "__main__":
    cfg = get_ingestion_config()
    print("✅ Config loaded successfully:")
    print(yaml.dump(cfg, sort_keys=False, allow_unicode=True))

load_ingest_config = get_ingestion_config
