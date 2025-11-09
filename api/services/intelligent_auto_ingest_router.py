# ==========================================================
# 🚀 intelligent_auto_ingest_router.py — Unified Ingestion Router (v4.1)
# ==========================================================
"""
Central routing module for all intelligent ingestion types:
Excel, PDF, DOCX, TXT, JSON, RSS/API.

Automatically detects file type or source mode,
loads YAML config for routing preferences, and calls the
appropriate ingestion module with unified storage and taxonomy linking.

✅ Supports Excel, PDF, DOCX, TXT, JSON, RSS/API
✅ Auto-detects file extensions or source types
✅ Centralized logging + error tracking
✅ Taxonomy-aware classification via LLM
✅ YAML configuration driven

Author: MarketingAdvantage™ AI Core Team
Version: 4.1
"""

import os
import logging
from datetime import datetime

# ----------------------------------------------------------
# 🧩 Import Config + Ingestion Modules
# ----------------------------------------------------------
from api.config.ingestion_config import get_ingestion_config
from api.services.intelligent_excel_ingest import ingest_excel_or_csv
from api.services.intelligent_docx_ingest import ingest_docx
from api.services.intelligent_pdf_ingest import ingest_pdf
from api.services.intelligent_text_ingest import ingest_text
from api.services.intelligent_json_ingest import ingest_json
from api.services.intelligent_rss_ingest import ingest_rss_or_api

# ----------------------------------------------------------
# ⚙️ CONFIGURATION
# ----------------------------------------------------------
config = get_ingestion_config()
active_profile_name = config.get("profiles", {}).get("active", "balanced")
active_profile = config.get("profiles", {}).get(active_profile_name, {})

print(f"[Profile] 🧠 Active Ingestion Profile: {active_profile_name.upper()}")

LOG_PATH = os.path.join(config.get("paths", {}).get("logs_dir", "logs"), "auto_ingest_router.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [AutoRouter] %(message)s",
)
logger = logging.getLogger("intelligent_auto_ingest_router")

# ==========================================================
# 🧩 DETECTION
# ==========================================================
def detect_ingest_type(input_path_or_url: str) -> str:
    """
    Detect ingestion mode based on file extension or URL pattern.
    Returns: excel | pdf | docx | txt | json | rss | api | unknown
    """
    if input_path_or_url.startswith("http"):
        if "feed" in input_path_or_url or "rss" in input_path_or_url:
            return "rss"
        return "api"

    ext = os.path.splitext(input_path_or_url.lower())[1]
    if ext in [".xlsx", ".xls", ".csv"]:
        return "excel"
    elif ext == ".pdf":
        return "pdf"
    elif ext == ".docx":
        return "docx"
    elif ext == ".txt":
        return "txt"
    elif ext == ".json":
        return "json"
    else:
        return "unknown"

# ==========================================================
# 🚀 ROUTING
# ==========================================================
def route_ingestion(input_path_or_url: str):
    
    """
    Automatically routes the ingestion process based on file type or URL.
    Applies LLM and chunking settings dynamically based on active profile.
    """
    ingest_type = detect_ingest_type(input_path_or_url)
    logger.info(f"[Route] Detected ingestion type: {ingest_type.upper()} for {input_path_or_url}")
    logger.info(f"[Route] Using ingestion profile: {active_profile_name.upper()}")

    # 🔧 Profile-based feature flags
    llm_enabled = active_profile.get("llm_enabled", True)
    semantic_chunking = active_profile.get("semantic_chunking", True)
    recursive_fallback = active_profile.get("recursive_fallback", True)
    confidence_threshold = active_profile.get("classification_confidence_threshold", 0.8)
    
    try:
        if ingest_type == "excel":
            result = ingest_excel_or_csv(input_path_or_url)
        elif ingest_type == "pdf":
            result = ingest_pdf(input_path_or_url)
        elif ingest_type == "docx":
            result = ingest_docx(input_path_or_url)
        elif ingest_type == "txt":
            result = ingest_text(input_path_or_url, file_mode=True)
        elif ingest_type == "json":
            result = ingest_json(input_path_or_url)
        elif ingest_type in ["rss", "api"]:
            result = ingest_rss_or_api(input_path_or_url, source_type=ingest_type)
        else:
            logger.warning(f"[Skip] Unsupported file/source type: {input_path_or_url}")
            return {
                "status": "skipped",
                "reason": "unsupported_format",
                "source": input_path_or_url,
            }

        logger.info(f"[Complete] ✅ Ingestion completed for {input_path_or_url}")
        return {
            "status": "success",
            "type": ingest_type,
            "details": result,
        }

    except Exception as e:
        logger.error(f"[Error] Ingestion failed for {input_path_or_url}: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "source": input_path_or_url,
        }

# ==========================================================
# 📦 BULK INGESTION
# ==========================================================
def bulk_ingest_from_directory(folder_path: str):
    """
    Scans a directory and automatically ingests all recognized files.
    """
    logger.info(f"[BulkStart] Scanning directory: {folder_path}")
    results = []

    for root, _, files in os.walk(folder_path):
        for f in files:
            full_path = os.path.join(root, f)
            result = route_ingestion(full_path)
            results.append(result)

    logger.info(f"[BulkComplete] Finished bulk ingestion for {folder_path}")
    return results

# ==========================================================
# 📅 SCHEDULED RSS / API LOOP
# ==========================================================
def scheduled_rss_ingestion():
    """
    Pulls RSS feeds listed in YAML config and re-ingests on schedule.
    """
    rss_sources = config.get("rss_feeds", [])
    if not rss_sources:
        logger.warning("[RSS] No RSS feeds configured in ingestion_config.yaml")
        return

    for url in rss_sources:
        try:
            logger.info(f"[RSS] Scheduled ingestion for feed: {url}")
            ingest_rss_or_api(url, source_type="rss")
        except Exception as e:
            logger.error(f"[RSS] Failed to ingest {url}: {e}")

# ==========================================================
# 🧪 TEST EXECUTION
# ==========================================================
if __name__ == "__main__":
    import json

    test_files = [
        "data/uploads/Enterprise_Taxonomy_Test.xlsx",
        "data/uploads/Agri_Cold_Storage_Project.pdf",
        "data/uploads/Digital_Marketing_Strategy.docx",
        "data/uploads/Business_Strategy_Notes.txt",
        "data/uploads/EV_Industry_Trends.json",
        "https://feeds.feedburner.com/TechCrunch",
    ]

    results = []
    for f in test_files:
        res = route_ingestion(f)
        results.append(res)

    print(json.dumps(results, indent=2))
    print("✅ Intelligent Auto Ingestion Router completed successfully.")
