# ==========================================================
# 🚀 intelligent_json_ingest.py — Intelligent JSON Ingestion (v4.0)
# ==========================================================
"""
Handles ingestion of structured JSON files or API-fetched payloads.
Automatically extracts meaningful text fields, applies semantic
paragraph chunking with recursive fallback, classifies each section
via the enterprise LLM, and stores structured data in PostgreSQL + Chroma.

✅ Parses nested and array-based JSON structures
✅ Identifies text-rich fields automatically
✅ Semantic + Recursive Chunking
✅ Config-driven (from ingestion_config.yaml)
✅ Deduplication + Trace Logging
✅ Unified taxonomy-aware classification

Author: MarketingAdvantage™ AI Core Team
Version: 4.0
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import List, Any, Dict

from api.services.text_chunker import semantic_paragraph_chunking
from api.services.taxonomy_helper import get_best_taxonomy_match
from api.config.ingestion_config import get_ingestion_config
from api.services.llm_classifier import classify_text_with_llm
from api.services.unified_ingest_helper import (
    clean_text_for_postgres,
    compute_fingerprint,
    process_and_store_document,
)

# ==========================================================
# ⚙️ CONFIGURATION
# ==========================================================
config = get_ingestion_config()

active_profile_name = config.get("profiles", {}).get("active", "balanced")
active_profile = config.get("profiles", {}).get(active_profile_name, {})
MODEL_NAME = config["llm"]["model_name"]
MAX_CHUNK_SIZE = config["chunking"]["max_chunk_size"]
RECURSIVE_THRESHOLD = config["chunking"]["recursive_threshold"]
ENABLE_SEMANTIC = config["chunking"]["semantic_chunking"]
LOG_PATH = config["logging"]["file"]

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [JSONIngest] %(message)s"
)
logger = logging.getLogger("intelligent_json_ingest")

# ==========================================================
# 🧩 HELPERS
# ==========================================================
def load_json_file(file_path: str) -> Any:
    """Loads JSON from file safely."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"[Extract] Loaded JSON file: {os.path.basename(file_path)}")
    return data


def extract_text_fields(data: Any) -> List[str]:
    """
    Recursively extracts all text-rich fields from JSON.
    This includes 'title', 'description', 'summary', etc.
    """
    text_segments = []

    def recursive_extract(obj: Any):
        if isinstance(obj, dict):
            for key, val in obj.items():
                # Heuristic: process probable text fields
                if isinstance(val, str) and len(val.strip()) > 20:
                    if re.search(r"(title|desc|summary|content|body|text|detail)", key, re.IGNORECASE):
                        text_segments.append(val.strip())
                elif isinstance(val, (dict, list)):
                    recursive_extract(val)
        elif isinstance(obj, list):
            for item in obj:
                recursive_extract(item)

    recursive_extract(data)
    logger.info(f"[Parse] Extracted {len(text_segments)} candidate text fields from JSON.")
    return text_segments


def semantic_chunk_text(text: str, max_chunk_size: int) -> List[str]:
    """Performs semantic chunking with recursive fallback."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    for p in paragraphs:
        if len(p) <= max_chunk_size:
            chunks.append(p)
        else:
            # Recursive fallback (sentence-level split)
            sentences = re.split(r"(?<=[.!?])\s+", p)
            temp_chunk = ""
            for s in sentences:
                if len(temp_chunk) + len(s) <= max_chunk_size:
                    temp_chunk += s + " "
                else:
                    chunks.append(temp_chunk.strip())
                    temp_chunk = s + " "
            if temp_chunk.strip():
                chunks.append(temp_chunk.strip())

    logger.info(f"[Chunking] Generated {len(chunks)} semantic chunks.")
    return chunks


# ==========================================================
# 🚀 MAIN INGEST FUNCTION
# ==========================================================
def ingest_json(file_path: str):
    """
    Main intelligent ingestion entry point for JSON files.
    - Extracts text fields
    - Performs semantic chunking
    - Classifies and stores structured content
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    logger.info(f"[Start] Intelligent JSON ingestion started for {file_path}")
    data = load_json_file(file_path)

    text_blocks = extract_text_fields(data)
    if not text_blocks:
        logger.warning(f"[Skip] No text-rich fields found in {file_path}")
        return []

    results = []

    for i, block in enumerate(text_blocks, start=1):
        # Semantic + recursive chunking per text field
        chunks = semantic_chunk_text(block, MAX_CHUNK_SIZE)

        for j, chunk in enumerate(chunks, start=1):
            try:
                cleaned = clean_text_for_postgres(chunk)
                classification = classify_text_with_llm(cleaned)

                category = classification.get("category_level_1", "Unclassified")
                subcategory = classification.get("category_level_2_sub", "Ambiguous Content")
                confidence = classification.get("extraction_confidence", 0.0)

                record = process_and_store_document(
                    text=cleaned,
                    file_name=os.path.basename(file_path),
                    source="json",
                    category=category,
                    subcategory=subcategory,
                    confidence=confidence,
                )

                results.append({
                    "field": i,
                    "chunk": j,
                    "category": category,
                    "subcategory": subcategory,
                    "confidence": confidence,
                    "status": record["status"],
                })

                logger.info(f"[ChunkSuccess] {file_path} | Field {i}, Chunk {j}: {category} > {subcategory}")

            except Exception as e:
                logger.error(f"[ChunkError] {file_path} | Field {i}, Chunk {j} failed: {e}")

    logger.info(f"[Complete] JSON ingestion finished for {file_path}")
    return results


# ==========================================================
# 🧪 TEST EXECUTION
# ==========================================================
if __name__ == "__main__":
    test_file = os.path.join(os.getcwd(), "data", "uploads", "EV_Industry_Trends.json")
    if os.path.exists(test_file):
        output = ingest_json(test_file)
        import json
        print(json.dumps(output, indent=2))
        print("✅ Intelligent JSON ingestion completed successfully.")
    else:
        print(f"⚠️ Test file not found: {test_file}")
