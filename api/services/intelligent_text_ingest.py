# ==========================================================
# 🚀 intelligent_text_ingest.py — Intelligent Plain Text Ingestion (v4.0)
# ==========================================================
"""
Handles ingestion of plain text (.txt) files or raw string input.
Performs semantic paragraph chunking with recursive fallback,
classifies via the enterprise LLM, and stores in PostgreSQL + ChromaDB.

✅ Handles .txt or API-fed text
✅ Semantic + Recursive Chunking
✅ Config-driven (from ingestion_config.yaml)
✅ Taxonomy & Business classification
✅ Fingerprint deduplication
✅ Logging for QA & Traceability

Author: MarketingAdvantage™ AI Core Team
Version: 4.0
"""

import os
import re
import logging
from datetime import datetime
from typing import List

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
    format="%(asctime)s [%(levelname)s] [TextIngest] %(message)s"
)
logger = logging.getLogger("intelligent_text_ingest")

# ==========================================================
# 🧩 TEXT CHUNKING HELPERS
# ==========================================================
def read_text_file(file_path: str) -> str:
    """Reads text from .txt file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    logger.info(f"[Extract] Extracted {len(text)} chars from {os.path.basename(file_path)}")
    return text


def semantic_chunk_text(text: str, max_chunk_size: int) -> List[str]:
    """
    Performs semantic chunking with recursive fallback for .txt files or raw text.
    """
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

    logger.info(f"[Chunking] Generated {len(chunks)} text chunks.")
    return chunks


# ==========================================================
# 🚀 MAIN INGEST FUNCTION
# ==========================================================
def ingest_text(source_input: str, file_mode: bool = True):
    """
    Main ingestion entry point for text data.
    - If file_mode=True → reads from file
    - Else → treats source_input as raw text
    """
    if file_mode:
        if not os.path.exists(source_input):
            raise FileNotFoundError(f"Text file not found: {source_input}")
        logger.info(f"[Start] Intelligent text ingestion started for {source_input}")
        raw_text = read_text_file(source_input)
        file_name = os.path.basename(source_input)
    else:
        logger.info("[Start] Intelligent ingestion started for raw text input.")
        raw_text = source_input.strip()
        file_name = "raw_text_input.txt"

    if not raw_text:
        logger.warning(f"[Skip] Empty text input: {file_name}")
        return []

    # Semantic + Recursive Chunking
    chunks = semantic_chunk_text(raw_text, MAX_CHUNK_SIZE)
    results = []

    for i, chunk in enumerate(chunks, start=1):
        try:
            cleaned = clean_text_for_postgres(chunk)
            classification = classify_text_with_llm(cleaned)

            category = classification.get("category_level_1", "Unclassified")
            subcategory = classification.get("category_level_2_sub", "Ambiguous Content")
            confidence = classification.get("extraction_confidence", 0.0)

            # Store structured content
            record = process_and_store_document(
                text=cleaned,
                file_name=file_name,
                source="text",
                category=category,
                subcategory=subcategory,
                confidence=confidence,
            )

            results.append({
                "chunk": i,
                "category": category,
                "subcategory": subcategory,
                "confidence": confidence,
                "status": record["status"],
            })

            logger.info(f"[ChunkSuccess] {file_name} | Chunk {i}: {category} > {subcategory}")

        except Exception as e:
            logger.error(f"[ChunkError] {file_name} | Chunk {i} failed: {e}")

    logger.info(f"[Complete] Text ingestion finished for {file_name}")
    return results


# ==========================================================
# 🧪 TEST EXECUTION
# ==========================================================
if __name__ == "__main__":
    test_file = os.path.join(os.getcwd(), "data", "uploads", "Business_Strategy_Notes.txt")
    if os.path.exists(test_file):
        output = ingest_text(test_file)
        import json
        print(json.dumps(output, indent=2))
        print("✅ Intelligent text ingestion completed successfully.")
    else:
        print(f"⚠️ Test file not found: {test_file}")

    # Test with direct raw input
    sample_text = "The company plans to expand into renewable energy consulting for urban projects."
    result = ingest_text(sample_text, file_mode=False)
    print("✅ Raw text ingestion:", result)
