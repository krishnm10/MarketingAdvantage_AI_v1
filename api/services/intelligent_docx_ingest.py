# ==========================================================
# 🚀 intelligent_docx_ingest.py — Intelligent DOCX Ingestion (v4.0)
# ==========================================================
"""
Reads Word (.docx) files, extracts structured text content,
performs semantic paragraph chunking with recursive fallback,
classifies via the LLM (Llama or DeepSeek), and stores structured
results into PostgreSQL + ChromaDB.

✅ Semantic + Recursive Chunking
✅ Taxonomy & Business-aware Classification
✅ Config-driven ingestion (from ingestion_config.yaml)
✅ Fingerprint deduplication + rollback-safe transactions
✅ Logs all classification traces and results

Author: MarketingAdvantage™ AI Core Team
Version: 4.0
"""

import os
import re
import logging
from datetime import datetime
from typing import List
from docx import Document

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
CHROMA_ENABLED = True
ENABLE_SEMANTIC = config["chunking"]["semantic_chunking"]
LOG_PATH = config["logging"]["file"]

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [DOCXIngest] %(message)s"
)
logger = logging.getLogger("intelligent_docx_ingest")

# ==========================================================
# 🧩 TEXT EXTRACTION HELPERS
# ==========================================================
def extract_text_from_docx(file_path: str) -> str:
    """Extracts readable text from a Word (.docx) file."""
    doc = Document(file_path)
    text_blocks = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            text_blocks.append(text)
    raw_text = "\n".join(text_blocks)
    logger.info(f"[Extract] Extracted {len(raw_text)} chars from {os.path.basename(file_path)}")
    return raw_text


def semantic_chunk_text(text: str, max_chunk_size: int) -> List[str]:
    """
    Splits DOCX text semantically into paragraphs and sentences.
    Includes recursive fallback for long text blocks.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []

    for p in paragraphs:
        if len(p) <= max_chunk_size:
            chunks.append(p)
        else:
            # Recursive fallback — split by sentence
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

    logger.info(f"[Chunking] Generated {len(chunks)} semantic chunks from DOCX.")
    return chunks


# ==========================================================
# 🚀 MAIN INGEST FUNCTION
# ==========================================================
def ingest_docx(file_path: str):
    """
    Main intelligent ingestion entry point for DOCX files.
    Performs extraction, chunking, LLM classification, and DB storage.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DOCX file not found: {file_path}")

    logger.info(f"[Start] Intelligent DOCX ingestion started for {file_path}")
    raw_text = extract_text_from_docx(file_path)
    if not raw_text:
        logger.warning(f"[Skip] Empty DOCX: {file_path}")
        return []

    # Semantic + recursive chunking
    chunks = semantic_chunk_text(raw_text, MAX_CHUNK_SIZE)
    results = []

    for i, chunk in enumerate(chunks, start=1):
        try:
            cleaned = clean_text_for_postgres(chunk)
            classification = classify_text_with_llm(cleaned)

            category = classification.get("category_level_1", "Unclassified")
            subcategory = classification.get("category_level_2_sub", "Ambiguous Content")
            confidence = classification.get("extraction_confidence", 0.0)

            record = process_and_store_document(
                text=cleaned,
                file_name=os.path.basename(file_path),
                source="docx",
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

            logger.info(f"[ChunkSuccess] {file_path} | Chunk {i}: {category} > {subcategory}")

        except Exception as e:
            logger.error(f"[ChunkError] {file_path} | Chunk {i} failed: {e}")

    logger.info(f"[Complete] DOCX ingestion finished for {file_path}")
    return results


# ==========================================================
# 🧪 TEST EXECUTION
# ==========================================================
if __name__ == "__main__":
    test_file = os.path.join(os.getcwd(), "data", "uploads", "Digital_Marketing_Strategy.docx")
    if os.path.exists(test_file):
        output = ingest_docx(test_file)
        import json
        print(json.dumps(output, indent=2))
        print("✅ Intelligent DOCX ingestion completed successfully.")
    else:
        print(f"⚠️ Test file not found: {test_file}")
