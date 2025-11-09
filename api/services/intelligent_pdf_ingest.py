# ==========================================================
# 🚀 intelligent_pdf_ingest.py — Intelligent PDF Ingestion (v4.0)
# ==========================================================
"""
Reads PDF files, extracts clean text using semantic paragraph
chunking with recursive fallback, classifies each chunk via LLM,
and stores structured results in PostgreSQL + ChromaDB.

✅ Semantic paragraph chunking (sentence-based)
✅ Recursive fallback for large paragraphs
✅ YAML-configurable model + chunking
✅ Deduplication via fingerprinting
✅ Unified taxonomy linking
✅ Logging + tracing for QA

Author: MarketingAdvantage™ AI Core Team
Version: 4.0
"""

import os
import re
import fitz  # PyMuPDF
import logging
from datetime import datetime
from typing import List

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
    format="%(asctime)s [%(levelname)s] [PDFIngest] %(message)s"
)
logger = logging.getLogger("intelligent_pdf_ingest")

# ==========================================================
# 🧩 TEXT EXTRACTION HELPERS
# ==========================================================
def extract_text_from_pdf(file_path: str) -> str:
    """Extracts raw text from PDF using PyMuPDF."""
    text = []
    with fitz.open(file_path) as doc:
        for page in doc:
            text.append(page.get_text("text"))
    raw = "\n".join(text)
    logger.info(f"[Extract] Extracted {len(raw)} chars from {os.path.basename(file_path)}")
    return raw.strip()


def semantic_chunk_text(text: str, max_chunk_size: int) -> List[str]:
    """
    Splits text semantically (paragraph/sentence-based) with recursive fallback.
    - Uses sentence and paragraph heuristics.
    - Falls back to recursive split if still too large.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []

    for p in paragraphs:
        if len(p) <= max_chunk_size:
            chunks.append(p)
        else:
            # Recursive fallback — sentence-level splitting
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
def ingest_pdf(file_path: str):
    """
    Main intelligent ingestion entry point for PDFs.
    - Extracts text
    - Chunks semantically
    - Classifies via LLM
    - Stores in DB + Chroma
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    logger.info(f"[Start] Intelligent PDF ingestion started for {file_path}")
    raw_text = extract_text_from_pdf(file_path)
    if not raw_text:
        logger.warning(f"[Skip] Empty PDF: {file_path}")
        return []

    # Semantic chunking with fallback
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
                file_name=os.path.basename(file_path),
                source="pdf",
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

    logger.info(f"[Complete] PDF ingestion finished for {file_path}")
    return results


# ==========================================================
# 🧪 TEST EXECUTION
# ==========================================================
if __name__ == "__main__":
    test_file = os.path.join(os.getcwd(), "data", "uploads", "Agri_Cold_Storage_Project.pdf")
    if os.path.exists(test_file):
        output = ingest_pdf(test_file)
        import json
        print(json.dumps(output, indent=2))
        print("✅ Intelligent PDF ingestion completed successfully.")
    else:
        print(f"⚠️ Test file not found: {test_file}")
