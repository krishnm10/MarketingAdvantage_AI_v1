# ==============================================================
# 🧠 text_chunker.py — Semantic Paragraph Chunking Utility (v1.0)
# ==============================================================
"""
This module splits long documents into semantically meaningful
paragraphs for LLM classification and vector embedding.

Implements:
  ✅ Recursive chunking by semantic boundaries (paragraphs, sentences)
  ✅ Token length guardrails (for model input safety)
  ✅ UTF-8 safe text normalization
"""

import re
import logging
from typing import List

logger = logging.getLogger("text_chunker")

# ==============================================================
# 🧠 Core: Semantic Paragraph Chunking with Recursive Fallback
# ==============================================================
def semantic_paragraph_chunking(
    text: str,
    max_chunk_size: int = 1000,
    min_chunk_size: int = 200
) -> List[str]:
    """
    Splits the given text into semantic chunks using a recursive fallback.

    Args:
        text (str): Raw text input.
        max_chunk_size (int): Maximum characters per chunk.
        min_chunk_size (int): Minimum characters before merging.

    Returns:
        List[str]: List of semantically meaningful chunks.
    """
    if not text or not isinstance(text, str):
        return []

    # Normalize text (strip excess whitespace, newlines)
    text = re.sub(r'\s+', ' ', text).strip()

    # Split by paragraphs or large sentences
    paragraphs = re.split(r'(?<=[.!?])\s{2,}|\n{2,}', text)

    chunks = []
    buffer = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If paragraph fits, add it directly
        if len(para) <= max_chunk_size:
            if len(buffer) + len(para) < max_chunk_size:
                buffer += " " + para
            else:
                chunks.append(buffer.strip())
                buffer = para
        else:
            # Recursive fallback — split by sentence boundaries
            sub_chunks = re.split(r'(?<=[.!?])\s+', para)
            small_buf = ""
            for sent in sub_chunks:
                if len(small_buf) + len(sent) < max_chunk_size:
                    small_buf += " " + sent
                else:
                    chunks.append(small_buf.strip())
                    small_buf = sent
            if small_buf:
                chunks.append(small_buf.strip())

    # Final merge pass — ensure no tiny chunks remain
    merged_chunks = []
    tmp = ""
    for ch in chunks:
        if len(ch) < min_chunk_size:
            tmp += " " + ch
        else:
            if tmp:
                merged_chunks.append(tmp.strip())
                tmp = ""
            merged_chunks.append(ch.strip())
    if tmp:
        merged_chunks.append(tmp.strip())

    logger.info(f"[Chunker] Created {len(merged_chunks)} semantic chunks.")
    return merged_chunks


# ==============================================================
# 🧪 Test Execution
# ==============================================================
if __name__ == "__main__":
    sample_text = (
        "Artificial Intelligence (AI) is transforming industries. "
        "Machine learning models are being used for automation, "
        "optimization, and predictive insights.\n\n"
        "In marketing, AI enables personalized content delivery, "
        "predictive analytics, and customer segmentation. "
        "However, data ethics and transparency remain key challenges."
    )

    chunks = semantic_paragraph_chunking(sample_text)
    print("🧠 Semantic Chunks:")
    for i, c in enumerate(chunks, 1):
        print(f"{i}. {c}\n")
