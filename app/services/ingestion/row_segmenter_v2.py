# =============================================
# row_segmenter_v2.py — Structured Data Chunker (Production-Ready)
# Fully aligned with DB schema and IngestionServiceV2
# =============================================

from typing import List, Dict, Any, Optional
import pandas as pd
import json
import math
from datetime import datetime

# <<< PATCH: use text_cleaner_v2 (newer version) >>>
from app.utils.text_cleaner_v2 import clean_text
from app.utils.logger import log_info
from app.services.ingestion.segmenter_v2 import make_chunk_dict

# -------------------------------------------------------------------
# MAIN FUNCTION: Parse structured/tabular data into semantic chunks
# -------------------------------------------------------------------
async def parse_dataframe_rows(
    df: pd.DataFrame,
    file_id: str,
    source_type: str,
    db_session=None,
    business_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Converts structured/tabular data (CSV, Excel, JSON array)
    into dedup-aware semantic chunks compatible with GlobalContentIndex.

    - Flattens each row into human-readable text
    - Cleans and normalizes text
    - Deduplicates using GlobalContentIndex (semantic hash)
    - Returns chunks fully aligned with ingestion DB schema
    """

    chunks: List[Dict[str, Any]] = []

    def _sanitize_for_json(obj):
        """
        Recursively replace Python float NaN / ±Inf with None so that
        json.dumps() produces valid JSON that PostgreSQL JSONB will accept.

        ROOT CAUSE: pandas row.to_dict() preserves float('nan') for NULL cells.
        When stored in meta_data (JSONB), Python serializes NaN as the bare
        token  NaN  which is valid Python but NOT valid JSON.
        PostgreSQL raises: invalid input syntax for type json — Token "NaN" is invalid.

        This guard runs on the raw_row dict before it enters metadata so no
        NaN ever reaches the DB serialisation path.
        """
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_for_json(v) for v in obj]
        return obj

    # compute columns once for efficiency and consistency
    columns = list(df.columns) if df is not None else []

    for row_index, row in df.iterrows():
        row_dict = row.to_dict()

        # Flatten row into readable text string safely (handle dicts/lists)
        parts = []
        for col, value in row_dict.items():
            if pd.isna(value):
                continue
            # convert complex structures to json where possible for readability
            try:
                if isinstance(value, (dict, list)):
                    sval = json.dumps(value, ensure_ascii=False)
                else:
                    sval = str(value)
            except Exception:
                sval = str(value)
            parts.append(f"{col}: {sval}")

        row_text = " | ".join(parts)

        if not row_text.strip():
            continue

        # make_chunk_dict is a pure synchronous function — do NOT await it.
        # BUG FIX: was `await make_chunk_dict(...)` which raised
        # TypeError: object dict can't be used in 'await' expression
        # because make_chunk_dict returns a plain dict, not a coroutine.
        chunk_data = make_chunk_dict(
            row_text,
            db_session=db_session,
            file_id=file_id,
            business_id=business_id,
            source_type=source_type,
        )

        # Defensive: ensure chunk_data contains expected keys
        if not chunk_data or not isinstance(chunk_data, dict):
            continue

        chunks.append(
            {
                "file_id":     file_id,
                "source_type": source_type,
                "row_index":   row_index,
                "columns":     columns,

                # Semantic core — type-safe fallbacks on every field.
                # BUG FIX: c.get("tokens") had no int() cast, could store NULL
                # in DB. AgenticValidation does `chunk.tokens > MIN` → TypeError.
                "text":          chunk_data.get("text") or row_text,
                "cleaned_text":  chunk_data.get("cleaned_text") or "",
                "tokens":        int(chunk_data.get("tokens") or 0),
                "semantic_hash": chunk_data.get("semantic_hash") or "",
                "confidence":    float(chunk_data.get("confidence") or 1.0),

                # GlobalContentIndex link
                "global_content_id": chunk_data.get("global_content_id"),

                # BUG FIX: reasoning_ingestion was never passed through from
                # make_chunk_dict. _insert_chunks stored {} in DB for ALL
                # row_segmenter chunks. AgenticValidation reads fields from this
                # JSON (extraction_confidence, origin_authority, etc.) and does
                # None > threshold → TypeError: NoneType > int/float.
                # Fix: pass through from make_chunk_dict, with safe defaults.
                "reasoning_ingestion": chunk_data.get("reasoning_ingestion") or {
                    "signal_type":           "narrative",
                    "business_function":     "general",
                    "time_horizon":          "timeless",
                    "origin_authority":      "primary_source",
                    "extraction_confidence": 0.90,
                    "granularity":           "tactical_detail",
                    "data_lineage_id":       chunk_data.get("semantic_hash") or "",
                    "potentially_regulated": False,
                    "extraction_timestamp":  datetime.utcnow().isoformat() + "Z",
                },

                # Metadata (aligned with DB meta_data JSONB field)
                # BUG FIX: row_dict contains float('nan') for NULL CSV cells
                # (pandas preserves NaN from missing values). json.dumps() writes
                # NaN as the bare token NaN — valid Python, invalid JSON.
                # PostgreSQL JSONB rejects it with:
                #   invalid input syntax for type json — Token "NaN" is invalid
                # Fix: sanitize raw_row through _sanitize_for_json() which
                # recursively replaces NaN/Inf with None before serialisation.
                "metadata": {
                    "row_index": row_index,
                    "columns":   columns,
                    "raw_row":   _sanitize_for_json(row_dict),
                    "dedup": {
                        "semantic_hash":     chunk_data.get("semantic_hash"),
                        "global_content_id": chunk_data.get("global_content_id"),
                    },
                },
            }
        )

    log_info(f"[row_segmenter_v2] Produced {len(chunks)} dedup-aware structured chunks")

    return chunks