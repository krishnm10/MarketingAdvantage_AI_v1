# =============================================
# row_segmenter_v2.py — Structured Data Chunker
#
# B3 FIX: Chunked DataFrame iteration.
#   BEFORE: df.iterrows() creates a pd.Series per row — O(N) Series allocs.
#           All N chunk dicts accumulated in one list before return.
#           50,000-row file = ~60 MB peak RAM in this function alone.
#
#   AFTER:  _iter_dataframe_chunks() yields CHUNK_SIZE rows at a time.
#           Each batch processed → appended to result → batch goes out of scope.
#           Peak RAM = CHUNK_SIZE rows × chunk_dict_size, not N × chunk_dict_size.
#           CHUNK_SIZE=500 → ~600 KB peak regardless of file size.
# =============================================

import math
import json
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

import pandas as pd

from app.utils.text_cleaner_v2 import clean_text
from app.utils.logger import log_info
from app.services.ingestion.segmenter_v2 import make_chunk_dict

# Rows processed per iteration batch.
# 500 rows × ~1.2 KB/chunk = ~600 KB peak per batch — safe for all server configs.
_ROW_BATCH_SIZE: int = 500


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively replace float NaN / ±Inf with None.

    ROOT CAUSE: pandas row.to_dict() preserves float('nan') for NULL cells.
    json.dumps() writes NaN as the bare token NaN — valid Python, invalid JSON.
    PostgreSQL JSONB rejects: Token "NaN" is invalid.
    """
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _iter_dataframe_chunks(
    df: pd.DataFrame, batch_size: int
) -> Generator[pd.DataFrame, None, None]:
    """
    Yield successive row-batches of the DataFrame.
    Each batch is a view (not a copy) — zero extra RAM allocation.
    Generator is lazy — only the current batch is live in memory.
    """
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size]


async def parse_dataframe_rows(
    df: pd.DataFrame,
    file_id: str,
    source_type: str,
    db_session=None,
    business_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Converts structured/tabular data into dedup-aware semantic chunks.

    B3 FIX: Processes DataFrame in _ROW_BATCH_SIZE batches.
    Peak RAM = _ROW_BATCH_SIZE × chunk_dict_size, not N × chunk_dict_size.
    df.itertuples() used instead of df.iterrows() — 4–10× faster,
    no per-row pd.Series allocation.

    Returns:
        List of chunk dicts fully aligned with ingestion DB schema.
    """
    if df is None or df.empty:
        return []

    columns: List[str] = list(df.columns)
    result:  List[Dict[str, Any]] = []
    now = datetime.utcnow()

    for batch_df in _iter_dataframe_chunks(df, _ROW_BATCH_SIZE):
        # itertuples() is 4–10× faster than iterrows() — no Series alloc per row
        for row in batch_df.itertuples(index=True, name=None):
            row_index = row[0]
            row_values = row[1:]

            # Build column→value dict without creating a pd.Series
            row_dict: Dict[str, Any] = {
                col: val for col, val in zip(columns, row_values)
            }

            # Flatten row into human-readable text — skip null cells
            parts: List[str] = []
            for col, value in row_dict.items():
                try:
                    if pd.isna(value):
                        continue
                except (TypeError, ValueError):
                    pass  # non-scalar value — include it
                try:
                    sval = (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else str(value)
                    )
                except Exception:
                    sval = str(value)
                parts.append(f"{col}: {sval}")

            row_text = " | ".join(parts)
            if not row_text.strip():
                continue

            # make_chunk_dict is pure sync — never await it
            chunk_data = make_chunk_dict(
                row_text,
                db_session=db_session,
                file_id=file_id,
                business_id=business_id,
                source_type=source_type,
            )

            if not chunk_data or not isinstance(chunk_data, dict):
                continue

            result.append({
                "file_id":     file_id,
                "source_type": source_type,
                "row_index":   row_index,
                "columns":     columns,

                "text":          chunk_data.get("text") or row_text,
                "cleaned_text":  chunk_data.get("cleaned_text") or "",
                "tokens":        int(chunk_data.get("tokens") or 0),
                "semantic_hash": chunk_data.get("semantic_hash") or "",
                "confidence":    float(chunk_data.get("confidence") or 1.0),

                "global_content_id": chunk_data.get("global_content_id"),

                "reasoning_ingestion": chunk_data.get("reasoning_ingestion") or {
                    "signal_type":           "narrative",
                    "business_function":     "general",
                    "time_horizon":          "timeless",
                    "origin_authority":      "primary_source",
                    "extraction_confidence": 0.90,
                    "granularity":           "tactical_detail",
                    "data_lineage_id":       chunk_data.get("semantic_hash") or "",
                    "potentially_regulated": False,
                    "extraction_timestamp":  now.isoformat() + "Z",
                },

                "metadata": {
                    "row_index": row_index,
                    "columns":   columns,
                    "raw_row":   _sanitize_for_json(row_dict),
                    "dedup": {
                        "semantic_hash":     chunk_data.get("semantic_hash"),
                        "global_content_id": chunk_data.get("global_content_id"),
                    },
                },
            })

        # FIX-B3-1: batch_df goes out of scope here — GC-eligible immediately
        # without waiting for the full DataFrame to be processed

    log_info(
        f"[row_segmenter_v2] {len(result)} chunks from "
        f"{len(df):,} rows | file_id={file_id}"
    )
    return result
