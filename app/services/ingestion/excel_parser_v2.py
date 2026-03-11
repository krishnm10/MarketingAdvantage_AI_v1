# =============================================
# excel_parser_v2.py — Multi-Sheet Excel Parser (Production-Ready)
# Now includes unified LLM toggle control (local + global)
# Fully aligned with ingestion_v2 pipeline and row_segmenter_v2
# =============================================

import pandas as pd
import asyncio
import os
import time
from typing import Dict, Any, List, Optional

from app.services.ingestion.row_segmenter_v2 import parse_dataframe_rows
from app.services.ingestion.llm_rewriter import rewrite_batch
from app.utils.logger import log_info, log_warning
from app.config.ingestion_settings import ENABLE_LLM_NORMALIZATION  # ✅ Global toggle

# -------------------------------------------------------------------
# Local parser-level toggle
# -------------------------------------------------------------------
# True  → Force enable LLM normalization for this parser
# False → Force disable LLM normalization for this parser
# None  → Inherit from global flag
LOCAL_LLM_TOGGLE = None

def is_llm_enabled() -> bool:
    """Returns the effective LLM toggle for this parser."""
    return ENABLE_LLM_NORMALIZATION if LOCAL_LLM_TOGGLE is None else LOCAL_LLM_TOGGLE


# -------------------------------------------------------------------
# FILE READER
# -------------------------------------------------------------------
def try_read_excel(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Attempts to read all sheets from an Excel file safely.
    Returns a dict of { sheet_name: DataFrame }
    Handles merged cells, empty sheets, and large workbooks gracefully.
    """
    max_retries = 3
    retry_delay_sec = 1.0
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            return pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            is_zip_error = "file is not a zip file" in msg

            # Common with watcher-based ingestion when file write is not fully flushed yet.
            if is_zip_error and attempt < max_retries:
                time.sleep(retry_delay_sec)
                continue

            # Fallback: some users upload CSV data with .xlsx extension.
            if is_zip_error:
                try:
                    df = pd.read_csv(file_path)
                    log_warning(
                        f"[excel_parser_v2] '{file_path}' is not a valid xlsx zip. "
                        "Parsed as CSV fallback."
                    )
                    return {"Sheet1": df}
                except Exception:
                    pass

            break

    raise ValueError(
        f"[excel_parser_v2] Failed to read Excel file: {file_path}: {last_error}"
    )


# -------------------------------------------------------------------
# DATA NORMALIZER
# -------------------------------------------------------------------
def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame before segmentation:
      - strip column names
      - fill NaNs with None
      - trim whitespace from string columns
    """
    df.columns = [str(col).strip() for col in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    df = df.where(pd.notnull(df), None)
    return df


# -------------------------------------------------------------------
# MAIN PARSER PIPELINE
# -------------------------------------------------------------------
async def parse_excel(
    file_path: str,
    file_id: Optional[str] = None,
    business_id: Optional[str] = None,
    db_session=None,
) -> Dict[str, Any]:
    """
    Multi-sheet Excel parser with async-safe segmentation and optional LLM normalization.
    Steps:
      1. Load all sheets
      2. Normalize data
      3. Segment each sheet using row_segmenter_v2
      4. Optionally normalize chunk texts with LLM
      5. Combine all chunks and metadata
    """

    log_info(f"[excel_parser_v2] Reading Excel: {file_path}")

    try:
        sheets = await asyncio.to_thread(try_read_excel, file_path)
    except Exception as e:
        log_warning(f"[excel_parser_v2] Error reading Excel: {e}")
        raise ValueError(f"Failed to read Excel: {e}")

    if not sheets:
        raise ValueError(f"[excel_parser_v2] No sheets found in {file_path}")

    all_chunks: List[Dict[str, Any]] = []
    metadata_sheets = []

    for sheet_name, df in sheets.items():
        if df.empty:
            log_warning(f"[excel_parser_v2] Empty sheet skipped: {sheet_name}")
            continue

        df = normalize_dataframe(df)
        log_info(f"[excel_parser_v2] Processing sheet '{sheet_name}' with {len(df)} rows.")

        try:
            sheet_chunks = await parse_dataframe_rows(
                df=df,
                file_id=file_id,
                source_type="excel",
                db_session=db_session,
                business_id=business_id,
            )
        except Exception as e:
            log_warning(f"[excel_parser_v2] Segmentation failed on {sheet_name}: {e}")
            continue

        if not sheet_chunks:
            continue

        all_chunks.extend(sheet_chunks)
        metadata_sheets.append(
            {
                "sheet_name": sheet_name,
                "rows": len(df),
                "columns": list(df.columns),
            }
        )

    if not all_chunks:
        raise ValueError(f"[excel_parser_v2] No valid chunks extracted from {file_path}")

    # ✅ Step 4: Optional LLM normalization (batch-based)
    if is_llm_enabled():
        try:
            texts = [chunk.get("text", "") for chunk in all_chunks if chunk.get("text")]
            if texts:
                log_info(f"[excel_parser_v2] Sending {len(texts)} chunks for LLM normalization...")
                normalized_texts = await rewrite_batch(texts)
                for i, chunk in enumerate(all_chunks):
                    if i < len(normalized_texts):
                        chunk["normalized_text"] = normalized_texts[i]
            else:
                log_warning("[excel_parser_v2] No valid text chunks for normalization.")
        except Exception as e:
            log_warning(f"[excel_parser_v2] ⚠️ LLM normalization failed: {e}")
    else:
        log_info("[excel_parser_v2] LLM normalization skipped (disabled).")

    log_info(f"[excel_parser_v2] Parsed {len(all_chunks)} total chunks across {len(metadata_sheets)} sheets.")

    return {
        "sheet_count": len(sheets),
        "chunks": all_chunks,
        "source_type": "excel",
        "metadata": {
            "file_name": os.path.basename(file_path),
            "sheets": metadata_sheets,
            "parser": "excel_v2 (pandas/openpyxl)",
        },
    }
