# parsers_router_v2.py — Unified Parser Router (Production-Ready)
# Gap-3 fix: replaced 10 individual if-statements with PARSER_MAP dict.
# Zero behavior change — same parsers, same fallback, same async handling.
from app.utils.logger import log_info, log_warning

from app.services.ingestion.pdf_parser_v2 import parse_pdf
from app.services.ingestion.text_parser_v2 import parse_text
from app.services.ingestion.json_parser_v2 import parse_json
from app.services.ingestion.csv_parser_v2 import parse_csv
from app.services.ingestion.excel_parser_v2 import parse_excel
from app.services.ingestion.xml_parser_v2 import parse_xml
from app.services.ingestion.docx_parser_v2 import parse_docx
from app.services.ingestion.web_scraper_v2 import ingest_webpage
from app.services.ingestion.api_ingestor_v2 import ingest_api_data
from app.services.ingestion.rss_ingestor_v2 import parse_rss

# ---------------------------------------------------------------------------
# PARSER MAP — single source of truth for all supported types.
# Adding a new format = 1 import above + 1 line here. Nothing else changes.
# ---------------------------------------------------------------------------
PARSER_MAP = {
    "pdf":  parse_pdf,
    "docx": parse_docx,
    "txt":  parse_text,
    "json": parse_json,
    "csv":  parse_csv,
    "xlsx": parse_excel,
    "xls":  parse_excel,
    "xml":  parse_xml,
    "web":  ingest_webpage,
    "rss":  parse_rss,
    "api":  ingest_api_data,
}


class ParserRouterV2:

    @staticmethod
    async def parse(file_path_or_url: str, file_type: str):
        """
        Unified router that dynamically delegates ingestion
        to the appropriate parser or ingestor based on extension or type.
        Handles: file uploads, web URLs, RSS feeds, and APIs.

        Adding a new file type:
            1. Create app/services/ingestion/<type>_parser_v2.py
            2. Import the parse function above
            3. Add one entry to PARSER_MAP
            Done. This method never needs to change.
        """

        # Normalize file_type: accept ".pdf", "pdf", "application/pdf", etc.
        ext = (file_type or "").lower()
        if "/" in ext:
            # MIME type like "application/pdf" → take subtype "pdf"
            ext = ext.split("/")[-1]
        ext = ext.lstrip(".").split("?")[0]

        log_info(f"[ParserRouterV2] Selecting parser for {file_path_or_url} ({ext})")

        parser_func = PARSER_MAP.get(ext)

        if not parser_func:
            log_warning(f"[ParserRouterV2] Unsupported file or source type: {ext}")
            return {
                "raw_text":     "",
                "cleaned_text": "",
                "chunks":       [],
                "metadata":     {"source": ext},
                "source_type":  ext,
            }

        return await parser_func(file_path_or_url)
