# parsers_router_v2.py — Unified Parser Router (Production-Ready)
# Minimal safe fixes: await async ingestors + normalize file_type
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


class ParserRouterV2:

    @staticmethod
    async def parse(file_path_or_url: str, file_type: str):
        """
        Unified router that dynamically delegates ingestion
        to appropriate parser or ingestor based on extension or type.
        Handles: file uploads, web URLs, RSS feeds, and APIs.
        """

        # Normalize file_type: accept ".pdf", "pdf", "application/pdf", etc.
        ext = (file_type or "").lower()
        if "/" in ext:
            # if MIME type like application/pdf -> take subtype
            ext = ext.split("/")[-1]
        ext = ext.lstrip(".").split("?")[0]

        log_info(f"[ParserRouterV2] Selecting parser for {file_path_or_url} ({ext})")

        # ---------------------------
        # FILE-BASED PARSERS
        # ---------------------------
        if ext == "pdf":
            return await parse_pdf(file_path_or_url)

        if ext == "docx":
            return await parse_docx(file_path_or_url)

        if ext == "txt":
            return await parse_text(file_path_or_url)

        if ext == "json":
            return await parse_json(file_path_or_url)

        if ext == "csv":
            return await parse_csv(file_path_or_url)

        if ext in {"xlsx", "xls"}:
            return await parse_excel(file_path_or_url)

        if ext == "xml":
            return await parse_xml(file_path_or_url)

        # ---------------------------
        # WEB / API / RSS INGESTORS
        # ---------------------------
        if ext == "web":
            # ingest_webpage is async — await it
            return await ingest_webpage(file_path_or_url)

        if ext == "rss":
            # parse_rss is async — await it
            return await parse_rss(file_path_or_url)

        if ext == "api":
            # ingest_api_data is async — await it
            return await ingest_api_data(file_path_or_url)

        # ---------------------------
        # DEFAULT FALLBACK
        # ---------------------------
        log_warning(f"[ParserRouterV2] Unsupported file or source type: {ext}")
        return {
            "raw_text": "",
            "cleaned_text": "",
            "chunks": [],
            "metadata": {"source": ext},
            "source_type": ext,
        }
