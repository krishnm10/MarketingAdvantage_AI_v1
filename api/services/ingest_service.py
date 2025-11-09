# ==============================================================
# 🚀 ingest_service.py — Web / RSS Smart Ingestion
# ==============================================================

import logging
from datetime import datetime
from api.services.unified_ingest_helper import ingest_smart
from api.services.taxonomy_helper import get_best_taxonomy_match
from api.connection import get_db_connection
from api.db.models import Content

# ==============================================================
# 🧠 Auto-categorize fetched article
# ==============================================================
def auto_categorize_text(text: str):
    match = get_best_taxonomy_match(text)
    if match and match["confidence"] >= 0.99:
        return match["category"], match["subcategory"], match["confidence"]
    logging.warning(f"[AutoCat] RSS/LLM content low confidence; assigning to Uncategorized.")
    return "Uncategorized", "General Business", match.get("confidence", 0) if match else 0

# ==============================================================
# 🌐 Unified ingestion for any fetched article
# ==============================================================
def process_article(article_data: dict):
    """
    Expected article_data:
    {
        "title": "...",
        "content": "...",
        "source": "rss" or "web" or "llm",
        "business_name": "Some Biz",
        "industry": "RetailTech"
    }
    """
    text = article_data.get("content", "")
    category, subcategory, conf = auto_categorize_text(text)

    with get_db_connection() as db:
        content_row = Content(
            title=article_data.get("title", "Untitled"),
            text=text,
            summary=text[:250],
            source=article_data.get("source", "rss"),
            category=category,
            sub_category=subcategory,
            confidence=conf,
            created_at=datetime.utcnow(),
        )
        db.add(content_row)
        db.commit()
        db.refresh(content_row)

        ingest_result = ingest_smart(
            entity_type="content",
            entity_id=str(content_row.id),
            business_name=article_data.get("business_name", "RSS Business"),
            category_name=category,
            subcategory_name=subcategory,
            content_text=text,
            industry=article_data.get("industry", None),
            metadata={"source": article_data.get("source", "rss")},
        )

    logging.info(f"[IngestService] ✅ Article '{content_row.title}' processed.")
    return ingest_result
