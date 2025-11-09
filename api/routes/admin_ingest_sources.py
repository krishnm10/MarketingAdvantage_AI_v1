# ===============================================
# 🚀 admin_ingest_sources.py — Ingestion Monitoring & Feed Analytics
# ===============================================
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from api.db.models import IngestSource
from api.connection import get_db

router = APIRouter(prefix="/admin/ingest_sources", tags=["Admin / Ingestion Monitoring"])


# ==========================================================
# 🔹 GET /admin/ingest_sources — List all feed metrics
# ==========================================================
@router.get("/")
def list_ingest_sources(db: Session = Depends(get_db)):
    """Return current ingestion feed metrics and health status."""
    sources = db.query(IngestSource).order_by(IngestSource.category, IngestSource.last_fetched.desc()).all()
    return [
        {
            "feed_url": s.feed_url,
            "category": s.category,
            "articles_added": s.articles_added,
            "partials": s.partials,
            "failures": s.failures,
            "last_fetched": s.last_fetched,
            "status": s.status,
            "avg_confidence": s.avg_confidence,
            "error_message": s.error_message or "",
        }
        for s in sources
    ]


# ==========================================================
# 🔹 GET /admin/ingest_sources/stats — Aggregated summary
# ==========================================================
@router.get("/stats")
def get_ingest_stats(db: Session = Depends(get_db)):
    """Aggregate stats — success/partial/failure counts & totals."""
    total_sources = db.query(func.count(IngestSource.id)).scalar()
    total_articles = db.query(func.sum(IngestSource.articles_added)).scalar() or 0
    total_partials = db.query(func.sum(IngestSource.partials)).scalar() or 0
    total_failures = db.query(func.sum(IngestSource.failures)).scalar() or 0

    return {
        "total_sources": total_sources,
        "total_articles": total_articles,
        "total_partials": total_partials,
        "total_failures": total_failures,
        "uptime_status": "✅ Healthy" if total_failures < total_sources else "⚠️ Some failing sources"
    }


# ==========================================================
# 🔹 DELETE /admin/ingest_sources/reset — Reset metrics
# ==========================================================
@router.delete("/reset")
def reset_ingest_metrics(db: Session = Depends(get_db)):
    """Delete all ingestion source logs."""
    count = db.query(IngestSource).delete()
    db.commit()
    return {"message": f"✅ {count} ingestion source logs cleared."}


# ==========================================================
# 🔹 PATCH /admin/ingest_sources/retry — Mark source for retry
# ==========================================================
@router.patch("/retry/{feed_url}")
def retry_source(feed_url: str, db: Session = Depends(get_db)):
    """Mark a failed source for retry by resetting its status."""
    source = db.query(IngestSource).filter_by(feed_url=feed_url).first()
    if not source:
        raise HTTPException(status_code=404, detail="Feed not found.")

    source.status = "idle"
    source.error_message = None
    source.failures = 0
    db.commit()
    return {"message": f"🔁 {feed_url} marked for retry."}
