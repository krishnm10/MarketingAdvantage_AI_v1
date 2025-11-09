# ==============================================================
# 🚀 taxonomy_sync.py — Scheduled Taxonomy JSON Sync (Option B)
# ==============================================================

import json, logging
from datetime import datetime
from api.connection import get_db_connection
from api.db.models import TaxonomyCategory

SYNC_FILE = "data/taxonomy_master.json"

def export_taxonomy():
    """Export all taxonomy categories from DB → JSON file."""
    with get_db_connection() as db:
        rows = db.query(TaxonomyCategory).all()
        data = [
            {"id": str(r.id), "name": r.name, "group": r.group, "description": r.description}
            for r in rows
        ]

    with open(SYNC_FILE, "w", encoding="utf-8") as f:
        json.dump({"updated_at": datetime.utcnow().isoformat(), "categories": data}, f, indent=2)

    logging.info(f"[TaxonomySync] ✅ Exported {len(data)} categories → {SYNC_FILE}")
    return len(data)

if __name__ == "__main__":
    export_taxonomy()
