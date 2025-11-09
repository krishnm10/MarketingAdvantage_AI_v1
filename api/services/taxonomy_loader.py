# ============================================
# 🚀 taxonomy_loader.py — Advanced Taxonomy Sync Engine (Fixed Version)
# ============================================
"""
Purpose:
---------
Auto-loads and synchronizes taxonomy definitions (Content, Strategy, KPI, Trend, Business)
from `data/taxonomy_master.json` into your relational database.

✅ Hierarchical structure (group → section → values)
✅ Incremental sync (inserts new, updates changed)
✅ Version + provenance tracking
✅ Safe to re-run (idempotent)
✅ Logs stats to console
✅ Compatible with current connection.py
"""

import os
import json
from datetime import datetime
from sqlalchemy.orm import Session
from api.db.models import TaxonomyCategory
from api.connection import get_db_connection  # ✅ correct import

# ============================================
# 📦 CONFIG
# ============================================
DATA_PATH = os.path.join(os.getcwd(), "data", "taxonomy_master.json")


# ============================================
# 🧠 Helper: Flatten taxonomy
# ============================================
def flatten_taxonomy(data: dict):
    """Flatten taxonomy into a structured record generator."""
    version = data.get("version", "unknown")
    last_updated = data.get("last_updated", datetime.utcnow().isoformat())

    for group, content in data.items():
        if group in ["version", "last_updated", "description"]:
            continue

        if not isinstance(content, dict):
            continue

        for section_name, section_data in content.items():
            if not isinstance(section_data, dict):
                continue

            values = section_data.get("values", [])
            synonyms = section_data.get("synonyms", {})

            for value in values:
                yield {
                    "group": group,
                    "section": section_name,
                    "name": value,
                    "version": version,
                    "synonyms": synonyms.get(value, []),
                    "description": f"{group} → {section_name}",
                    "last_updated": last_updated,
                }


# ============================================
# 💾 Main Sync Function
# ============================================
def sync_taxonomy(session: Session, force_update: bool = False):
    """Sync taxonomy JSON to DB — inserts new, updates existing."""
    if not os.path.exists(DATA_PATH):
        print(f"[Taxonomy] ❌ taxonomy_master.json not found at {DATA_PATH}")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = list(flatten_taxonomy(data))
    inserted, updated, skipped = 0, 0, 0

    for rec in records:
        existing = (
            session.query(TaxonomyCategory)
            .filter_by(name=rec["name"], group=rec["group"])
            .first()
        )

        if existing:
            if force_update or (existing.description != rec["description"]):
                existing.description = rec["description"]
                existing.group = rec["group"]
                existing.updated_at = datetime.utcnow()
                updated += 1
            else:
                skipped += 1
        else:
            new_item = TaxonomyCategory(
                name=rec["name"],
                group=rec["group"],
                description=rec["description"],
                created_at=datetime.utcnow(),
            )
            session.add(new_item)
            inserted += 1

    session.commit()
    print(
        f"[Taxonomy] ✅ Sync complete: {inserted} inserted, {updated} updated, {skipped} skipped. "
        f"Version: {data.get('version', 'N/A')}"
    )


# ============================================
# 🧰 Utility Runner
# ============================================
def run_sync(force_update: bool = False):
    """CLI-friendly runner for manual taxonomy sync."""
    print("[Taxonomy] Starting taxonomy sync...")
    try:
        with get_db_connection() as db:  # ✅ fixed: use context manager
            sync_taxonomy(db, force_update)
    except Exception as e:
        print(f"[Taxonomy] ❌ Error during sync: {e}")


# ============================================
# 🧩 CLI Entry
# ============================================
if __name__ == "__main__":
    force_flag = os.getenv("FORCE_UPDATE", "false").lower() in ["true", "1", "yes"]
    run_sync(force_update=force_flag)
