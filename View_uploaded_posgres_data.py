# ================================================
# 🧠 view_postgres_data.py — Inspect Postgres Contents Table
# ================================================
from api.connection import get_db_connection
from api.db.models import Content

with get_db_connection() as db:
    print("📊 Total content records:", db.query(Content).count())

    recent = db.query(Content).order_by(Content.created_at.desc()).limit(5).all()
    print("\n🧾 Recent Entries:")
    for r in recent:
        print(f"🪪 ID: {r.id}")
        print(f"📄 Title: {r.title}")
        print(f"📁 Category: {r.category} / {r.sub_category}")
        print(f"🏷️ Tags: {r.tags}")
        print(f"📅 Created: {r.created_at}")
        print("-" * 60)
