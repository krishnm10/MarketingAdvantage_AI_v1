# ================================================
# 🧩 taxonomy_helper.py — Unified Taxonomy Sync Helper
# ================================================
"""
Handles intelligent taxonomy detection, creation, and synchronization between
taxonomy_master.json and the Postgres taxonomy_categories table.
"""

import os
import json
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from api.db.models import TaxonomyCategory
from api.connection import get_db_connection

logger = logging.getLogger("taxonomy_helper")

TAXONOMY_JSON_PATH = os.path.join(os.getcwd(), "data", "taxonomy_master.json")

# ------------------------------------------------
# Utility: Load & Save JSON
# ------------------------------------------------
def load_master_taxonomy() -> dict:
    if not os.path.exists(TAXONOMY_JSON_PATH):
        return {}
    with open(TAXONOMY_JSON_PATH, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)

def save_master_taxonomy(data: dict):
    data["last_updated"] = datetime.utcnow().isoformat()
    with open(TAXONOMY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ------------------------------------------------
# Inference Logic
# ------------------------------------------------
def infer_taxonomy(text: str):
    lower_text = text.lower()
    if "marketing" in lower_text:
        return ("Content", "Marketing")
    if "ai" in lower_text or "automation" in lower_text:
        return ("Technology", "AI & Automation")
    if "finance" in lower_text or "investment" in lower_text:
        return ("Business", "Finance")
    if "manufacturing" in lower_text or "supply chain" in lower_text:
        return ("Industry", "Manufacturing")
    if "health" in lower_text or "medical" in lower_text:
        return ("Healthcare", "HealthTech")
    if "education" in lower_text:
        return ("Education", "EdTech")
    if "agriculture" in lower_text or "farming" in lower_text:
        return ("Agriculture", "AgriTech")
    return ("General", "Miscellaneous")

# ------------------------------------------------
# Main classify_or_create
# ------------------------------------------------
def classify_or_create(session: Session, text: str):
    """
    1. Tries to classify text using DB taxonomy.
    2. Falls back to taxonomy_master.json.
    3. Creates & syncs new taxonomy entries to both.
    """
    db_taxonomies = session.query(TaxonomyCategory).all()

    for t in db_taxonomies:
        if t.name.lower() in text.lower():
            return (t.group, t.name)

    json_data = load_master_taxonomy()
    for group, content in json_data.items():
        if group in ["version", "last_updated", "description"]:
            continue
        for section, details in content.items():
            values = details.get("values", [])
            for v in values:
                if v.lower() in text.lower():
                    return (group, v)

    inferred_group, inferred_name = infer_taxonomy(text)

    # Add to DB
    new_entry = TaxonomyCategory(
        name=inferred_name,
        group=inferred_group,
        description=f"Auto-generated category inferred from file text",
        created_at=datetime.utcnow(),
    )
    session.add(new_entry)
    session.commit()

    # Add to JSON
    json_data.setdefault(inferred_group, {}).setdefault("Inferred", {"values": [], "synonyms": {}})
    if inferred_name not in json_data[inferred_group]["Inferred"]["values"]:
        json_data[inferred_group]["Inferred"]["values"].append(inferred_name)
        save_master_taxonomy(json_data)

    logger.info(f"[Taxonomy] 🧩 Added new taxonomy: {inferred_group} → {inferred_name}")
    return (inferred_group, inferred_name)

# ------------------------------------------------
# Optional manual sync function
# ------------------------------------------------
def sync_to_db():
    """Trigger manual taxonomy sync (optional)."""
    from api.services.taxonomy_loader import run_sync
    run_sync()

# ==============================================================
# 🧠 get_best_taxonomy_match — Semantic Category Detection
# ==============================================================

import json, os, difflib, logging
from typing import Dict, Optional
from chromadb.utils import embedding_functions

# Path to your taxonomy master file
TAXONOMY_FILE = os.path.join(os.path.dirname(__file__), "taxonomy_master.json")

# Use Chroma’s default embedding model for lightweight similarity
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

def _load_taxonomy_data() -> list:
    """Load taxonomy categories from JSON file or return empty list."""
    if not os.path.exists(TAXONOMY_FILE):
        logging.warning("[TaxonomyHelper] taxonomy_master.json not found.")
        return []
    try:
        with open(TAXONOMY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("categories", [])
    except Exception as e:
        logging.error(f"[TaxonomyHelper] Failed to load taxonomy: {e}")
        return []

def _text_similarity(a: str, b: str) -> float:
    """Compute normalized string similarity (0–1)."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def get_best_taxonomy_match(text: str) -> Optional[Dict]:
    """
    Detect the best matching taxonomy category and subcategory for the given text.
    Uses string + embedding similarity.
    Returns { "category": ..., "subcategory": ..., "confidence": float }
    """
    taxonomy_data = _load_taxonomy_data()
    if not taxonomy_data:
        return {"category": "Uncategorized", "subcategory": "General Business", "confidence": 0.0}

    # Create embedding for the input text
    try:
        query_emb = embedding_fn([text])[0]
    except Exception as e:
        logging.warning(f"[TaxonomyHelper] Embedding fallback: {e}")
        query_emb = []

    best_match = None
    best_score = 0.0

    for entry in taxonomy_data:
        name = entry.get("name", "")
        group = entry.get("group", "")
        desc = entry.get("description", "")

        # Simple text similarity
        str_sim = _text_similarity(text, f"{name} {group} {desc}")

        # Embedding similarity (dot product)
        emb_sim = 0
        if query_emb:
            try:
                entry_emb = embedding_fn([name])[0]
                emb_sim = sum(a*b for a, b in zip(query_emb, entry_emb)) / (
                    (sum(a*a for a in query_emb)**0.5) * (sum(b*b for b in entry_emb)**0.5)
                )
            except Exception:
                emb_sim = 0

        score = (str_sim * 0.6) + (emb_sim * 0.4)

        if score > best_score:
            best_score = score
            best_match = entry

    if not best_match:
        return {"category": "Uncategorized", "subcategory": "General Business", "confidence": 0.0}

    # Confidence threshold (fallback if low)
    confidence = round(best_score, 3)
    if confidence < 0.99:
        return {
            "category": "Uncategorized",
            "subcategory": "General Business",
            "confidence": confidence,
        }

    return {
        "category": best_match.get("group", "General Business"),
        "subcategory": best_match.get("name", "Uncategorized"),
        "confidence": confidence,
    }
