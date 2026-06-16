# taxonomy_service.py

from .canonical_normalizer import canonical_normalize
from .similarity_engine import find_similar_taxonomy
from .phase1_taxonomy_models import PendingTaxonomy, Taxonomy


def create_pending_taxonomy(db, raw_name):
    canonical_name = canonical_normalize(raw_name)

    # Check exact canonical match
    existing = db.query(Taxonomy).filter(Taxonomy.canonical_name == canonical_name).first()
    if existing:
        raise ValueError("Duplicate taxonomy (canonical match).")

    # Find similar taxonomy entries
    similar = find_similar_taxonomy(db, canonical_name, top_n=5)

    # Determine suggested level and parent
    suggested_level, suggested_parent = auto_detect_level(db, similar)

    # Create pending entry
    pending = PendingTaxonomy(
        raw_name=raw_name,
        canonical_name=canonical_name,
        suggested_parent_id=suggested_parent,
        suggested_level=suggested_level,
        similar_existing_ids=[s["id"] for s in similar],
        confidence=max([s["score"] for s in similar]) if similar else 0.0,
    )

    db.add(pending)
    db.commit()
    return pending
