# Scalability Improvements — Architectural Plan

These items require infrastructure changes and should be planned as separate sprints.
They are documented here for reference and future implementation.

---

## 5.1 — Celery Task Chain for Ingestion

**Current state:** `ingest_document()` runs all stages sequentially (parse → chunk → embed → upsert → dedup) in a single Celery task.

**Recommended architecture:**
```
ingest_document.s(file_path, config)
  | parse_document.s()
  | chunk_document.s()
  | embed_chunks.s()
  | upsert_to_vectordb.s()
  | run_deduplication.s()
```

**Benefits:**
- Each stage can be retried independently (no re-parsing on embed failure)
- Better visibility in Celery Flower (see which stage failed)
- Can scale embed workers independently (GPU nodes)
- Enables dead-letter queue per stage

**Implementation notes:**
- Use `celery.chain()` or `celery.chord()` for fan-out embedding
- Each stage receives and returns a dict with `task_context`
- Add `on_failure` callbacks per stage for granular error handling
- Preserve current single-task path as fallback via env var `CELERY_USE_CHAIN=true`

---

## 5.2 — Separate Validation Worker Pool

**Current state:** Content validation (schema checks, trust scoring, quality gates) runs in the same worker pool as ingestion.

**Recommended architecture:**
- Dedicated Celery queue: `validation_queue`
- Route validation tasks: `app.conf.task_routes = {"app.worker.tasks.validate_*": {"queue": "validation_queue"}}`
- Separate worker process: `celery -A app.worker worker -Q validation_queue -c 4`

**Benefits:**
- Validation failures don't block ingestion pipeline
- Can scale validation workers independently
- Enables async post-ingestion quality checks

---

## 5.3 — Qdrant ID Format Migration Utility

**Current state:** Some Qdrant collections may have mixed UUID vs string IDs from early ingestion runs.

**Recommended utility:**
```python
# scripts/migrate_qdrant_ids.py
# 1. Scroll all points in collection
# 2. For each point with non-UUID string ID, generate deterministic UUID5
# 3. Upsert with new UUID, delete old point
# 4. Log mapping for rollback
```

**Implementation notes:**
- Use `uuid.uuid5(uuid.NAMESPACE_URL, old_string_id)` for deterministic mapping
- Process in batches of 100 with progress bar
- Create backup scroll file before migration
- Add `--dry-run` flag to preview changes
- Gate behind confirmation prompt: "This will modify X points. Continue? [y/N]"

---

## Priority

| Item | Effort | Impact | Priority |
|------|--------|--------|----------|
| 5.1 Task Chain | High | High (resilience) | P1 — next sprint |
| 5.2 Validation Workers | Medium | Medium (isolation) | P2 |
| 5.3 Qdrant ID Migration | Low | Low (edge case) | P3 |
