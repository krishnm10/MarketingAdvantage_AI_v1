# Golden sets — retrieval & faithfulness evaluation

Labeled query packs for offline metrics (P@K, R@K, MRR, NDCG) and live E2E checks (route, substrings, LLM faithfulness).

## Layout

```
tests/golden_sets/
  schema/golden_set.schema.json
  invoice/vaidyanad_inv_1101.json
  fixtures/mini_smoke.json          # CI without live server (synthetic IDs)
```

## Labeling `relevant_chunk_ids`

Chat path returns PostgreSQL chunk UUIDs in `results[].chunk_id` (from `IngestedContentV2.id`).

After ingesting seed documents for a tenant:

```bash
python scripts/bootstrap_golden_chunk_ids.py --set invoice/vaidyanad_inv_1101.json --case inv_1101_total
```

This calls `POST /api/v2/retrieve/chat`, prints ranked `chunk_id` values, and can patch the JSON file.

RAG pipeline path returns vector point `id` in `retrieved_chunks[]` (may differ from DB UUID). Prefer labeling from **chat** for cross-path consistency, or maintain separate sets per path.

## Running evaluation

```bash
# Live (requires uvicorn + tenant corpus)
python scripts/run_golden_set_eval.py --set invoice/vaidyanad_inv_1101.json --paths chat,rag

# Pytest (loader + metrics unit tests; integration skipped unless GOLDEN_SET_LIVE=1)
pytest tests/test_retrieval_quality_golden_set.py -q
pytest tests/test_retrieval_quality_golden_set.py -m integration  # needs server
```

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `GOLDEN_SET_BASE_URL` | `http://127.0.0.1:8000` | API base |
| `GOLDEN_SET_ADMIN_USER` | `admin` | Auth |
| `GOLDEN_SET_ADMIN_PASS` | `admin` | Auth |
| `GOLDEN_SET_LIVE` | unset | Set `1` to enable integration tests |
| `GOLDEN_MIN_RECALL_AT_5` | optional | Fail runner if below threshold |

## `corpus_fingerprint`

Set in the golden file after ingest (hash of sorted `file_id` list). Runner warns when fingerprint drifts.
