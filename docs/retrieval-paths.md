# Authoritative retrieval paths

## Production path

`POST /api/v2/retrieve/chat` ([retrieve_chat_api.py](../../app/api/v2/retrieve_chat_api.py)) is the **production** retrieval + generation path for the admin UI and tenant chat. Chunk IDs are PostgreSQL `IngestedContentV2.chunk_id` values (`chunk_source: postgres`).

## Programmatic / eval path

`POST /api/v2/rag/query` ([rag_api.py](../../app/api/v2/rag_api.py)) is **auth-gated admin/eval only** (golden-set runner, pipeline debugging). Chunk IDs may originate from vector payload metadata (`chunk_source: vector_payload`) and are normalized via `chunk_id_from_payload`.

## Shared contract

Both paths expose `chunk_id` + `chunk_source` on result rows. The frozen `ChunkReference` dataclass in [types_retrieve.py](../../app/retrieval/types_retrieve.py) is the canonical identity contract; parity tests live in `tests/test_chunk_reference_parity.py`.
