# Request-flow diagrams (numbered steps + filenames)

## Files

| File | Use |
|------|-----|
| **`three_rag_retrieve_apis_comparison.html`** | **Side-by-side:** `POST /api/v2/rag/query` (pluggable `RAGPipeline`), `POST /api/v2/retrieve/query`, `POST /api/v2/retrieve/chat` — auth, tenant strictness, file lists, hybrid/rerank **order** (chat vs retrieve), and Mermaid comparison diagrams. |
| **`retrieve_from_ui_detailed_flow.html`** | **Dashboard retrieve / RAG-lite:** `POST /api/v2/retrieve/query` from `retrieve/page.tsx` — tenant validation, config vs legacy fallback, query-time `scan_text`, HyDE, embed via ingest pipeline, `RetrievalRuntime`, optional BM25/RRF + reranker, optional LLM answer + context scan. **No Celery.** |
| **`upload_from_ui_detailed_flow.html`** | **UI file upload only:** numbered sequence (every step), Celery continuation, layer diagram, and a **25-row table** with **file · method · config**. |
| **`data_flow_with_files.html`** | Broader request flows (login, retrieve, RAG, …). |
| **`request_flow_steps.csv`** | Same flows as **rows for Excel**: `FlowID`, `Step`, `From`, `To`, `Action`, `PrimaryFilesOrModules`. Filter by `FlowID` (`A_API_SHELL`, `C_INGEST_UPLOAD`, …). |
| **`data_flow_nodes.csv`** / **`data_flow_edges.csv`** | Layer map and connections (no per-request step numbers). |

## PDF

1. Open `data_flow_with_files.html` in Chrome or Edge.  
2. **Ctrl+P** → **Save as PDF**.

If Mermaid does not load from `file://`:

```powershell
cd c:\ProjectK\MarketingAdvantage_AI_v1\docs\diagrams
python -m http.server 8765
```

Then visit `http://localhost:8765/data_flow_with_files.html`.

## Scope

These flows cover the **main** admin/API paths. Other routes (config, RAG config, Kafka, WebSocket ingestion progress, etc.) follow the same **shell** (flow **A**) then their own handler file under `app/api/v2/`.
