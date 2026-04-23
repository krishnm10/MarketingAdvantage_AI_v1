---
name: vector-db-migration-planner
description: Plan and validate migrations between vector databases (e.g., Pinecone, Qdrant, Milvus, Weaviate, Chroma), with checks for embedding compatibility, schema and metadata mapping, indexing strategies, and cutover/downtime plans. Use when designing, reviewing, or executing plans to move embeddings and metadata from one vector store to another in a RAG system.
---

# Vector DB Migration Planner

## When to use this skill

Use this skill whenever the user asks about:

- Moving from one vector database to another
- Reindexing embeddings due to model or schema changes
- Designing a migration or cutover strategy for a RAG/vector search backend

Always think in terms of the full RAG pipeline:
ingestion → chunking → embedding → storage → retrieval (hybrid) → reranking → LLM reasoning.

## Quick planning checklist

When planning a migration, always produce an output with these sections:

1. **Context & assumptions**
2. **Embedding compatibility**
3. **Schema & metadata mapping**
4. **Indexing & performance plan**
5. **Migration steps**
6. **Downtime & cutover strategy**
7. **Re-indexing plan**
8. **Risks, validation, and rollback**

Keep the plan concise but specific to the user’s stack and constraints.

## Inputs to collect

Before proposing a plan, collect or infer:

- Source and target vector DB vendors/versions and hosting (managed/self-hosted)
- Current embedding model(s), dimension, and similarity metric
- Approximate scale: number of vectors, collections/indexes, and average metadata size
- RAG usage: query types, hybrid search, filters, rerankers, and evaluation setup (if any)
- Availability requirements: accepted downtime/read-only window, RPO/RTO
- Multi-tenant or single-tenant model; which metadata fields carry tenant or access control info
- Any planned embedding model change as part of the migration

If information is missing, state assumptions explicitly in the plan.

## 1. Embedding compatibility

Always assess embedding compatibility first. This is a common failure point.

- Check that:
  - Vector dimensions match between existing embeddings and the target index.
  - Similarity metric (cosine, dot, L2, inner product) is supported and configured identically.
  - All stored vectors in a collection were produced by the same embedding model/config.
- If there is any change to:
  - Embedding model
  - Tokenization or preprocessing
  - Vector dimension
  - Similarity metric
  then:
  - Treat this as a **full re-embedding and re-indexing** task, not a raw copy.
  - Explicitly flag **mixed embedding spaces** as a **critical error** and forbid partial mixing.
- Decide:
  - **Reuse existing embeddings** (binary copy) only if model, dimension, and metric are identical and target DB supports them safely.
  - Otherwise, plan to:
    - Re-embed from original documents or canonical text in an ingestion store.
    - Batch embeddings to control cost and latency.
    - Version embeddings (for example, an `embedding_version` metadata field).

In the final plan, include a short **Embedding compatibility** section that clearly states:

- Whether re-embedding is required
- Estimated embedding volume
- High-level cost and latency implications (batching, concurrency, caching)

## 2. Schema and metadata mapping

Compare and map schema and metadata carefully; retrieval and filters depend on this.

Focus on:

- **Identifiers**
  - How primary keys/vector IDs are represented in source vs target.
  - Any namespace, collection, partition, or tenant structures.
- **Vector fields**
  - Field/column names storing vectors.
  - Whether multiple vectors per record are used and how that maps in the target.
- **Metadata**
  - How metadata is stored (JSON blob vs typed columns).
  - Size limits and indexing rules for metadata fields.
  - Fields used for filtering (for example, tenant, document_id, section_id, tags, language).
- **Hybrid search**
  - How sparse signals (BM25/inverted index) are stored or configured.
  - Whether hybrid search will live in the same product or an adjacent search engine.

Create a concise mapping table in the plan, for example:

- `id`: source → target
- `vector`: source field → target field and index
- Metadata: for each important field, specify name, type, purpose, and how filters will be implemented
- Namespaces/collections: source logical groups → target equivalents

Highlight any schema gaps and propose workarounds (for example, flatten nested metadata, encode enums).

## 3. Indexing and configuration

Design target indexing with performance and retrieval quality in mind.

For each collection/index:

- Choose index type and parameters appropriate for the target DB (for example, HNSW/IVF/PQ, `ef_search`, `M`, `nprobe`).
- Align similarity metric with the embedding model’s training objective.
- Configure replicas/shards for expected QPS, latency SLOs, and growth.

Plan index creation and ingestion:

- Pre-create collections/indexes before data backfill.
- Tune batch sizes and concurrency for ingestion to avoid overload.
- If hybrid search is used:
  - Ensure equivalent or better hybrid capabilities are available and configured.
  - Decide whether sparse indexes live inside the same product or a separate search engine.
- If reranking is used:
  - Verify that the migration preserves the fields the reranker expects (raw text, titles, metadata).

Include an **Indexing & performance plan** section that:

- Lists indexes/collections and key parameters
- Notes any differences vs the source system
- Flags components likely to affect latency or recall

## 4. Migration strategies and downtime

Choose and justify a migration strategy. Default to **low-risk, observable cutovers**.

Common strategies:

- **Cold cutover (offline reindex)**
  - Stop writes (and optionally reads) to the old DB.
  - Re-embed and/or reindex into the new DB.
  - Switch traffic when verification passes.
  - Use when downtime is acceptable and data volume is moderate.
- **Dual-write + shadow-read**
  - Write to both source and target during a backfill period.
  - Replay historical data into the target if needed.
  - Run shadow queries against the target and compare quality/latency vs source.
  - Promote target when confidence is high.
  - Use for high-availability systems.
- **Blue/green RAG backend**
  - Keep both vector DBs behind a routing layer.
  - Gradually shift a percentage of traffic to the new backend.
  - Roll back quickly by toggling the router if issues appear.

For each strategy, explicitly document:

- Expected downtime or read-only windows
- How writes are handled during migration
- Rollback plan and data consistency considerations

Match the strategy to the user’s RPO/RTO, QPS, and operational maturity.

## 5. Detailed migration steps

Always output a numbered, high-level step list tailored to the user’s stack.

Structure it roughly as:

1. **Discovery and design**
   - Inventory collections, indexes, and embedding configs.
   - Decide on embedding reuse vs re-embedding.
   - Design schema and metadata mapping.
2. **Target environment preparation**
   - Provision target DB infrastructure and networking.
   - Create collections and indexes with desired configuration.
   - Set up ACLs, auth, and tenant isolation.
3. **Backfill and reindex**
   - Extract source data or canonical documents.
   - Embed (if re-embedding) and write to target in batches.
   - Monitor ingestion errors, throughput, and index build time.
4. **Validation and comparison**
   - Run a golden set of RAG queries against both source and target.
   - Compare relevance, recall@k, and latency (P50/P95/P99).
   - Fix retrieval or ranking regressions before cutover.
5. **Cutover**
   - Enable dual-read or switch read traffic to target.
   - Monitor error rates, timeouts, and quality signals.
6. **Decommissioning**
   - After a safe period, decommission or archive the old vector DB.
   - Update documentation and runbooks.

Tailor each step to the named technologies (for example, Pinecone → Qdrant, Milvus → Weaviate) when the user provides them.

## 6. Re-indexing plan

If re-embedding is required, include a concrete re-indexing plan.

Specify:

- **Data source for re-embedding**
  - Original documents
  - Canonical chunk store
  - Changefeed or event log (for incremental backfill)
- **Embedding job strategy**
  - Parallelism and batching
  - Rate limits and backoff for embedding API calls
  - Caching and deduplication to avoid redundant embeddings
- **Validation**
  - Cardinality checks (record counts per collection/tenant before vs after).
  - Spot-check similarity of representative queries vs ground truth.
  - RAG evaluation on a golden dataset using relevance, recall, and faithfulness.
- **Versioning**
  - Track embedding model/version and chunking scheme.
  - Keep old embeddings available until cutover is complete and validated.

Call out token and cost implications, especially for large-scale re-embedding, and recommend batching and caching.

## 7. Risks, validation, and rollback

Include a **Risks & validation** section that covers likely failure modes:

- **Retrieval failures**
  - Missing relevant documents due to schema, filter, or ingestion issues.
- **Ranking failures**
  - Different ANN/index behavior impacting result ordering or recall@k.
- **Context failures**
  - Chunking or schema changes that break RAG context assembly (for example, wrong document boundaries or missing metadata).
- **Generation failures**
  - Hallucinations due to degraded retrieval quality or missing grounding.

For each, propose:

- **Detection**
  - Targeted evaluation queries pre- and post-migration.
  - Metrics and tracing (latency, error rate, recall proxies).
  - Sampled manual review if possible.
- **Mitigation**
  - Index parameter tuning (for example, HNSW parameters, nprobe).
  - Schema or metadata mapping fixes.
  - Temporary fallbacks to the old vector DB via a router.

Define a **rollback plan**:

- Conditions that trigger rollback (for example, error rate or quality threshold breaches).
- How to quickly route traffic back to the old DB.
- Data reconciliation steps if writes occurred during partial cutover.

## 8. Final output format

When responding to the user, organize the migration plan using headings like:

- **Context & assumptions**
- **Embedding compatibility**
- **Schema & metadata mapping**
- **Indexing & performance plan**
- **Migration steps**
- **Downtime & cutover strategy**
- **Re-indexing plan**
- **Risks, validation, and rollback**

Where useful, include short tables or bullet lists instead of prose walls. Keep the plan deterministic, pipeline-aware, and explicit about risks, downtime, and re-indexing requirements.

