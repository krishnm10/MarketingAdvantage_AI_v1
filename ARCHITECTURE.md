# MarketingAdvantage AI — Ingestion Architecture
## Complete Data Flow: File Upload → Broker → Worker → Parse → Chunk → Embed → Vector DB

---

## The Big Picture

```
                 ┌─────────────────────────────────────────┐
                 │         CLIENT / USER / SYSTEM          │
                 │   (browser upload, API call, webhook)   │
                 └─────────────────┬───────────────────────┘
                                   │  HTTP POST /api/v2/ingest/upload
                                   ▼
                 ┌─────────────────────────────────────────┐
                 │          FASTAPI APPLICATION            │
                 │       ingestion_api_v2.py               │
                 │   • Authenticate user (JWT)             │
                 │   • Save file to disk                   │
                 │   • Check: CELERY_ENABLED = true/false  │
                 └──────────┬──────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              │                            │
     CELERY_ENABLED=true          CELERY_ENABLED=false
              │                            │
              ▼                            ▼
  ┌──────────────────────┐    ┌──────────────────────────┐
  │   MESSAGE BROKER     │    │   INLINE (Direct) MODE   │
  │  (pluggable broker)  │    │   Slower, no broker      │
  │  Redis/Kafka/Rabbit  │    │   needed. Runs inside    │
  │  NATS/SQS/Pulsar...  │    │   the API process.       │
  └──────────┬───────────┘    └────────────┬─────────────┘
             │                             │
             ▼                             │
  ┌──────────────────────┐                 │
  │   CELERY WORKER      │                 │
  │  tasks.py            │                 │
  │  run_ingestion_      │                 │
  │  pipeline(...)       │                 │
  └──────────┬───────────┘                 │
             │                             │
             └────────────┬────────────────┘
                          │
                          ▼
          ┌───────────────────────────────────────┐
          │         INGESTION PIPELINE            │
          │     ingestion_service_v2.py           │
          │                                       │
          │  1. PARSE    (pdf/csv/docx/json...)   │
          │  2. CHUNK    (semantic/overlap/rust)  │
          │  3. DEDUP    (L1 hash, L2 GCI, L3 AI) │
          │  4. STORE    (PostgreSQL)             │
          │  5. EMBED    (AI model → vectors)     │
          │  6. UPSERT   (Vector DB)              │
          └───────────────────────────────────────┘
```

**Answer to your question:**
> "After loading files/data through any broker, it processes and sends back to our ingestion system — to parse, chunk, embed, and upsert into vector DB"

YES — exactly right. The broker is just a **waiting room / delivery truck**. The actual work (parse → chunk → embed → upsert) is **always done by the same ingestion pipeline** (`IngestionServiceV2`), regardless of which broker you use. The broker only determines **how fast and how many** files can be processed in parallel.

---

## Mode 1: Celery + Broker (Fast, Scalable)

```
CELERY_ENABLED=true
CELERY_BROKER=redis          ← change this one line to swap broker
```

### What happens step by step:

```
Step 1  ─── User uploads a file via HTTP POST
             ↓
Step 2  ─── API saves file to disk: /uploaded_files/report_abc123.pdf
             ↓
Step 3  ─── API computes SHA256 hash of the file
             ↓
Step 4  ─── API checks PostgreSQL: "Has this exact file been ingested before?"
             → YES: Return {"status": "duplicate_skipped"}   — STOP HERE
             → NO:  Continue
             ↓
Step 5  ─── API creates a record in PostgreSQL (status = "uploaded")
             INSERT INTO ingested_files (id, filename, status="uploaded", ...)
             ↓
Step 6  ─── API sends a MESSAGE to the broker:
             run_ingestion_pipeline.delay(
                 file_id   = "abc-123-uuid",
                 saved_path = "/uploaded_files/report_abc123.pdf",
                 file_ext   = ".pdf"
             )
             ↓
             Return to user immediately: {"status": "queued", "task_id": "xyz-789"}
             ← USER GETS RESPONSE IN MILLISECONDS (no waiting for processing)
             ↓
Step 7  ─── BROKER stores the message (Redis list / Kafka topic / RabbitMQ queue)
             Queue name: "ingestion"
             ↓
Step 8  ─── CELERY WORKER picks up the message:
             Worker process running on same machine or separate server
             Calls: run_ingestion_pipeline("abc-123-uuid", "/uploaded_files/...", ".pdf")
             ↓
Step 9  ─── Worker runs the FULL INGESTION PIPELINE (see Pipeline section below)
             ↓
Step 10 ─── Worker updates PostgreSQL record: status = "processed"
```

---

## Mode 2: Inline / Direct (Slow, Simple)

```
CELERY_ENABLED=false
```

### What happens step by step:

```
Step 1  ─── User uploads file via HTTP POST
             ↓
Step 2-5 ─── Same as above (save, hash, dedup check, DB record)
             ↓
Step 6  ─── API calls the pipeline DIRECTLY (no broker, no worker):
             parsed = await parser_func(saved_path)
             await IngestionServiceV2.ingest_parsed_output(file_id, parsed)
             ↓
             User waits for the entire pipeline to finish
             ← Response returned AFTER processing completes (can take seconds/minutes)
             ↓
Step 7  ─── Same pipeline runs (parse → chunk → dedup → store → embed → upsert)
```

**Why use inline?** No broker to set up. Good for development, testing, or low-traffic setups.
**Why use Celery?** User gets response instantly. Multiple files process in parallel. Worker can retry on failure.

---

## The Broker — Just a Delivery Truck

```
YOUR .ENV FILE
══════════════

CELERY_ENABLED=true      ← master on/off switch
CELERY_BROKER=redis      ← which broker to use

Just change this ONE line to swap brokers:
  redis       → uses CELERY_REDIS_URL
  rabbitmq    → uses RABBITMQ_URL
  kafka       → uses KAFKA_BOOTSTRAP_SERVERS + KAFKA_SASL_*
  nats        → uses NATS_URL
  pulsar      → uses PULSAR_URL
  sqs         → uses SQS_REGION + AWS credentials
  pubsub      → uses GOOGLE_CLOUD_PROJECT
  eventhubs   → uses KAFKA_BOOTSTRAP_SERVERS (EventHubs Kafka endpoint)
  upstash     → uses UPSTASH_BROKER_TYPE (redis or kafka)
  redpanda    → uses KAFKA_BOOTSTRAP_SERVERS (Redpanda is Kafka-compatible)
  ...and 8 more
```

### How the broker selection works internally:

```
broker_config.py reads .env:
    CELERY_BROKER = "redis"
              ↓
    Calls _build_redis()
              ↓
    Returns:
      broker_url      = "redis://default:pass@35.154.x.x:6379/0"
      result_backend  = "redis://default:pass@35.154.x.x:6379/0"
      extra_conf      = { visibility_timeout: 3600 }
              ↓
celery_app.py uses those values:
    Celery("mai_worker", broker=broker_url, backend=result_backend)
```

**The broker does NOT process data.** It only holds the message:
```
Message in broker =  {
    "task": "tasks.run_ingestion_pipeline",
    "args": ["abc-123-uuid", "/uploaded_files/report.pdf", ".pdf"],
    "retries": 0
}
```
The worker reads this message, unpacks the arguments, and calls the ingestion pipeline.

---

## The Ingestion Pipeline (Same for Both Modes)

This is the core — runs whether Celery is on or off.

```
INPUT: file_id + saved_path + file_ext
         ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1: PARSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File extension → correct parser:

  .pdf   → parse_pdf()    → extracts pages, tables, headings
  .docx  → parse_docx()   → extracts paragraphs, styles
  .xlsx  → parse_excel()  → extracts rows/columns as structured data
  .csv   → parse_csv()    → rows as structured data
  .txt   → parse_text()   → raw text
  .json  → parse_json()   → nested key-value pairs
  .xml   → parse_xml()    → nodes and attributes

OUTPUT: parsed_output dict  e.g. {"text": "full content...", "metadata": {...}}
         ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 2: CHUNK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Break the text into small pieces (chunks) that fit into embedding models.
Strategy selected from CHUNKING_STRATEGY env var:

  semantic          → group sentences by meaning (uses NLP)
  overlap           → fixed-size windows with overlap (e.g., 512 tokens, 50 overlap)
  smart_check       → auto-selects best strategy per content type
  recursive_overlap → recursively split by paragraph → sentence → word
  rust              → ultra-fast Rust-based splitter (default, production)

Special cases:
  CSV/Excel    → each row becomes one chunk (no splitting needed)
  Visual docs  → LLM is called to EXPLAIN charts/tables first, then chunk the explanation

OUTPUT: List of chunk dicts:
  {
    "text":          "The Q3 revenue was $4.2M...",
    "cleaned_text":  "q3 revenue was 4 2 m ...",
    "tokens":        156,
    "semantic_hash": "a3f8c12d9e...",   ← SHA256 of cleaned text
    "source_type":   "pdf",
    "metadata":      {"page": 3, "heading": "Q3 Results"}
  }
         ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 3: DEDUPLICATE (3 Layers)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before storing anything — check if content already exists.

LAYER 1 — Hash Dedup (MAI_DEDUP_L1_ENABLED=true)
  "Is this exact text in the CURRENT batch?"
  → Compare semantic_hash within the batch of chunks
  → Exact match = mark as duplicate, skip
  → FAST: O(n) hash comparison, no DB query

LAYER 2 — GCI Dedup (MAI_DEDUP_L2_ENABLED=true)
  "Has this exact text been ingested EVER before across ALL files?"
  → Query GlobalContentIndex table in PostgreSQL
  → SELECT id WHERE semantic_hash IN (all_new_hashes)
  → Match = mark as duplicate, link to original
  → FAST: single SQL query for entire batch

LAYER 3 — Semantic AI Dedup (MAI_DEDUP_L3_ENABLED=true)
  "Is this text MEANINGFULLY SIMILAR to something already in the vector DB?"
  → Embed the chunk (create vector)
  → Search vector DB for top-1 nearest neighbor
  → If similarity > threshold (e.g., 0.95 = 95% similar) = mark as duplicate
  → SLOWER: requires embedding + vector search per chunk
  → Skipped for CSV/Excel rows (each row is independent data)

OUTPUT:
  unique_chunks   = chunks that passed all 3 layers
  dedup_stats     = { total: 100, unique: 87, duplicates: 13, dedup_ratio: 0.13 }
         ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 4: REGISTER IN GCI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Record new unique chunks in the GlobalContentIndex (for future L2 dedup checks).

  INSERT INTO global_content_index (semantic_hash, file_id, business_id, ...)
  → This is what Layer 2 queries next time
         ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 5: STORE IN POSTGRESQL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Save ALL chunks (unique + duplicate) into the ingested_content table.
Duplicates are stored but marked:
  is_duplicate = true
  duplicate_of = "original_chunk_uuid"
  similarity_score = 0.97   (for L3 duplicates)

Bulk insert — all chunks in ONE database round trip (fast).
         ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 6: EMBED + UPSERT TO VECTOR DB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Only UNIQUE chunks get embedded and stored in the vector DB.

STEP A: Determine chosen embedding model from MAI_EMBEDDER:
  huggingface → local HuggingFace sentence-transformers model
  ollama      → local Ollama embedding endpoint
  openai      → OpenAI text-embedding-3-small/large
  cohere      → Cohere embed-v3

STEP B: Ensure vector DB collection exists:
  vectordb.ensure_collection("ingested_content", embedding_dim=768)

STEP C: Batch embedding (parallel):
  Split chunks into batches (INGEST_BATCH_SIZE, e.g., 256)
  For each batch concurrently (INGEST_EMBED_PARALLELISM workers):
    texts      = ["chunk text 1", "chunk text 2", ...]
    vectors    = embedder.embed_documents(texts)   ← calls AI model
    → Returns: [[0.12, -0.87, 0.34, ...], ...]    ← float arrays (768 or 1536 dims)

STEP D: Upsert to chosen Vector DB (MAI_VECTORDB):
  chroma    → local ChromaDB (file-based)
  qdrant    → Qdrant (local or cloud)
  milvus    → Milvus (local or Zilliz cloud)
  pinecone  → Pinecone managed cloud
  weaviate  → Weaviate (local or cloud)
  redis     → Redis Stack with vector search

  vectordb.batch_upsert(
    collection = "ingested_content",
    ids        = ["hash1", "hash2", ...],     ← semantic_hash as ID
    vectors    = [[0.12, -0.87,...], ...],    ← embedding vectors
    payloads   = [
      {"file_id": "abc", "business_id": "xyz", "source_type": "pdf", ...},
      ...
    ]
  )

OUTPUT: All unique chunks now live in both:
  ✓ PostgreSQL (full text, metadata, dedup info)
  ✓ Vector DB  (embedding vectors, for similarity search)
         ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 7: UPDATE FILE STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  UPDATE ingested_files SET
    status           = "processed",
    total_chunks     = 100,
    unique_chunks    = 87,
    duplicate_chunks = 13,
    dedup_ratio      = 0.13
  WHERE id = "abc-123-uuid"
```

---

## End-to-End Timeline (Real Example)

**Scenario:** User uploads `marketing_report_Q3.pdf` (50 pages)

```
T+0ms    ──→  HTTP POST /api/v2/ingest/upload  (file arrives)
T+10ms   ──→  File saved to disk
T+15ms   ──→  SHA256 hash computed
T+20ms   ──→  PostgreSQL check: not a duplicate
T+25ms   ──→  PostgreSQL record created (status=uploaded)
T+30ms   ──→  Message sent to Redis broker
T+35ms   ──→  User gets response: {"status":"queued","task_id":"xyz-789"}
              ← API is DONE. User doesn't wait for the rest.

              ─── BROKER holds the message ───

T+100ms  ──→  Celery worker picks up the message
T+200ms  ──→  PARSE: pdf → extracted text from 50 pages
T+300ms  ──→  CHUNK: text split into ~200 chunks
T+400ms  ──→  DEDUP L1: 12 exact duplicates found in batch → 188 unique
T+600ms  ──→  DEDUP L2: 5 more found in GCI table → 183 unique
T+900ms  ──→  DEDUP L3: 2 more near-duplicates via vector search → 181 unique
T+950ms  ──→  GCI registration: 181 new entries
T+1000ms ──→  PostgreSQL: INSERT 200 chunks (all, with duplicate flags)
T+2500ms ──→  EMBED: 181 chunks embedded in parallel (6 batches of ~30)
T+3000ms ──→  VECTOR DB UPSERT: 181 vectors stored in Qdrant
T+3100ms ──→  PostgreSQL: status = "processed", 181 unique / 19 duplicates
T+3200ms ──→  DONE ✓
```
**The user experienced only a 35ms wait.** Processing happened in the background.

---

## How the Pluggable Connectors Fit Together

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PLUGGABLE CONNECTORS MAP                             │
│                                                                             │
│  MAI_VECTORDB=chroma        Which DB stores the embedding vectors          │
│  MAI_EMBEDDER=huggingface   Which AI model creates the vectors             │
│  MAI_LLM=ollama             Which LLM explains visual content              │
│  CHUNKING_STRATEGY=rust     How text is split into chunks                  │
│  CELERY_ENABLED=true        Whether a message queue is used                │
│  CELERY_BROKER=redis        Which message queue technology                 │
│                                                                             │
│  All of these can be swapped by changing ONE env var each.                 │
│  No code changes needed.                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Supported Broker Tiers

| Tier | Broker | When to Use |
|------|--------|-------------|
| **1 — Built-in** | Redis | Default. You already have Redis running. |
| **1 — Built-in** | RabbitMQ | Enterprise teams, complex routing rules. |
| **1 — Built-in** | Amazon SQS | AWS-native, serverless, no broker to manage. |
| **2 — Community** | Apache Kafka | Very high throughput (millions of msgs/day). |
| **2 — Community** | Redpanda | Kafka-compatible, faster startup, lower cost. |
| **2 — Community** | WarpStream | Kafka API, cloud-native, zero ops. |
| **2 — Community** | NATS JetStream | Lightweight, ultra-low latency. |
| **2 — Community** | Apache Pulsar | Multi-tenant, geo-replication. |
| **3 — Cloud** | Azure Event Hubs | Azure-native, Kafka API endpoint. |
| **3 — Cloud** | Google Pub/Sub | GCP-native, auto-scaled. |
| **3 — Cloud** | Amazon Kinesis | AWS streaming, time-ordered records. |
| **3 — Cloud** | Upstash | Serverless Redis or Kafka, pay-per-use. |
| **3 — Cloud** | Aiven for Kafka | Managed Kafka, any cloud. |
| **3 — Cloud** | StreamNative | Managed Pulsar. |
| **3 — Cloud** | Tinybird | Analytics-first Kafka endpoint. |
| **3 — Cloud** | GlassFlow | Stream processing pipeline. |
| **3 — Cloud** | Redis Streams | Redis 5.0+ consumer groups. |

**Regardless of which broker you pick — the pipeline is identical.**
The broker only changes HOW the task message is delivered to the worker.

---

## Configuration Quick Reference

```ini
# ── MASTER SWITCH ───────────────────────────────────────────
CELERY_ENABLED=false        # false = inline mode (no broker needed)
                             # true  = Celery + broker mode (fast)

# ── WHICH BROKER ────────────────────────────────────────────
CELERY_BROKER=redis          # change to: rabbitmq / kafka / nats / sqs ...

# ── BROKER CONNECTION (only the selected one is used) ───────
CELERY_REDIS_URL=redis://localhost:6379/0

# ── WHICH VECTOR DB RECEIVES THE EMBEDDINGS ─────────────────
MAI_VECTORDB=chroma          # chroma / qdrant / milvus / pinecone / weaviate / redis

# ── WHICH AI MODEL CREATES THE EMBEDDINGS ───────────────────
MAI_EMBEDDER=huggingface     # huggingface / ollama / openai / cohere

# ── HOW TEXT IS CHUNKED ─────────────────────────────────────
CHUNKING_STRATEGY=rust       # rust / semantic / overlap / smart_check / recursive_overlap

# ── DEDUPLICATION LAYERS ────────────────────────────────────
MAI_DEDUP_L1_ENABLED=true    # Hash exact-match within batch
MAI_DEDUP_L2_ENABLED=true    # GCI table lookup (all-time)
MAI_DEDUP_L3_ENABLED=true    # AI similarity search (most thorough)
```

---

## Retry & Fault Tolerance

When using Celery, failures are handled automatically:

```
Worker encounters error:
  → Celery retries up to 3 times (max_retries=3)
  → Waits 60 seconds between retries (default_retry_delay=60)
  → If all 3 retries fail → task moves to "failure" state
  → PostgreSQL file status stays "uploaded" (not "processed")
  → File can be re-submitted for manual reprocessing
```

With inline mode there is no retry — if the API call fails, the user gets an error response and must upload again.

---

*Generated: March 22, 2026 — MarketingAdvantage AI v1*
