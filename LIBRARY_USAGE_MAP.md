# Library Usage Map — MarketingAdvantage AI v1

> Auto-generated from AST import scan of all Python files in `app/`.
> Last updated: March 19, 2026.
> Each section shows: **Library** → **Purpose** → **Files that use it**

---

## Table of Contents

1. [Web Framework & Server](#1-web-framework--server)
2. [Database & ORM](#2-database--orm)
3. [Authentication & Security](#3-authentication--security)
4. [AI / LLM Providers](#4-ai--llm-providers)
5. [Vector Databases](#5-vector-databases)
6. [Embeddings & Rerankers](#6-embeddings--rerankers)
7. [Document & File Parsers](#7-document--file-parsers)
8. [Media Processing (Audio / Video / Image)](#8-media-processing-audio--video--image)
9. [Web Scraping & HTTP](#9-web-scraping--http)
10. [NLP & Text Processing](#10-nlp--text-processing)
11. [Data & Science Libraries](#11-data--science-libraries)
12. [Configuration & Settings](#12-configuration--settings)
13. [Observability & Monitoring](#13-observability--monitoring)
14. [File & System Utilities](#14-file--system-utilities)
15. [Testing](#15-testing)
16. [Requirements Summary Table](#16-requirements-summary-table)

---

## 1. Web Framework & Server

### `fastapi` (`fastapi==0.120.0`)
Core web framework. All HTTP routes, request/response models, dependency injection, WebSockets.

| File | Usage |
|------|-------|
| `app/main.py` | App entry point, lifespan startup/shutdown |
| `app/api/v2/admin_audit_api.py` | Admin audit log endpoints |
| `app/api/v2/auth_api.py` | Login / token endpoints |
| `app/api/v2/config_api.py` | Client config endpoints |
| `app/api/v2/ingestion_admin_api.py` | Ingestion management endpoints |
| `app/api/v2/ingestion_api_v2.py` | File upload / ingestion trigger endpoint |
| `app/api/v2/ingestion_health.py` | Health check endpoint |
| `app/api/v2/ingestion_integrity_api.py` | Data integrity endpoints |
| `app/api/v2/ingestion_sync_api.py` | Sync trigger endpoints |
| `app/api/v2/ingestion_ws_api.py` | WebSocket ingestion progress |
| `app/api/v2/rag_api.py` | RAG query endpoints |
| `app/api/v2/retrieve_api.py` | Retrieval endpoints |
| `app/auth/deps.py` | JWT auth dependencies (`Depends(...)`) |
| `app/auth/guards.py` | Role-based access guard |
| `app/services/ingestion/csv_parser_v2.py` | FastAPI `UploadFile` type |
| `app/services/ingestion/file_router_v2.py` | File routing via FastAPI background tasks |
| `app/services/ingestion/phantom_config_bridge.py` | Phantom config API bridge |
| `app/services/ingestion/watcher_ingestor_v2.py` | File-watcher ingestor endpoint |
| `app/services/ingestion/xml_parser_v2.py` | XML upload handler |
| `app/utils/file_utils.py` | `UploadFile` helper utilities |

---

### `uvicorn` (`uvicorn==0.38.0`)
ASGI server that runs the FastAPI app.
- **Used by**: Launch command only (`uvicorn app.main:app`). Not imported in code.
- **Pin reason**: Must stay at 0.38.x — newer versions may break Starlette 0.48.

---

### `uvloop` (`uvloop>=0.21.0 ; sys_platform != "win32"`)
Ultra-fast asyncio event loop replacement (Linux/macOS only).

| File | Usage |
|------|-------|
| `app/main.py` | `uvloop.install()` called before asyncio starts (wrapped in `try/except` — skipped on Windows) |

---

### `starlette` (`starlette==0.48.0`)
FastAPI's underlying ASGI toolkit. Middleware, request/response primitives.
- **Used by**: FastAPI internally. Not imported directly in app code.
- **Pin reason**: Must stay at 0.48.0 — fastapi 0.120.0 requires this exact version.

---

### `pydantic` (`pydantic==2.12.3`) & `pydantic-settings` (`pydantic-settings==2.13.1`)
Data validation and settings management.

| File | Usage |
|------|-------|
| `app/api/v2/auth_api.py` | Request/response schemas |
| `app/api/v2/config_api.py` | Config request schemas |
| `app/api/v2/ingestion_admin_api.py` | Admin request schemas |
| `app/api/v2/rag_api.py` | RAG query/response schemas |
| `app/api/v2/retrieve_api.py` | Retrieval schemas |
| `app/core/config/client_config_schema.py` | Client config Pydantic models |
| `app/core/settings.py` | **`pydantic-settings`** — `AppSettings` BaseSettings singleton (all env vars) |

---

## 2. Database & ORM

### `sqlalchemy` (`SQLAlchemy==2.0.44`)
Async ORM and query builder for PostgreSQL.

| File | Usage |
|------|-------|
| `app/db/base.py` | `DeclarativeBase` — ORM base class |
| `app/db/session_v2.py` | Async session factory |
| `app/db/models/admin_audit_log.py` | ORM model |
| `app/db/models/business_classification_v2.py` | ORM model |
| `app/db/models/classification_logs_v2.py` | ORM model |
| `app/db/models/global_content_index_v2.py` | ORM model |
| `app/db/models/ingested_content_v2.py` | ORM model |
| `app/db/models/ingested_file_v2.py` | ORM model |
| `app/db/models/pending_taxonomy_v2.py` | ORM model |
| `app/db/models/taxonomy_alias_v2.py` | ORM model |
| `app/db/models/taxonomy_v2.py` | ORM model |
| `app/api/v2/admin_audit_api.py` | DB queries |
| `app/api/v2/auth_api.py` | User lookup |
| `app/api/v2/config_api.py` | Config CRUD |
| `app/api/v2/ingestion_admin_api.py` | Ingestion record queries |
| `app/api/v2/ingestion_api_v2.py` | Ingestion state writes |
| `app/api/v2/ingestion_integrity_api.py` | Integrity validation queries |
| `app/api/v2/ingestion_sync_api.py` | Sync status queries |
| `app/api/v2/retrieve_api.py` | Retrieval queries |
| `app/core/connectors/base.py` | Base connector with DB support |
| `app/core/connectors/api_connector.py` | API source connector |
| `app/core/connectors/rss_connector.py` | RSS connector |
| `app/core/connectors/web_connector.py` | Web connector |
| `app/retrieval/repository.py` | Retrieval data fetching |
| `app/services/classification/*.py` | Classification result writes |
| `app/services/ingestion/deduplication_engine_v2.py` | Dedup lookups |
| `app/services/ingestion/file_router_v2.py` | File routing state |
| `app/services/ingestion/ingestion_service_v2.py` | ingestion orchestration |
| `app/services/ingestion/media/audio_ingestor_v1.py` | Audio record writes |
| `app/services/ingestion/media/image_ingestor_v1.py` | Image record writes |
| `app/services/ingestion/media/video_ingestor_v1.py` | Video record writes |
| `app/services/validation/*.py` | Validation record writes |
| `app/versions/fix_ingested_file_status.py` | Alembic migration helper |

---

### `asyncpg` (`asyncpg==0.30.0`)
Async PostgreSQL driver used by SQLAlchemy's async engine.
- **Used by**: SQLAlchemy internally via `create_async_engine("postgresql+asyncpg://...")`. Not imported directly.

---

### `psycopg2-binary` (`psycopg2-binary==2.9.11`)
Sync PostgreSQL driver required by Alembic migrations.
- **Used by**: Alembic (`alembic/env.py`) for synchronous migration runs. Not imported in app code directly.

---

### `alembic` (`alembic==1.17.2`)
Database schema migration tool.

| File | Usage |
|------|-------|
| `alembic/env.py` | Migration environment setup |
| `alembic/versions/*.py` | Migration scripts |
| `app/versions/fix_ingested_file_status.py` | Custom migration helper |

---

## 3. Authentication & Security

### `authlib` (`authlib>=1.3.0`, installed `1.6.9`)
JWT creation and verification. Replaced `python-jose` in the last migration.

| File | Usage |
|------|-------|
| `app/auth/generate_token.py` | `jwt.encode()` — creates access tokens with HS256 |
| `app/auth/deps.py` | `jwt.decode()` — validates bearer tokens in every protected route |

---

### `bcrypt` (`bcrypt==5.0.0`)
Password hashing.
- **Used by**: Auth service for hashing and verifying user passwords. Imported via `passlib` or directly in the auth service layer.

---

### `python-dotenv` (`python-dotenv==1.2.1`)
Loads `.env` file into environment variables at startup.

| File | Usage |
|------|-------|
| `app/auth/generate_token.py` | `load_dotenv()` — loads `JWT_SECRET_KEY` |
| `app/db/session_v2.py` | `load_dotenv()` — loads `DATABASE_URL` |

---

### `cryptography` (`cryptography==46.0.3`)
Low-level cryptographic primitives.
- **Used by**: `authlib` internally for RSA/ECDSA operations and token signing.

---

### `PyJWT` (`PyJWT==2.10.1`)
JWT utility library.
- **Used by**: Certain token utilities. `authlib` is the primary JWT library; `PyJWT` remains as a compatibility dependency.

---

## 4. AI / LLM Providers

### `anthropic` (`anthropic==0.85.0`)
Anthropic Claude API client.

| File | Usage |
|------|-------|
| `app/core/llms/anthropic_v1.py` | `AsyncAnthropic` client — Claude chat completions |

---

### `openai` (`openai==2.6.1`)
OpenAI GPT and Whisper API client.

| File | Usage |
|------|-------|
| `app/core/llms/openai_v1.py` | GPT-4/GPT-3.5 chat completions |
| `app/core/embedders/openai_v1.py` | `text-embedding-3-*` embeddings |

---

### `google-generativeai` (`google-generativeai>=0.8.0`)
Google Gemini API client.

| File | Usage |
|------|-------|
| `app/core/llms/gemini_v1.py` | `GenerativeModel` — Gemini Pro chat completions |

---

### `groq` (`groq>=0.9.0`)
Groq LPU inference API client (fast Llama/Mixtral inference).

| File | Usage |
|------|-------|
| `app/core/llms/groq_v1.py` | `AsyncGroq` client — ultra-fast LLM inference |

---

### `ollama` (`ollama>=0.4.0`)
Local LLM inference via Ollama server.

| File | Usage |
|------|-------|
| `app/core/llms/ollama_v1.py` | Local LLM completions (Llama, Mistral, etc.) |
| `app/core/embedders/ollama_v1.py` | Local embeddings via Ollama |

---

### `whisper` / `openai-whisper` (`openai-whisper==20250625`)
Audio transcription model (runs locally).

| File | Usage |
|------|-------|
| `app/ai/providers/gpu/whisper_gpu_v1.py` | GPU-accelerated transcription |
| `app/ai/providers/local/whisper_cpu_v1.py` | CPU transcription fallback |
| `app/services/ingestion/media/audio_ingestor_v1.py` | Transcribes audio files during ingestion |
| `app/services/ingestion/media/video_ingestor_v1.py` | Transcribes video audio track |

---

## 5. Vector Databases

All vector DB adapters live under `app/core/vectordb/`. The active backend is selected via `MAI_VECTORDB` env var.

### `chromadb` (`chromadb==1.3.0`)
Local embedded vector store (default dev backend).

| File | Usage |
|------|-------|
| `app/core/vectordb/chroma_v1.py` | ChromaDB adapter — store/query embeddings |
| `app/api/v2/ingestion_health.py` | Health check: ping ChromaDB |
| `app/services/classification/classification_service.py` | Semantic classification via chroma |
| `app/services/classification/embedding_ranker.py` | Embedding ranking against chroma index |
| `app/services/classification/taxonomy_loader.py` | Taxonomy embeddings stored in chroma |
| `app/services/retrieval/chroma_search.py` | Direct chroma query |
| `app/services/retrieval/chroma_search_service.py` | Chroma search service wrapper |

---

### `pymilvus` (`pymilvus==2.6.10`)
Milvus distributed vector database client.

| File | Usage |
|------|-------|
| `app/core/vectordb/milvus_v1.py` | Milvus adapter — insert/search vectors |

---

### `pinecone-client` (`pinecone-client==6.0.0`)
Pinecone managed vector database client.

| File | Usage |
|------|-------|
| `app/core/vectordb/pinecone_v1.py` | Pinecone adapter — upsert/query vectors |

---

### `qdrant-client` (`qdrant-client>=1.9.0`)
Qdrant vector database client.

| File | Usage |
|------|-------|
| `app/core/vectordb/qdrant_v1.py` | Qdrant adapter — insert/search vectors |

---

### `redis` / `redis[hiredis]` (`redis[hiredis]>=5.0`)
Redis vector search (via RediSearch module) + general caching.

| File | Usage |
|------|-------|
| `app/core/vectordb/redis_v1.py` | Redis vector adapter |
| `app/api/v2/ingestion_health.py` | Health check: ping Redis |

---

### `weaviate-client` (`weaviate-client==4.20.4`)
Weaviate vector database client.

| File | Usage |
|------|-------|
| `app/core/vectordb/weaviate_v1.py` | Weaviate adapter — insert/query vectors |

---

## 6. Embeddings & Rerankers

### `sentence-transformers` (`sentence-transformers==5.1.2`)
Sentence-level embedding models (HuggingFace ecosystem).

| File | Usage |
|------|-------|
| `app/core/embedders/huggingface_st_v1.py` | Primary local embedder (e.g., `all-MiniLM-L6-v2`) |
| `app/core/rerankers/bge_reranker_v1.py` | BGE reranker via SentenceTransformer |
| `app/core/rerankers/crossencoder_v1.py` | Cross-encoder reranking |
| `app/services/classification/embedding_ranker.py` | Embedding-based taxonomy ranking |
| `app/services/classification/taxonomy_loader.py` | Taxonomy embedding generation |
| `app/services/ingestion/*.py` (multiple copy files) | Segment embedding during ingestion |

---

### `cohere` (`cohere==5.20.7`)
Cohere API — embeddings and reranking.

| File | Usage |
|------|-------|
| `app/core/embedders/cohere_v1.py` | `embed-english-v3.0` embeddings via Cohere API |
| `app/core/rerankers/cohere_v1.py` | `rerank-english-v3.0` reranking |

---

### `FlagEmbedding` (`FlagEmbedding==1.3.5`)
BAAI BGE embedding and reranking models.

| File | Usage |
|------|-------|
| `app/core/rerankers/bge_reranker_v1.py` | BGE-M3 / BGE-reranker models |

---

### `flashrank` (`flashrank==0.2.10`)
Ultra-fast cross-encoder reranking (no GPU required).

| File | Usage |
|------|-------|
| `app/core/rerankers/flashrank_v1.py` | `Ranker` — fast lightweight reranking |

---

### `ragatouille` (`ragatouille>=0.0.8`)
ColBERT-based late-interaction reranking.

| File | Usage |
|------|-------|
| `app/core/rerankers/colbert_v1.py` | ColBERT reranker (optional — graceful fallback if not installed) |

---

### `torch` (`torch==2.9.0`)
PyTorch deep learning framework.

| File | Usage |
|------|-------|
| `app/core/embedders/huggingface_st_v1.py` | GPU/CPU tensor ops for embeddings |
| `app/core/rerankers/colbert_v1.py` | ColBERT tensor operations |
| `app/services/ingestion/phantom_hardware_profiler.py` | GPU detection (`torch.cuda.is_available()`) |

---

### `transformers` (`transformers==4.57.1`)
HuggingFace Transformers — model loading, tokenization.
- **Used by**: `sentence-transformers`, `FlagEmbedding`, `accelerate` internally. Not imported directly in app code but required by the embedding stack.

---

### `accelerate` (`accelerate==1.11.0`)
HuggingFace model acceleration (multi-GPU, mixed precision).
- **Used by**: `transformers` / `sentence-transformers` internally for device management.

---

## 7. Document & File Parsers

### `PyMuPDF` / `fitz` (`PyMuPDF==1.26.5`)
High-performance PDF parsing with text extraction and coordinate data.

| File | Usage |
|------|-------|
| `app/services/ingestion/pdf_parser_v2.py` | Primary PDF text extractor |
| `app/services/ingestion/media/document_visual_interceptor_v1.py` | Visual page layout analysis |

---

### `pdfplumber` (`pdfplumber==0.11.8`)
Table-aware PDF parser (extracts tables from PDFs with high accuracy).

| File | Usage |
|------|-------|
| `app/services/ingestion/pdf_parser_v2.py` | Table extraction from PDFs (complementary to fitz) |

---

### `python-docx` / `docx` (`python-docx==1.2.0`)
Microsoft Word `.docx` file parser.

| File | Usage |
|------|-------|
| `app/services/ingestion/docx_parser_v2.py` | Extract text and structure from Word documents |
| `app/services/ingestion/media/document_visual_interceptor_v1.py` | Visual document layout |

---

### `openpyxl` (`openpyxl==3.1.5`)
Excel `.xlsx` read/write.
- **Used by**: `pandas` internally for Excel file reading. Not imported directly.

---

### `pandas` (`pandas==2.3.3`)
DataFrame library for tabular data (CSV, Excel, RSS).

| File | Usage |
|------|-------|
| `app/services/ingestion/csv_parser_v2.py` | CSV file parsing |
| `app/services/ingestion/excel_parser_v2.py` | Excel file parsing |
| `app/services/ingestion/ingestion_service_v2.py` | Data frame operations during ingestion |
| `app/services/ingestion/rss_ingestor_v2.py` | RSS feed data structuring |
| `app/services/ingestion/row_segmenter_v2.py` | Row-level segmentation |
| `app/services/ingestion/xml_parser_v2.py` | XML-to-DataFrame parsing |

---

### `feedparser` (`feedparser==6.0.12`)
RSS/Atom feed parser.

| File | Usage |
|------|-------|
| `app/services/ingestion/rss_ingestor_v2.py` | Parse RSS/Atom feeds into structured data |

---

### `chardet` (`chardet==5.2.0`)
Character encoding detection.

| File | Usage |
|------|-------|
| `app/services/ingestion/text_parser_v2.py` | Detect encoding of unknown text files before parsing |

---

### `PyYAML` / `yaml` (`PyYAML==6.0.3`)
YAML file parser.

| File | Usage |
|------|-------|
| `app/core/config/client_config_schema.py` | Load client config YAML files |

---

## 8. Media Processing (Audio / Video / Image)

### `Pillow` / `PIL` (`pillow==11.3.0`)
Image loading, transformation, and format conversion.

| File | Usage |
|------|-------|
| `app/ai/providers/local/image_caption_cpu_v1.py` | Load image for captioning |
| `app/services/ingestion/media/image_ingestor_v1.py` | Image ingestion and preprocessing |
| `app/services/ingestion/media/media_hash_utils.py` | Perceptual hash computation |

---

### `pytesseract` (`pytesseract==0.3.13`)
OCR (Optical Character Recognition) via Tesseract.

| File | Usage |
|------|-------|
| `app/ai/providers/local/image_caption_cpu_v1.py` | Extract text from images via OCR |

---

### `opencv-python` / `cv2` (`opencv-python==4.13.0.90`)
Computer vision — video frame extraction, image processing.

| File | Usage |
|------|-------|
| `app/services/ingestion/media/media_hash_utils.py` | Frame-level perceptual hashing |
| `app/services/ingestion/media/video_ingestor_v1.py` | Extract frames from video files |

---

### `moviepy` (`moviepy==1.0.3`)
Video editing and audio extraction from video files.

| File | Usage |
|------|-------|
| `app/services/ingestion/media/video_ingestor_v1.py` | Extract audio track from video for transcription |

---

### `mutagen` (`mutagen==1.47.0`)
Audio file metadata reader (ID3 tags, duration, bitrate).

| File | Usage |
|------|-------|
| `app/services/ingestion/media/audio_ingestor_v1.py` | Read audio metadata before transcription |

---

### `librosa` (`librosa==0.11.0`)
Audio analysis and feature extraction.

| File | Usage |
|------|-------|
| `app/services/ingestion/media/media_hash_utils.py` | Audio fingerprint and spectral analysis |

---

### `ImageHash` (`ImageHash==4.3.1`)
Perceptual image hashing for near-duplicate detection.

| File | Usage |
|------|-------|
| `app/services/ingestion/media/media_hash_utils.py` | `phash()` / `dhash()` — image dedup |

---

### `pyacoustid` / `acoustid` (`pyacoustid==1.3.0`)
Audio fingerprinting via AcoustID/Chromaprint.

| File | Usage |
|------|-------|
| `app/services/ingestion/media/media_hash_utils.py` | Audio deduplication by acoustic fingerprint |

---

### `imageio` / `imageio-ffmpeg` (`ImageIO==2.37.2`, `imageio-ffmpeg==0.6.0`)
Image/video I/O with FFmpeg backend.
- **Used by**: `moviepy` internally for video reading/writing.

---

### `numpy` (`numpy==2.2.6`)
Numerical arrays. Core dependency of nearly all ML and media libraries.

| File | Usage |
|------|-------|
| `app/services/ingestion/media/media_hash_utils.py` | Array operations for audio/image features |
| `app/services/ingestion/media/video_ingestor_v1.py` | Frame array manipulation |
| `app/services/ingestion/deduplication_engine_v2*.py` | Vector similarity computations |

---

## 9. Web Scraping & HTTP

### `httpx` (`httpx==0.28.1`)
Async HTTP client (replaces `requests` for async code).

| File | Usage |
|------|-------|
| `app/core/connectors/auth/providers.py` | OAuth token requests |
| `app/core/vectordb/weaviate_v1.py` | Custom HTTP calls to Weaviate |
| `app/api/v2/ingestion_health.py` | Health probe HTTP calls |
| `app/services/ingestion/api_ingestor_v2.py` | Fetch data from external APIs |
| `app/services/ingestion/web_scraper_v2.py` | Fetch web pages for scraping |

---

### `aiohttp` (`aiohttp==3.13.2`)
Async HTTP client/server.

| File | Usage |
|------|-------|
| `app/services/ingestion/llm_rewriter.py` | Async HTTP calls to LLM rewrite endpoints |

---

### `aiofiles` (`aiofiles==25.1.0`)
Async file I/O.

| File | Usage |
|------|-------|
| `app/api/v2/ingestion_api_v2.py` | Async file saving during upload |
| `app/services/ingestion/file_router_v2.py` | Async file reading/writing |
| `app/services/ingestion/json_parser_v2.py` | Async JSON file reading |

---

### `beautifulsoup4` / `bs4` (`beautifulsoup4==4.14.2`)
HTML parsing and scraping.

| File | Usage |
|------|-------|
| `app/services/ingestion/text_cleaner_v2.py` | Strip HTML tags from ingested content |
| `app/services/ingestion/web_scraper_v2.py` | Parse scraped HTML |
| `app/utils/html_cleaner.py` | HTML → plain text conversion |
| `app/utils/text_cleaner.py` | Text cleaning utility |
| `app/utils/text_cleaner_v2.py` | Enhanced text cleaning |

---

### `readability-lxml` / `readability` (`readability-lxml==0.8.4.1`)
Extracts main article content from web pages (like browser reader mode).

| File | Usage |
|------|-------|
| `app/services/ingestion/web_scraper_v2.py` | Clean article body extraction from raw HTML |

---

### `requests` (`requests==2.32.5`)
Synchronous HTTP client.
- **Used by**: Several third-party SDKs (cohere, weaviate, pinecone) internally. App code itself uses `httpx` for async calls.

---

### `watchfiles` (`watchfiles==1.1.1`)
Cross-platform async file system watcher.

| File | Usage |
|------|-------|
| `app/services/ingestion/watcher_ingestor_v2.py` | Watch directories for new files to auto-ingest |

---

## 10. NLP & Text Processing

### `nltk` (`nltk==3.9.3`)
Natural Language Toolkit — tokenization, stemming, stop words.
- **Used by**: Text processing utilities. Required at runtime for `punkt` tokenizer and stopword lists.

---

### `rapidFuzz` (`rapidFuzz==3.14.3`)
Fast fuzzy string matching (Levenshtein, token set ratio).

| File | Usage |
|------|-------|
| `app/services/classification/canonicalizer.py` | Fuzzy taxonomy label matching and canonicalization |

---

### `langdetect` (`langdetect==1.0.9`)
Language detection from text.
- **Used by**: Text ingestion pipeline for multi-language content detection.

---

### `tiktoken` (`tiktoken==0.12.0`)
OpenAI tokenizer — count tokens for LLM context window management.
- **Used by**: LLM client utilities for prompt budget management.

---

### `html2text` (`html2text==2025.4.15`)
Converts HTML to Markdown/plain text.
- **Used by**: Content extraction utilities for cleaning scraped HTML.

---

### `lxml` / `lxml_html_clean` (`lxml==6.0.2`, `lxml_html_clean==0.4.3`)
Fast XML/HTML parser.
- **Used by**: `readability-lxml`, `beautifulsoup4` (as backend parser), and direct XML processing.

---

## 11. Data & Science Libraries

### `scikit-learn` (`scikit-learn==1.7.2`)
Machine learning utilities.
- **Used by**: Classification service for cosine similarity, TF-IDF, and clustering utilities.

---

### `scipy` (`scipy==1.15.3`)
Scientific computing.
- **Used by**: `scikit-learn` internally and any distance/similarity computations.

---

### `matplotlib` (`matplotlib==3.10.7`)
Plotting and visualization.
- **Used by**: Diagnostic/analysis scripts (not in runtime hot path).

---

### `huggingface-hub` (`huggingface-hub==0.36.0`)
HuggingFace model registry client — download models.
- **Used by**: `sentence-transformers`, `transformers`, `FlagEmbedding` for model downloads.

---

### `safetensors` (`safetensors==0.6.2`)
Safe model weight serialization format.
- **Used by**: `transformers` / `sentence-transformers` for loading model files.

---

### `tokenizers` (`tokenizers==0.22.1`)
Fast HuggingFace tokenizers (Rust-backed).
- **Used by**: `transformers` internally.

---

### `onnxruntime` (`onnxruntime==1.23.2`)
ONNX model inference runtime.
- **Used by**: `flashrank` for fast CPU-optimized reranker inference.

---

### `tenacity` (`tenacity==9.1.2`)
Retry library with exponential backoff.
- **Used by**: LLM client wrappers, API calls that need retry logic.

---

## 12. Configuration & Settings

### `pydantic-settings` (`pydantic-settings==2.13.1`)
Typed settings management reading from env vars / `.env` file.

| File | Usage |
|------|-------|
| `app/core/settings.py` | `AppSettings` singleton — all env vars typed and validated. Import as `from app.core.settings import settings` |

---

### `python-dotenv` (`python-dotenv==1.2.1`)
Loads `.env` file variables into `os.environ`.

| File | Usage |
|------|-------|
| `app/auth/generate_token.py` | `load_dotenv()` |
| `app/db/session_v2.py` | `load_dotenv()` |

---

### `PyYAML` (`PyYAML==6.0.3`)
YAML parser.

| File | Usage |
|------|-------|
| `app/core/config/client_config_schema.py` | `yaml.safe_load()` for client config files |

---

## 13. Observability & Monitoring

### `sentry-sdk` (`sentry-sdk==2.55.0`)
Error tracking and performance monitoring (Sentry.io).

| File | Usage |
|------|-------|
| `app/main.py` | `sentry_sdk.init(dsn=SENTRY_DSN)` — only activates when `SENTRY_DSN` is set in `.env`. Wrapped in `try/except ImportError` for graceful fallback. |

---

### `structlog` (`structlog==25.5.0`)
Structured logging with JSON output support.

| File | Usage |
|------|-------|
| `app/utils/logger.py` | JSON log mode when `LOG_FORMAT=json` env var is set. Falls back to colored console logging otherwise. |

---

### `slowapi` (`slowapi==0.1.9`) & `limits` (`limits==5.8.0`)
FastAPI rate limiting middleware.

| File | Usage |
|------|-------|
| `app/main.py` | `Limiter(key_func=get_remote_address)` — 200 req/min default. Override with `RATE_LIMIT_DEFAULT` env var. Wrapped in `try/except ImportError`. |

---

### `opentelemetry-*` (API, SDK, OTLP exporters)
Distributed tracing and metrics collection (OpenTelemetry standard).
- `opentelemetry-api==1.38.0`
- `opentelemetry-sdk==1.38.0`
- `opentelemetry-exporter-otlp-proto-grpc==1.38.0`
- **Used by**: `chromadb` and other observability-enabled SDKs internally.

---

### `psutil` (`psutil==7.1.2`)
System resource monitoring (CPU, RAM, GPU).

| File | Usage |
|------|-------|
| `app/services/ingestion/phantom_hardware_profiler.py` | Detect available CPU/memory for adaptive pipeline sizing |

---

### `aiocache` (`aiocache==0.12.3`)
Async caching layer (in-memory, Redis, memcached backends).
- **Used by**: Caching expensive LLM calls and embedding results. Available as `from aiocache import cached`.

---

## 14. File & System Utilities

### `aiofiles` (`aiofiles==25.1.0`)
Async file I/O — see [Web Scraping & HTTP](#9-web-scraping--http) section above.

---

### `python-multipart` (`python-multipart==0.0.20`)
Multipart form data parsing for file uploads.
- **Used by**: FastAPI for `UploadFile` / `Form(...)` endpoints. Required at runtime.

---

### `orjson` (`orjson==3.11.4`)
Ultra-fast JSON serialization/deserialization.
- **Used by**: FastAPI for JSON responses (significant throughput improvement over stdlib `json`).

---

### `colorama` (`colorama==0.4.6`)
ANSI color output on Windows terminals.
- **Used by**: `app/utils/logger.py` ColorFormatter for terminal log coloring.

---

### `coloredlogs` (`coloredlogs==15.0.1`)
Colored log output.
- **Used by**: Logger utilities for formatted console output.

---

### `rich` (`rich==14.2.0`)
Rich terminal output (tables, progress bars, syntax highlighting).
- **Used by**: CLI utilities and diagnostic scripts.

---

### `typer` (`typer==0.20.0`)
CLI framework built on Click.
- **Used by**: `retrieve_cli.py` and other CLI entry points.

---

### `click` (`click==8.3.0`)
CLI argument parsing.
- **Used by**: `typer` internally.

---

### `certifi` (`certifi==2025.10.5`)
SSL certificate bundle.
- **Used by**: `httpx`, `requests`, and all HTTPS clients.

---

## 15. Testing

### `pytest` (`pytest==9.0.1`) & `pytest-cov` (`pytest-cov==7.0.0`)
Test runner with coverage reporting.

| File | Usage |
|------|-------|
| `tests/` directory | All unit and integration tests |
| `run_dedup_tests.py` | Deduplication test runner |
| `test_*.py` (root level) | Connection and integrity tests |

---

### `coverage` (`coverage==7.11.3`)
Code coverage measurement.
- **Used by**: `pytest-cov` plugin.

---

## 16. Requirements Summary Table

| Library | Version | Category | Active Files |
|---------|---------|----------|-------------|
| `fastapi` | 0.120.0 | Web Framework | 20+ files |
| `sqlalchemy` | 2.0.44 | Database ORM | 30+ files |
| `pydantic` | 2.12.3 | Validation | 8 files |
| `pydantic-settings` | 2.13.1 | Config | `core/settings.py` |
| `authlib` | >=1.3.0 | Auth/JWT | `auth/generate_token.py`, `auth/deps.py` |
| `chromadb` | 1.3.0 | Vector DB (default) | 7 files |
| `pymilvus` | 2.6.10 | Vector DB | `core/vectordb/milvus_v1.py` |
| `pinecone-client` | 6.0.0 | Vector DB | `core/vectordb/pinecone_v1.py` |
| `qdrant-client` | >=1.9.0 | Vector DB | `core/vectordb/qdrant_v1.py` |
| `weaviate-client` | 4.20.4 | Vector DB | `core/vectordb/weaviate_v1.py` |
| `redis[hiredis]` | >=5.0 | Vector DB + Cache | `redis_v1.py`, health check |
| `sentence-transformers` | 5.1.2 | Embeddings | 5+ files |
| `cohere` | 5.20.7 | Embeddings + Rerank | 2 files |
| `FlagEmbedding` | 1.3.5 | Reranking | `rerankers/bge_reranker_v1.py` |
| `flashrank` | 0.2.10 | Reranking | `rerankers/flashrank_v1.py` |
| `ragatouille` | >=0.0.8 | Reranking | `rerankers/colbert_v1.py` |
| `torch` | 2.9.0 | ML Backend | 3 files |
| `anthropic` | 0.85.0 | LLM | `llms/anthropic_v1.py` |
| `openai` | 2.6.1 | LLM + Embeddings | 2 files |
| `google-generativeai` | >=0.8.0 | LLM | `llms/gemini_v1.py` |
| `groq` | >=0.9.0 | LLM | `llms/groq_v1.py` |
| `ollama` | >=0.4.0 | LLM + Embeddings | 2 files |
| `openai-whisper` | 20250625 | Audio Transcription | 4 files |
| `PyMuPDF` (fitz) | 1.26.5 | PDF Parsing | 2 files |
| `pdfplumber` | 0.11.8 | PDF Tables | `pdf_parser_v2.py` |
| `python-docx` | 1.2.0 | DOCX Parsing | 2 files |
| `pandas` | 2.3.3 | Tabular Data | 6 files |
| `httpx` | 0.28.1 | Async HTTP | 5 files |
| `aiofiles` | 25.1.0 | Async File I/O | 3 files |
| `beautifulsoup4` | 4.14.2 | HTML Parsing | 5 files |
| `readability-lxml` | 0.8.4.1 | Web Scraping | `web_scraper_v2.py` |
| `feedparser` | 6.0.12 | RSS | `rss_ingestor_v2.py` |
| `Pillow` | 11.3.0 | Image Processing | 3 files |
| `opencv-python` | 4.13.0.90 | Video/Image CV | 2 files |
| `moviepy` | 1.0.3 | Video Processing | `video_ingestor_v1.py` |
| `mutagen` | 1.47.0 | Audio Metadata | `audio_ingestor_v1.py` |
| `librosa` | 0.11.0 | Audio Analysis | `media_hash_utils.py` |
| `ImageHash` | 4.3.1 | Image Dedup | `media_hash_utils.py` |
| `rapidFuzz` | 3.14.3 | Fuzzy Matching | `canonicalizer.py` |
| `sentry-sdk` | 2.55.0 | Error Tracking | `main.py` |
| `structlog` | 25.5.0 | Structured Logging | `utils/logger.py` |
| `slowapi` | 0.1.9 | Rate Limiting | `main.py` |
| `aiocache` | 0.12.3 | Async Caching | Available, not yet wired |
| `alembic` | 1.17.2 | DB Migrations | `alembic/` |
| `uvicorn` | 0.38.0 | ASGI Server | Runtime only |
| `pytest` | 9.0.1 | Testing | `tests/` |

---

## Notes

### Pluggable Vector DB
The vector DB backend is selected at runtime via the `MAI_VECTORDB` environment variable. All adapters (`chroma_v1.py`, `milvus_v1.py`, etc.) implement the same base interface in `app/core/vectordb/base.py`. Only the selected backend is actively used.

### Pluggable LLM / Embedder / Reranker
Same pattern applies to LLMs (`app/core/llms/`), embedders (`app/core/embedders/`), and rerankers (`app/core/rerankers/`). Each has a `register.py` that maps name strings to implementations.

### Enterprise Libraries (Graceful Fallback)
`sentry-sdk`, `slowapi`, and `uvloop` are all wrapped in `try/except ImportError` in `main.py`. If not installed, the server still starts normally — error tracking and rate limiting are simply disabled.

### Copy/Backup Files
Files with ` - Copy.py`, `-before-*`, `-latest-*` suffixes are historical snapshots, not active code. They do not affect the running application.
