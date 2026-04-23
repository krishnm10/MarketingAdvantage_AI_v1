# ============================================================
# app/api/v2/ingestion_health.py  —  Comprehensive System Health (v2)
#
# IMPORTANT: This file must NEVER import gRPC-based VDB client
# libraries (pymilvus, qdrant-client, weaviate-client) because
# their internal event-loops deadlock inside FastAPI's asyncio
# loop.  All network checks use httpx / raw sockets only.
# ============================================================
from __future__ import annotations

from fastapi import APIRouter, Query
import asyncio
import datetime
import os
import socket

router = APIRouter(prefix="/api/v2/ingestion", tags=["Ingestion Health"])

_T = 4  # default per-service timeout in seconds


# ── helpers ───────────────────────────────────────────────────────────

def _ok(msg: str) -> dict:
    return {"status": "online", "message": msg}

def _fail(msg: str) -> dict:
    return {"status": "offline", "message": msg[:200]}

def _skip(msg: str) -> dict:
    return {"status": "not_configured", "message": msg}

def _tcp_reachable(host: str, port: int, timeout: float = 2) -> bool:
    """True if a TCP connect succeeds within *timeout* seconds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
        return True
    except Exception:
        return False


# ── async httpx helper (no threads, native async) ────────────────────

async def _http_get(url: str, *, headers: dict | None = None,
                    timeout: float = _T) -> dict | None:
    """GET via httpx.AsyncClient.  Returns parsed JSON or None."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(url, headers=headers or {})
            if r.status_code < 400:
                return r.json()
            return None
    except Exception:
        return None

async def _http_post(url: str, *, json: dict, headers: dict | None = None,
                     timeout: float = _T) -> tuple[int, dict | None]:
    """POST via httpx.AsyncClient.  Returns (status_code, json | None)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(url, json=json, headers=headers or {})
            return r.status_code, (r.json() if r.status_code < 500 else None)
    except Exception as e:
        return 0, {"error": str(e)[:200]}


# ══════════════════════════════════════════════════════════════════════
#  1. DATABASES
# ══════════════════════════════════════════════════════════════════════

async def _check_postgres() -> dict:
    try:
        from app.db.session_v2 import async_engine
        from sqlalchemy import text
        async with async_engine.connect() as conn:
            await asyncio.wait_for(conn.execute(text("SELECT 1")), timeout=_T)
        return _ok("Connected")
    except asyncio.TimeoutError:
        return _fail(f"Timed out ({_T}s)")
    except Exception as e:
        return _fail(str(e))


# ══════════════════════════════════════════════════════════════════════
#  2. VECTOR DATABASES — pure httpx / socket checks (NO client libs)
# ══════════════════════════════════════════════════════════════════════

async def _check_qdrant() -> dict:
    """Qdrant REST API   GET /collections"""
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    url  = os.getenv("QDRANT_URL") or f"http://{host}:{port}"
    api_key = os.getenv("QDRANT_API_KEY") or None
    hdrs = {"api-key": api_key} if api_key else {}
    data = await _http_get(f"{url}/collections", headers=hdrs)
    if data and "result" in data:
        n = len(data["result"].get("collections", []))
        return _ok(f"{n} collection(s) | {host}:{port}")
    return _fail(f"Cannot reach {host}:{port}")


async def _check_chroma() -> dict:
    """ChromaDB — remote HttpClient OR local PersistentClient.

    Remote mode: CHROMA_HOST is set → use httpx to probe heartbeat and
                 collections endpoints.  Tries v2 API first (ChromaDB
                 ≥ 1.0 removed v1), then falls back to v1 for older
                 servers.
    Local mode:  CHROMA_HOST is empty → use PersistentClient singleton.
    """
    chroma_host = os.getenv("CHROMA_HOST") or None

    if chroma_host:
        # ── Remote mode — httpx (no client libs, no gRPC) ─────────
        port    = int(os.getenv("CHROMA_PORT") or "8000")
        use_ssl = os.getenv("CHROMA_SSL", "").lower() in ("1", "true", "yes")
        scheme  = "https" if use_ssl else "http"
        url     = f"{scheme}://{chroma_host}:{port}"
        api_key = os.getenv("CHROMA_API_KEY") or None
        hdrs    = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        tenant  = os.getenv("CHROMA_TENANT", "default_tenant")
        database = os.getenv("CHROMA_DATABASE", "default_database")

        # 1) heartbeat — try v2 first, fall back to v1
        hb = await _http_get(f"{url}/api/v2/heartbeat", headers=hdrs)
        if hb is None:
            hb = await _http_get(f"{url}/api/v1/heartbeat", headers=hdrs)
        if hb is None:
            return _fail(f"Cannot reach {chroma_host}:{port}")

        # 2) collection count — v2 path-based, then v1 query-param
        coll_data = await _http_get(
            f"{url}/api/v2/tenants/{tenant}/databases/{database}/collections",
            headers=hdrs,
        )
        if coll_data is None:
            coll_data = await _http_get(
                f"{url}/api/v1/collections?tenant={tenant}&database={database}",
                headers=hdrs,
            )
        if coll_data is not None:
            n = len(coll_data) if isinstance(coll_data, list) else 0
            return _ok(f"{n} collection(s) | {chroma_host}:{port}")
        # heartbeat OK but collections endpoint failed — still online
        return _ok(f"Connected | {chroma_host}:{port}")

    # ── Local mode — PersistentClient singleton ───────────────────
    # MUST use identical Settings as chroma_v1.py (anonymized_telemetry=False)
    # so that chromadb returns the existing singleton instead of raising
    # 'instance already exists with different settings'.
    try:
        import chromadb
        from chromadb.config import Settings
        chroma_path = os.getenv("CHROMA_PATH") or "./pluggable_db"
        client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        colls = client.list_collections()
        n = len(colls) if isinstance(colls, (list, tuple)) else 0
        return _ok(f"{n} collection(s) | {chroma_path}")
    except Exception as e:
        return _fail(str(e))


async def _check_milvus() -> dict:
    """Milvus — TCP socket check + optional REST health (port 9091).
    NEVER imports pymilvus here (gRPC deadlocks w/ asyncio)."""
    host = os.getenv("MILVUS_HOST", "localhost")
    port = int(os.getenv("MILVUS_PORT", "19530"))
    uri  = os.getenv("MILVUS_URI")

    target_host, target_port = host, port
    if uri:
        from urllib.parse import urlparse
        p = urlparse(uri if "://" in uri else f"http://{uri}")
        target_host = p.hostname or host
        target_port = p.port or port

    # 1) fast TCP probe on gRPC port
    tcp_ok = await asyncio.get_event_loop().run_in_executor(
        None, _tcp_reachable, target_host, target_port, 3
    )
    if not tcp_ok:
        return _fail(f"Cannot reach {target_host}:{target_port}")

    # 2) try REST health endpoint (Milvus 2.3+ exposes port 9091)
    web_port = int(os.getenv("MILVUS_WEB_PORT", "9091"))
    data = await _http_get(f"http://{target_host}:{web_port}/healthz", timeout=3)
    if data is not None:
        return _ok(f"Healthy | {target_host}:{target_port}")

    # TCP connected but REST unavailable — still report online
    return _ok(f"Reachable (gRPC) | {target_host}:{target_port}")


async def _check_pinecone() -> dict:
    if os.getenv("PINECONE_MODE", "cloud").strip().lower() == "local":
        local_path = os.getenv("PINECONE_LOCAL_PATH") or "./pinecone_local_db"
        return _ok(f"Local emulation | {local_path}")
    api_key = os.getenv("PINECONE_API_KEY", "")
    if not api_key:
        return _skip("PINECONE_API_KEY not set")
    data = await _http_get(
        "https://api.pinecone.io/indexes",
        headers={"Api-Key": api_key},
        timeout=_T + 3,
    )
    if data and "indexes" in data:
        n = len(data["indexes"])
        return _ok(f"{n} index(es) | {os.getenv('PINECONE_REGION','us-east-1')}")
    if data is None:
        return _fail("Cannot reach Pinecone API")
    return _fail("Unexpected response")


async def _check_weaviate() -> dict:
    url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    api_key = os.getenv("WEAVIATE_API_KEY") or None
    hdrs = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    import httpx

    try:
        async with httpx.AsyncClient(timeout=_T, trust_env=False) as client:
            resp = await client.get(f"{url}/v1/.well-known/ready", headers=hdrs)
            if resp.status_code < 400:
                return _ok(f"Ready | {url}")

            # Fallback for deployments exposing the meta endpoint but not the ready endpoint.
            meta = await client.get(f"{url}/v1/meta", headers=hdrs)
            if meta.status_code < 400:
                return _ok(f"Connected | {url}")
    except Exception:
        pass

    return _fail(f"Cannot reach {url}")


async def _check_redis() -> dict:
    url = os.getenv("REDIS_URL")
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    # quick TCP probe
    reachable = await asyncio.get_event_loop().run_in_executor(
        None, _tcp_reachable, host, port, 2
    )
    if not reachable:
        return _fail(f"Cannot reach {host}:{port}")
    client = None
    try:
        import redis as redis_lib
        if url:
            client = redis_lib.Redis.from_url(url, socket_timeout=_T)
        else:
            client = redis_lib.Redis(
                host=host,
                port=port,
                password=os.getenv("REDIS_PASSWORD") or None,
                socket_timeout=_T,
            )
        pong = client.ping()
        addr = url or f"{host}:{port}"
        return _ok(f"Connected | {addr}") if pong else _fail("Ping failed")
    except ImportError:
        return {"status": "not_installed", "message": "redis-py not installed"}
    except Exception as e:
        return _fail(str(e))
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
            try:
                client.connection_pool.disconnect()
            except Exception:
                pass


_VECTORDB_CHECKS = {
    "qdrant":   _check_qdrant,
    "chroma":   _check_chroma,
    "pinecone": _check_pinecone,
    "milvus":   _check_milvus,
    "weaviate": _check_weaviate,
    "redis":    _check_redis,
}


# ══════════════════════════════════════════════════════════════════════
#  3. EMBEDDERS
# ══════════════════════════════════════════════════════════════════════

async def _check_emb_huggingface() -> dict:
    model = os.getenv("HF_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
    try:
        cache_dir = os.path.join(
            os.getenv("SENTENCE_TRANSFORMERS_HOME",
                      os.path.join(os.path.expanduser("~"), ".cache", "torch", "sentence_transformers")),
        )
        model_safe = model.replace("/", "_")
        cached = any(
            model_safe in d or model in d
            for d in (os.listdir(cache_dir) if os.path.isdir(cache_dir) else [])
        )
        if cached:
            return _ok(f"{model} | cached locally")
        # check if library is importable
        import importlib
        if importlib.util.find_spec("sentence_transformers"):
            return _ok(f"{model} | library installed")
        return {"status": "not_installed", "message": "sentence-transformers not installed"}
    except Exception as e:
        return _fail(str(e))


async def _check_emb_ollama() -> dict:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model    = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    code, data = await _http_post(
        f"{base_url}/api/embeddings",
        json={"model": model, "prompt": "health"},
        timeout=_T + 5,
    )
    if code == 200 and data:
        dim = len(data.get("embedding", []))
        return _ok(f"{model} | dim={dim}")
    return _fail(f"HTTP {code}" if code else "Ollama unreachable")


async def _check_emb_openai() -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return _skip("OPENAI_API_KEY not set")
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    code, data = await _http_post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "input": "health"},
        timeout=_T + 5,
    )
    if code == 200 and data:
        dim = len(data["data"][0]["embedding"])
        return _ok(f"{model} | dim={dim}")
    return _fail(f"HTTP {code}")


async def _check_emb_cohere() -> dict:
    api_key = os.getenv("COHERE_API_KEY", "")
    if not api_key:
        return _skip("COHERE_API_KEY not set")
    model = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
    code, data = await _http_post(
        "https://api.cohere.ai/v1/embed",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "texts": ["health"], "input_type": "search_document"},
        timeout=_T + 5,
    )
    if code == 200 and data:
        dim = len(data["embeddings"][0])
        return _ok(f"{model} | dim={dim}")
    return _fail(f"HTTP {code}")


async def _check_emb_google() -> dict:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return _skip("GEMINI_API_KEY not set")
    model = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    # Use the embedContent endpoint to validate the model and key
    code, data = await _http_post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={api_key}",
        json={"content": {"parts": [{"text": "health"}]}, "model": f"models/{model}"},
        timeout=_T + 5,
    )
    if code == 200 and data:
        dim = len((data.get("embedding") or {}).get("values") or [])
        return _ok(f"{model} | dim={dim}")
    return _fail(f"HTTP {code}" if code else "Google API unreachable")


_EMBEDDER_CHECKS = {
    "huggingface": _check_emb_huggingface,
    "ollama":      _check_emb_ollama,
    "openai":      _check_emb_openai,
    "cohere":      _check_emb_cohere,
    "google":      _check_emb_google,    # Google Gemini embedder
    "gemini":      _check_emb_google,    # alias used by some configs
}


# ══════════════════════════════════════════════════════════════════════
#  4. LLMs
# ══════════════════════════════════════════════════════════════════════

async def _check_llm_ollama() -> dict:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model    = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
    data = await _http_get(f"{base_url}/api/tags", timeout=_T)
    if data:
        models = data.get("models", [])
        names  = [m.get("name", "") for m in models]
        found  = model in names or any(model in n for n in names)
        return _ok(f"{model} | {len(models)} model(s)" + (" | loaded" if found else " | NOT loaded"))
    return _fail("Ollama unreachable")


async def _check_llm_openai() -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return _skip("OPENAI_API_KEY not set")
    model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    data = await _http_get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=_T + 5,
    )
    return _ok(f"{model} | API key valid") if data else _fail("OpenAI unreachable")


async def _check_llm_groq() -> dict:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return _skip("GROQ_API_KEY not set")
    model = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")
    data = await _http_get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=_T + 5,
    )
    return _ok(f"{model} | API key valid") if data else _fail("Groq unreachable")


async def _check_llm_anthropic() -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _skip("ANTHROPIC_API_KEY not set")
    model = os.getenv("ANTHROPIC_LLM_MODEL", "claude-3-5-sonnet-20241022")
    code, data = await _http_post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={"model": model, "max_tokens": 1,
              "messages": [{"role": "user", "content": "hi"}]},
        timeout=_T + 5,
    )
    if code == 200:
        return _ok(f"{model} | API key valid")
    if code == 429:
        return _ok(f"{model} | rate-limited but reachable")
    return _fail(f"HTTP {code}")


async def _check_llm_gemini() -> dict:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return _skip("GEMINI_API_KEY not set")
    model = os.getenv("GEMINI_LLM_MODEL", "gemini-1.5-flash")
    data = await _http_get(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
        timeout=_T + 5,
    )
    if data:
        n = len(data.get("models", []))
        return _ok(f"{model} | {n} model(s)")
    return _fail("Gemini unreachable")


_LLM_CHECKS = {
    "ollama":    _check_llm_ollama,
    "openai":    _check_llm_openai,
    "groq":      _check_llm_groq,
    "grok":      _check_llm_groq,       # alias
    "anthropic": _check_llm_anthropic,
    "gemini":    _check_llm_gemini,
    "google":    _check_llm_gemini,     # alias — google maps to Gemini LLM
}


# ══════════════════════════════════════════════════════════════════════
#  5. MAIN HEALTH ENDPOINT
# ══════════════════════════════════════════════════════════════════════

# What makes a backend "worth pinging"?
_VDB_CONFIGURED = {
    "qdrant":   lambda: True,
    "chroma":   lambda: True,
    "milvus":   lambda: bool(os.getenv("MILVUS_HOST") and os.getenv("MILVUS_HOST") != "localhost") or bool(os.getenv("MILVUS_URI")),
    "pinecone": lambda: os.getenv("PINECONE_MODE", "cloud").strip().lower() == "local" or bool(os.getenv("PINECONE_API_KEY")),
    "weaviate": lambda: bool(os.getenv("WEAVIATE_API_KEY")) or (os.getenv("WEAVIATE_URL", "").replace("http://localhost:8080", "") != ""),
    "redis":    lambda: bool(os.getenv("REDIS_URL")) or bool(os.getenv("REDIS_PASSWORD")),
}

_EMB_CONFIGURED = {
    "huggingface": lambda: True,
    "ollama":      lambda: True,
    "openai":      lambda: bool(os.getenv("OPENAI_API_KEY")),
    "cohere":      lambda: bool(os.getenv("COHERE_API_KEY")),
    "google":      lambda: bool(os.getenv("GEMINI_API_KEY")),
    "gemini":      lambda: bool(os.getenv("GEMINI_API_KEY")),
}

_LLM_CONFIGURED = {
    "ollama":    lambda: True,
    "openai":    lambda: bool(os.getenv("OPENAI_API_KEY")),
    "groq":      lambda: bool(os.getenv("GROQ_API_KEY")),
    "grok":      lambda: bool(os.getenv("GROQ_API_KEY")),
    "anthropic": lambda: bool(os.getenv("ANTHROPIC_API_KEY")),
    "gemini":    lambda: bool(os.getenv("GEMINI_API_KEY")),
    "google":    lambda: bool(os.getenv("GEMINI_API_KEY")),
}


@router.get("/health")
async def ingestion_health(
    scope: str = Query(
        "configured",
        description="Health scope: active | configured | all",
    )
):
    """
    Comprehensive health — pings active + configured services only.
    All checks are native-async (httpx / socket). No gRPC libs imported.
    Typically completes in 1-5 s.
    """
    active_vdb = os.getenv("MAI_VECTORDB", "qdrant").lower()
    active_emb = os.getenv("MAI_EMBEDDER", "huggingface").lower()
    active_llm = os.getenv("MAI_LLM", "ollama").lower()
    mode = (scope or "configured").strip().lower()
    if mode not in {"active", "configured", "all"}:
        mode = "configured"

    def _should_check(name: str, active_name: str, cfg_map: dict) -> bool:
        if mode == "all":
            return True
        if mode == "active":
            return name == active_name
        return name == active_name or cfg_map.get(name, lambda: False)()

    # ── collect coroutines ───────────────────────────────────────────
    tasks: dict[str, any] = {}
    tasks["db__postgresql"] = _check_postgres()

    for name, fn in _VECTORDB_CHECKS.items():
        if _should_check(name, active_vdb, _VDB_CONFIGURED):
            tasks[f"vdb__{name}"] = fn()

    for name, fn in _EMBEDDER_CHECKS.items():
        if _should_check(name, active_emb, _EMB_CONFIGURED):
            tasks[f"emb__{name}"] = fn()

    seen: set = set()
    for name, fn in _LLM_CHECKS.items():
        if fn in seen:
            continue
        if _should_check(name, active_llm, _LLM_CONFIGURED):
            tasks[f"llm__{name}"] = fn()
            seen.add(fn)

    # ── Celery broker + worker health (sync, run in executor) ──────
    loop = asyncio.get_event_loop()
    tasks["infra__broker"] = loop.run_in_executor(None, _check_celery_broker_sync)
    worker_future = loop.run_in_executor(None, _check_celery_worker_sync)

    # ── execute with a hard overall timeout ──────────────────────────
    keys = list(tasks.keys())
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks.values(), return_exceptions=True),
            timeout=15,
        )
    except asyncio.TimeoutError:
        results = [{"status": "offline", "message": "Global timeout (15 s)"}] * len(keys)

    flat: dict = {}
    for k, r in zip(keys, results):
        if isinstance(r, Exception):
            flat[k] = _fail(str(r))
        elif isinstance(r, dict):
            flat[k] = r
        else:
            flat[k] = _fail("Unexpected result")

    # ── build categorised response ───────────────────────────────────
    def _entry(key, prefix, checks, active_name, cfg_map):
        tk = f"{prefix}__{key}"
        is_active = key == active_name
        if tk in flat:
            return {**flat[tk], "active": is_active}
        cfg = cfg_map.get(key, lambda: False)
        if not cfg():
            return {"status": "not_configured", "message": "Not configured", "active": is_active}
        return {"status": "offline", "message": "Skipped", "active": is_active}

    databases = {
        "postgresql": {**(flat.get("db__postgresql", _fail("Unknown"))), "active": True}
    }
    vectordbs = {k: _entry(k, "vdb", _VECTORDB_CHECKS, active_vdb, _VDB_CONFIGURED) for k in _VECTORDB_CHECKS}
    embedders = {k: _entry(k, "emb", _EMBEDDER_CHECKS, active_emb, _EMB_CONFIGURED) for k in _EMBEDDER_CHECKS}

    llms: dict = {}
    seen2: set = set()
    for k in _LLM_CHECKS:
        if _LLM_CHECKS[k] in seen2 and k != active_llm:
            continue
        llms[k] = _entry(k, "llm", _LLM_CHECKS, active_llm, _LLM_CONFIGURED)
        seen2.add(_LLM_CHECKS[k])

    active_entries = [
        databases.get("postgresql", {}),
        vectordbs.get(active_vdb, {}),
        embedders.get(active_emb, {}),
        llms.get(active_llm, {}),
    ]
    all_ok = all(s.get("status") == "online" for s in active_entries if s)

    # ── Infrastructure (Celery + broker + Kafka events) ────────────
    from app.worker.broker_config import is_celery_enabled
    broker_name = os.getenv("CELERY_BROKER", "redis").lower()
    try:
        worker_result = await asyncio.wait_for(worker_future, timeout=5)
    except asyncio.TimeoutError:
        worker_result = _fail("Worker check timed out")

    # Kafka event streaming health (independent of Celery broker)
    kafka_events_status = _skip("KAFKA_EVENTS_ENABLED is not true")
    kafka_events_enabled = os.getenv("KAFKA_EVENTS_ENABLED", "false").strip().lower() in ("true", "1", "yes")
    if kafka_events_enabled:
        try:
            from app.services.kafka import kafka_service
            kafka_events_status = await asyncio.wait_for(
                kafka_service.health_check(), timeout=8
            )
        except asyncio.TimeoutError:
            kafka_events_status = _fail("Kafka health check timed out")
        except Exception as e:
            kafka_events_status = _fail(str(e))

    infrastructure = {
        "celery_enabled": is_celery_enabled(),
        "broker_type": broker_name if is_celery_enabled() else "none",
        "broker": flat.get("infra__broker", _skip("Not checked")),
        "worker": worker_result,
        "kafka_events": {
            "enabled": kafka_events_enabled,
            **( kafka_events_status if isinstance(kafka_events_status, dict) else {"status": "unknown"}),
        },
    }

    return {
        "status": "ok" if all_ok else "degraded",
        "scope": mode,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "active": {"vectordb": active_vdb, "embedder": active_emb, "llm": active_llm},
        "databases": databases,
        "vectordbs": vectordbs,
        "embedders": embedders,
        "llms":      llms,
        "infrastructure": infrastructure,
    }


# ══════════════════════════════════════════════════════════════════════
#  6. PIPELINE CONFIG — returns active .env selections for the frontend
# ══════════════════════════════════════════════════════════════════════

@router.get("/pipeline-config")
async def pipeline_config():
    """
    Returns the active pipeline configuration from .env.
    Used by the frontend Pipeline page to render the Data Flow dynamically.
    """
    from app.worker.broker_config import is_celery_enabled
    broker_name = os.getenv("CELERY_BROKER", "redis").lower()

    return {
        "vectordb":  os.getenv("MAI_VECTORDB", "chroma").lower(),
        "embedder":  os.getenv("MAI_EMBEDDER", "ollama").lower(),
        "llm":       os.getenv("MAI_LLM", "ollama").lower(),
        "reranker":  os.getenv("MAI_RERANKER", "none").lower(),
        "broker":    broker_name if is_celery_enabled() else "none",
        "celery_enabled": is_celery_enabled(),
        "kafka_events_enabled": os.getenv("KAFKA_EVENTS_ENABLED", "false").strip().lower() in ("true", "1", "yes"),
        "kafka_bootstrap_servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092") if os.getenv("KAFKA_EVENTS_ENABLED", "false").strip().lower() in ("true", "1", "yes") else None,
    }


# ══════════════════════════════════════════════════════════════════════
#  7. CELERY & BROKER HEALTH
# ══════════════════════════════════════════════════════════════════════

def _redis_host_port() -> tuple:
    """Extract Redis host and port from env vars, parsing the URL if set."""
    redis_url = os.getenv("CELERY_REDIS_URL", "") or os.getenv("REDIS_URL", "")
    if redis_url:
        from urllib.parse import urlparse
        parsed = urlparse(redis_url)
        return parsed.hostname or "localhost", parsed.port or 6379
    return os.getenv("CELERY_REDIS_HOST", "localhost"), int(os.getenv("CELERY_REDIS_PORT", "6379"))


def _check_celery_broker_sync() -> dict:
    """Check the message broker used by Celery (Redis, RabbitMQ, etc.).
    Synchronous — intended to be called via run_in_executor."""
    from app.worker.broker_config import is_celery_enabled
    if not is_celery_enabled():
        return _skip("Celery disabled (CELERY_ENABLED!=true)")

    broker_name = os.getenv("CELERY_BROKER", "redis").lower()

    if broker_name in ("redis", "redis_streams", "upstash"):
        host, port = _redis_host_port()
        addr = f"{host}:{port}"
        # Fast TCP pre-check to avoid long OS-level SYN timeout on unreachable hosts
        if not _tcp_reachable(host, port, 3):
            return _fail(f"{broker_name} | {addr} unreachable")
        redis_url = os.getenv("CELERY_REDIS_URL", "") or os.getenv("REDIS_URL", "")
        try:
            import redis as redis_lib
            if redis_url:
                client = redis_lib.Redis.from_url(redis_url, socket_timeout=_T, socket_connect_timeout=3)
            else:
                client = redis_lib.Redis(host=host, port=port, socket_timeout=_T, socket_connect_timeout=3)
            pong = client.ping()
            client.close()
            return _ok(f"{broker_name} | {addr}") if pong else _fail("Ping failed")
        except Exception as e:
            return _fail(f"{broker_name}: {e}")

    elif broker_name == "rabbitmq":
        host = os.getenv("RABBITMQ_HOST", "localhost")
        port = int(os.getenv("RABBITMQ_PORT", "5672"))
        reachable = _tcp_reachable(host, port, 3)
        return _ok(f"RabbitMQ | {host}:{port}") if reachable else _fail(f"Cannot reach {host}:{port}")

    elif broker_name in ("kafka", "redpanda", "warpstream", "aiven"):
        servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        # Parse multiple bootstrap servers — probe the first one
        first_server = servers.split(",")[0].strip()
        host, _, port_str = first_server.partition(":")
        port = int(port_str) if port_str else 9092
        reachable = _tcp_reachable(host, port, 3)
        if not reachable:
            return _fail(f"Cannot reach {servers}")

        # Deep health check — query broker metadata if confluent_kafka is available
        try:
            from confluent_kafka.admin import AdminClient
            import time as _time
            admin_config = {
                "bootstrap.servers": servers,
                "client.id": "mai-health-check",
            }
            protocol = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT").strip()
            if protocol != "PLAINTEXT":
                admin_config["security.protocol"] = protocol
            mech = os.getenv("KAFKA_SASL_MECHANISM", "").strip()
            if mech:
                admin_config["sasl.mechanism"] = mech
                admin_config["sasl.username"] = os.getenv("KAFKA_SASL_USERNAME", "").strip()
                admin_config["sasl.password"] = os.getenv("KAFKA_SASL_PASSWORD", "").strip()
            if protocol in ("SSL", "SASL_SSL"):
                ssl_ca = os.getenv("KAFKA_SSL_CA_LOCATION", "").strip()
                if ssl_ca:
                    admin_config["ssl.ca.location"] = ssl_ca

            start = _time.monotonic()
            admin = AdminClient(admin_config)
            metadata = admin.list_topics(timeout=5)
            latency_ms = (_time.monotonic() - start) * 1000

            broker_count = len(metadata.brokers)
            topic_count = len([t for t in metadata.topics if not t.startswith("_")])
            cluster_id = metadata.cluster_id or "unknown"

            return _ok(
                f"{broker_name} | {servers} | "
                f"{broker_count} broker(s) | "
                f"{topic_count} topic(s) | "
                f"cluster={cluster_id} | "
                f"{latency_ms:.0f}ms"
            )
        except ImportError:
            # confluent_kafka not installed — fall back to TCP-only check
            return _ok(f"{broker_name} | {servers} (TCP reachable)")
        except Exception as e:
            # Metadata query failed but TCP was OK
            return _ok(f"{broker_name} | {servers} (TCP OK, metadata error: {str(e)[:100]})")

    elif broker_name == "sqs":
        return _ok("SQS | AWS managed")

    elif broker_name == "pubsub":
        return _ok("Pub/Sub | GCP managed")

    elif broker_name == "eventhubs":
        return _ok("Event Hubs | Azure managed")

    else:
        # Generic TCP check for unknown brokers
        return _ok(f"{broker_name} | configured")


def _check_celery_worker_sync() -> dict:
    """
    Check if a Celery worker is connected by inspecting the broker's
    Celery-internal control exchange. Uses Celery's ping() with a short timeout.
    This is intentionally synchronous — Celery's inspector API is blocking.
    """
    from app.worker.broker_config import is_celery_enabled
    if not is_celery_enabled():
        return _skip("Celery disabled (CELERY_ENABLED!=true)")

    # Fast pre-check: if broker host is unreachable, skip the slow Celery inspect
    broker_name = os.getenv("CELERY_BROKER", "redis").lower()
    if broker_name in ("redis", "redis_streams", "upstash"):
        host, port = _redis_host_port()
        if not _tcp_reachable(host, port, 3):
            return _fail(f"Broker ({host}:{port}) unreachable — cannot inspect workers")
    elif broker_name == "rabbitmq":
        host = os.getenv("RABBITMQ_HOST", "localhost")
        port = int(os.getenv("RABBITMQ_PORT", "5672"))
        if not _tcp_reachable(host, port, 3):
            return _fail(f"Broker ({host}:{port}) unreachable — cannot inspect workers")

    try:
        from app.worker.celery_app import celery_app
        inspector = celery_app.control.inspect(timeout=3)
        ping_result = inspector.ping()
        if ping_result:
            worker_names = list(ping_result.keys())
            return _ok(f"{len(worker_names)} worker(s) | {', '.join(worker_names)}")
        return _fail("No workers responding")
    except Exception as e:
        return _fail(f"Inspect failed: {e}")


@router.get("/health/celery")
async def celery_health():
    """
    Standalone Celery + broker health endpoint.
    Also included in the main /health response.
    """
    loop = asyncio.get_event_loop()
    broker_future = loop.run_in_executor(None, _check_celery_broker_sync)
    worker_future = loop.run_in_executor(None, _check_celery_worker_sync)

    try:
        broker_result = await asyncio.wait_for(broker_future, timeout=8)
    except asyncio.TimeoutError:
        broker_result = _fail("Broker check timed out")

    try:
        worker_result = await asyncio.wait_for(worker_future, timeout=8)
    except asyncio.TimeoutError:
        worker_result = _fail("Worker check timed out")

    from app.worker.broker_config import is_celery_enabled
    broker_name = os.getenv("CELERY_BROKER", "redis").lower()

    return {
        "celery_enabled": is_celery_enabled(),
        "broker_type": broker_name if is_celery_enabled() else "none",
        "broker": broker_result,
        "worker": worker_result,
    }
