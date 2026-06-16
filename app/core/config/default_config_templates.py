"""
Canonical JSON merge templates for pipeline PATCH and env-bootstrap paths.

Provider semantics (models, base URLs, devices) come from merged default.json
or static literals here — never from process environment.
Secrets are referenced only via secret_ref URIs on the resolved dicts.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from app.core.config.client_config_resolver import load_default_client_raw_dict


def _secret_ref_from_env(env_var: Optional[str]) -> Optional[Dict[str, str]]:
    if not env_var or not str(env_var).strip():
        return None
    return {"uri": f"env://{str(env_var).strip()}"}


def default_vectordb_dict_for_type(vt: str, prev: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a vectordb dict for *vt*, preserving collection and overlays from *prev*."""
    vt = vt.strip().lower()
    prev = prev or {}
    coll = prev.get("collection") or "ingested_content"

    defaults_root = deepcopy(load_default_client_raw_dict())
    base_vdb = defaults_root.get("vectordb") or {}
    base_type = str(base_vdb.get("type", "")).lower()

    if base_type == vt:
        vd = deepcopy(base_vdb)
        vd["collection"] = coll
        return vd

    prev_chroma = prev.get("chroma") if isinstance(prev.get("chroma"), dict) else {}
    base_chroma = base_vdb.get("chroma") if isinstance(base_vdb.get("chroma"), dict) else {}

    if vt == "chroma":
        return {
            "type": "chroma",
            "collection": coll,
            "chroma": {
                "persist_directory": (
                    prev_chroma.get("persist_directory")
                    or base_chroma.get("persist_directory")
                    or "./pluggable_db"
                ),
                "host": prev_chroma.get("host", base_chroma.get("host")),
                "port": int(prev_chroma.get("port") or base_chroma.get("port") or 8000),
                "ssl": bool(prev_chroma.get("ssl", base_chroma.get("ssl", False))),
                "secret_ref": _secret_ref_from_env(
                    (prev_chroma.get("secret_ref") or {}).get("uri")
                    if isinstance(prev_chroma.get("secret_ref"), dict)
                    else prev_chroma.get("api_key_env") or base_chroma.get("api_key_env")
                ),
                "tenant": str(prev_chroma.get("tenant") or base_chroma.get("tenant") or "default_tenant"),
                "database": str(
                    prev_chroma.get("database") or base_chroma.get("database") or "default_database"
                ),
                "anonymized_telemetry": bool(
                    prev_chroma.get("anonymized_telemetry", base_chroma.get("anonymized_telemetry", False))
                ),
            },
        }
    if vt == "qdrant":
        pq = prev.get("qdrant") if isinstance(prev.get("qdrant"), dict) else {}
        return {
            "type": "qdrant",
            "collection": coll,
            "qdrant": {
                "url": pq.get("url"),
                "secret_ref": _secret_ref_from_env(
                    (pq.get("secret_ref") or {}).get("uri")
                    if isinstance(pq.get("secret_ref"), dict)
                    else pq.get("api_key_env")
                ),
                "host": str(pq.get("host") or "localhost"),
                "port": int(pq.get("port") or 6333),
                "transport": str(pq.get("transport") or "auto"),
                "prefer_grpc": bool(pq.get("prefer_grpc", False)),
                "timeout": float(pq.get("timeout") or 30.0),
            },
        }
    if vt == "pinecone":
        pp = prev.get("pinecone") if isinstance(prev.get("pinecone"), dict) else {}
        mode = str(pp.get("mode") or "cloud").strip().lower()
        pine_mode = "local" if mode == "local" else "cloud"
        api_ref = None if pine_mode == "local" else _secret_ref_from_env(
            (pp.get("secret_ref") or {}).get("uri")
            if isinstance(pp.get("secret_ref"), dict)
            else pp.get("api_key_env")
        )
        pinecone_block: Dict[str, Any] = {
            "mode": pine_mode,
            "index_name": str(pp.get("index_name") or "ingested-content"),
            "namespace": str(pp.get("namespace") or "default"),
            "embedding_dim": int(pp.get("embedding_dim") or 768),
            "metric": str(pp.get("metric") or "cosine"),
            "cloud": str(pp.get("cloud") or "aws"),
            "region": str(pp.get("region") or "us-east-1"),
            "local_path": pp.get("local_path"),
        }
        if api_ref is not None:
            pinecone_block["secret_ref"] = api_ref
        return {
            "type": "pinecone",
            "collection": coll,
            "pinecone": pinecone_block,
        }
    if vt == "weaviate":
        pw = prev.get("weaviate") if isinstance(prev.get("weaviate"), dict) else {}
        return {
            "type": "weaviate",
            "collection": coll,
            "weaviate": {
                "url": str(pw.get("url") or "http://localhost:8080"),
                "secret_ref": _secret_ref_from_env(
                    (pw.get("secret_ref") or {}).get("uri")
                    if isinstance(pw.get("secret_ref"), dict)
                    else pw.get("api_key_env")
                ),
            },
        }
    if vt == "milvus":
        pm = prev.get("milvus") if isinstance(prev.get("milvus"), dict) else {}
        return {
            "type": "milvus",
            "collection": coll,
            "milvus": {
                "uri": pm.get("uri"),
                "secret_ref": _secret_ref_from_env(
                    (pm.get("secret_ref") or {}).get("uri")
                    if isinstance(pm.get("secret_ref"), dict)
                    else pm.get("token_env")
                ),
                "host": str(pm.get("host") or "localhost"),
                "port": int(pm.get("port") or 19530),
            },
        }
    if vt == "redis":
        pr = prev.get("redis") if isinstance(prev.get("redis"), dict) else {}
        return {
            "type": "redis",
            "collection": coll,
            "redis": {
                "url": pr.get("url") or "redis://localhost:6379/0",
            },
        }
    raise ValueError(f"Unsupported vectordb type '{vt}'.")


_STATIC_EMBEDDERS: Dict[str, Dict[str, Any]] = {
    "huggingface": {
        "type": "huggingface",
        "huggingface": {
            "model": "BAAI/bge-large-en-v1.5",
            "device": "auto",
            "trust_remote_code": False,
        },
        "query_prefix": "",
        "document_prefix": "",
    },
    "openai": {
        "type": "openai",
        "openai": {
            "model": "text-embedding-3-small",
            "secret_ref": {"uri": "env://OPENAI_API_KEY"},
        },
    },
    "ollama": {
        "type": "ollama",
        "ollama": {"model": "nomic-embed-text", "base_url": "http://localhost:11434"},
    },
    "cohere": {
        "type": "cohere",
        "cohere": {
            "model": "embed-english-v3.0",
            "secret_ref": {"uri": "env://COHERE_API_KEY"},
        },
    },
    "gemini": {
        "type": "gemini",
        "gemini": {
            "model": "gemini-embedding-001",
            "secret_ref": {"uri": "env://GOOGLE_API_KEY"},
        },
    },
}


def default_embedder_dict_for_type(embedder_type: str, prev: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    et = embedder_type.strip().lower()
    if et == "google":
        et = "gemini"
    prev = prev or {}

    defaults_root = deepcopy(load_default_client_raw_dict())
    base_emb = defaults_root.get("embedder")
    if isinstance(base_emb, dict) and str(base_emb.get("type", "")).lower() == et:
        emb = deepcopy(base_emb)
    else:
        static = _STATIC_EMBEDDERS.get(et)
        if static is None:
            raise ValueError(f"Unsupported embedder type '{embedder_type}'.")
        emb = deepcopy(static)

    sub = prev.get(et) if isinstance(prev.get(et), dict) else None
    if sub:
        emb[et] = {**(emb.get(et) or {}), **sub}
    for k in ("query_prefix", "document_prefix"):
        if k in prev and prev[k] is not None:
            emb[k] = prev[k]
    emb["type"] = et
    return emb


_STATIC_LLMS: Dict[str, Dict[str, Any]] = {
    "ollama": {
        "type": "ollama",
        "model": "llama3.2",
        "base_url": "http://localhost:11434",
    },
    "openai": {
        "type": "openai",
        "model": "gpt-4o-mini",
        "secret_ref": {"uri": "env://OPENAI_API_KEY"},
        "base_url": "https://api.openai.com/v1",
    },
    "gemini": {
        "type": "gemini",
        "model": "gemini-1.5-flash",
        "secret_ref": {"uri": "env://GOOGLE_API_KEY"},
        "base_url": "https://generativelanguage.googleapis.com/v1",
    },
    "google": {
        "type": "gemini",
        "model": "gemini-1.5-flash",
        "secret_ref": {"uri": "env://GOOGLE_API_KEY"},
        "base_url": "https://generativelanguage.googleapis.com/v1",
    },
    "groq": {
        "type": "groq",
        "model": "llama-3.1-8b-instant",
        "secret_ref": {"uri": "env://GROQ_API_KEY"},
        "base_url": "https://api.groq.com/openai/v1",
    },
    "anthropic": {
        "type": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
        "secret_ref": {"uri": "env://ANTHROPIC_API_KEY"},
        "base_url": "https://api.anthropic.com",
    },
    "xai": {
        "type": "xai",
        "model": "grok-2",
        "secret_ref": {"uri": "env://XAI_API_KEY"},
        "base_url": "https://api.x.ai/v1",
    },
    "deepseek": {
        "type": "deepseek",
        "model": "deepseek-chat",
        "secret_ref": {"uri": "env://DEEPSEEK_API_KEY"},
        "base_url": "https://api.deepseek.com",
    },
    "huggingface": {
        "type": "huggingface",
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "secret_ref": {"uri": "env://HF_TOKEN"},
        "base_url": "https://router.huggingface.co/v1",
    },
}


def default_llm_root_dict_for_provider(llm_provider: str, prev: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    lt = llm_provider.strip().lower()
    if lt == "google":
        lt = "gemini"
    if lt == "grok":
        lt = "xai"
    prev = prev or {}
    prev_single = prev.get("single") if isinstance(prev.get("single"), dict) else {}

    defaults_root = deepcopy(load_default_client_raw_dict())
    base_llm = defaults_root.get("llm")
    base_single = (base_llm or {}).get("single") if isinstance(base_llm, dict) else None
    if isinstance(base_single, dict) and str(base_single.get("type", "")).lower() == lt:
        single = deepcopy(base_single)
    else:
        static = _STATIC_LLMS.get(lt)
        if static is None:
            raise ValueError(f"Unsupported LLM provider '{llm_provider}'.")
        single = deepcopy(static)

    # When switching providers, do not let prev_single.type / model / secret_ref / base_url
    # from the old provider overwrite the new static single (would leave e.g. gemini after PATCH ollama).
    prev_t = str(prev_single.get("type", "")).strip().lower()
    if prev_t == "google":
        prev_t = "gemini"
    same_provider = bool(prev_t) and prev_t == lt
    if same_provider:
        single = {**single, **{k: v for k, v in prev_single.items() if v is not None}}
    else:
        passthrough_keys = frozenset({"temperature", "max_tokens", "system_prompt", "timeout"})
        overlay = {
            k: v
            for k, v in prev_single.items()
            if k in passthrough_keys and v is not None
        }
        single = {**single, **overlay}
    single.setdefault("temperature", 0.3)
    single.setdefault("max_tokens", 1024)
    if "base_url" not in single or single["base_url"] in (None, ""):
        fb = _STATIC_LLMS.get(lt) or {}
        if fb.get("base_url"):
            single["base_url"] = fb["base_url"]
    return {"single": single}
