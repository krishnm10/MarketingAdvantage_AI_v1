"""
Parity checks: merged ClientConfig drives both ingestion and query pipelines.

Run: pytest tests/tenant_isolation/test_pipeline_parity_fingerprint.py -v
"""

from __future__ import annotations

from app.core.config.client_config_resolver import get_client_config, get_config_fingerprint
from app.services.ingestion import ingestion_service_v2 as isv2


def test_config_fingerprint_for_default_tenant():
    cfg = get_client_config("default")
    fp = get_config_fingerprint(cfg)
    assert isinstance(fp, str) and len(fp) >= 16


def test_ingest_and_query_pipeline_share_vectordb_and_embedder_kind(monkeypatch):
    """Avoid instantiating real embedders (optional deps); assert factory parity contract."""

    class _VD:
        kind = "chroma"

    class _Emb:
        pass

    class _Pipe:
        vectordb = _VD()
        embedder = _Emb()

    def _fake_build(_cfg, skip_cache: bool = False):
        return _Pipe()

    monkeypatch.setattr(isv2.pipeline_factory, "build", _fake_build)
    isv2.clear_ingestion_pipeline_cache()

    cfg = get_client_config("default")
    ing = isv2._get_ingestion_pipeline_for_client(cfg.client_id)
    qry = isv2.get_query_pipeline_for_client(cfg.client_id)

    assert ing.vectordb.kind == qry.vectordb.kind
    assert type(ing.embedder) is type(qry.embedder)


def test_get_query_pipeline_for_client_is_public_alias():
    assert isv2.get_query_pipeline_for_client is isv2._get_pipeline_for_client
