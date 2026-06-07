from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.v2.rag_config_api as rag_config_api
from app.core.config.client_config_resolver import load_default_client_raw_dict


@pytest.fixture()
def cfg_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  cfg = tmp_path / "cfg"
  cfg.mkdir()
  # Point the RAG config API at an isolated config directory for this test module.
  monkeypatch.setattr(rag_config_api, "_CONFIG_DIRS", [cfg])
  monkeypatch.setattr(rag_config_api, "_CONFIGS_DIR", cfg)
  return cfg


@pytest.fixture()
def pipeline_app(cfg_dir: Path) -> FastAPI:  # noqa: ARG001 - cfg_dir used via monkeypatch side-effects
  app = FastAPI()
  app.include_router(rag_config_api.router)
  return app


def test_pipeline_answer_min_score_round_trip(pipeline_app: FastAPI, cfg_dir: Path) -> None:
  client = TestClient(pipeline_app, raise_server_exceptions=True)
  tenant_id = "acme_score"

  # Initial GET (no config file) should surface synthetic defaults including answer_min_score.
  r1 = client.get(f"/api/v2/rag-config/pipeline/{tenant_id}")
  assert r1.status_code == 200, r1.text
  body1 = r1.json()
  assert body1["client_id"] == tenant_id
  assert body1["is_default"] is True
  assert body1["retrieval"]["answer_min_score"] == pytest.approx(0.25)

  # Flat PUT should write retrieval.answer_min_score into the tenant config.
  r2 = client.put(
    f"/api/v2/rag-config/pipeline/{tenant_id}",
    json={"answer_min_score": 0.4},
  )
  assert r2.status_code == 200, r2.text

  # Follow-up GET should now read from persisted tenant config, reflecting the new value.
  r3 = client.get(f"/api/v2/rag-config/pipeline/{tenant_id}")
  assert r3.status_code == 200, r3.text
  body3 = r3.json()
  assert body3["is_default"] is False
  assert body3["retrieval"]["answer_min_score"] == pytest.approx(0.4)

  # Underlying JSON should contain retrieval.answer_min_score with the same value.
  cfg_path = rag_config_api._get_client_config_path(tenant_id)
  assert cfg_path is not None
  stored = json.loads(cfg_path.read_text(encoding="utf-8"))
  assert stored.get("retrieval", {}).get("answer_min_score") == pytest.approx(0.4)


def test_parser_patch_deep_merge_preserves_non_ui_fields(
  pipeline_app: FastAPI,
  cfg_dir: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  client = TestClient(pipeline_app, raise_server_exceptions=True)
  tenant_id = "acme_parser"

  # Satisfy schema/env requirements for default embedder/LLM configs during validation.
  monkeypatch.setenv("GOOGLE_API_KEY", "dummy-for-contract-test")
  monkeypatch.setenv("OPENAI_API_KEY", "dummy-for-contract-test")

  # Seed a tenant overlay with explicit parsers.* config including non-UI fields.
  base = load_default_client_raw_dict()
  base["client_id"] = tenant_id
  base["parsers"] = {
    "enable_pdf": True,
    "enable_docx": True,
    "enable_audio": True,
    "enable_video": True,
    "enable_ocr": True,
    "ocr_language": "eng+hin",
    "ocr_engine": "tesseract-custom",
  }
  (cfg_dir / f"{tenant_id}.json").write_text(json.dumps(base), encoding="utf-8")

  # Patch only a single enable_* flag via PipelinePluggablePatch.parser.
  r = client.patch(
    f"/api/v2/rag-config/pipeline-pluggable/{tenant_id}",
    json={"parser": {"enable_pdf": False}},
  )
  assert r.status_code == 200, r.text

  cfg_path = rag_config_api._get_client_config_path(tenant_id)
  assert cfg_path is not None
  stored = json.loads(cfg_path.read_text(encoding="utf-8"))
  parsers_cfg = stored.get("parsers") or {}

  # UI-changed flag updated, other booleans and non-UI fields preserved.
  assert parsers_cfg.get("enable_pdf") is False
  assert parsers_cfg.get("enable_docx") is True
  assert parsers_cfg.get("enable_audio") is True
  assert parsers_cfg.get("enable_video") is True
  assert parsers_cfg.get("enable_ocr") is True
  assert parsers_cfg.get("ocr_language") == "eng+hin"
  assert parsers_cfg.get("ocr_engine") == "tesseract-custom"

  # GET pipeline-pluggable must expose the merged parser subtree back to the UI.
  r2 = client.get(f"/api/v2/rag-config/pipeline-pluggable/{tenant_id}")
  assert r2.status_code == 200, r2.text
  identity = r2.json()
  parser_identity = identity.get("parser")
  assert isinstance(parser_identity, dict)
  assert parser_identity.get("enable_pdf") is False
  assert parser_identity.get("ocr_language") == "eng+hin"

