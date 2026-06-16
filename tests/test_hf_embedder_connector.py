"""Tests for HuggingFace embedder connector (trust_remote_code, prefixes, catalog)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.ai.catalog.catalog_loader import EmbedderCatalogEntry, load_catalog
from app.ai.registry.embedder_registry import _hf_pretrained_kwargs, _resolve_hf_tokenizer
from app.core.config.client_config_schema import HuggingFaceEmbedderConfig
from app.core.embedders.huggingface_st_v1 import HuggingFaceSTEmbedder
from app.core.embedders.prompting import EmbeddingPrompts, PromptedEmbedder


def test_huggingface_embedder_config_defaults_trust_remote_code_false():
    cfg = HuggingFaceEmbedderConfig(model="BAAI/bge-large-en-v1.5")
    assert cfg.trust_remote_code is False
    assert cfg.revision is None
    assert cfg.secret_ref is None


def test_sentence_transformer_receives_trust_remote_code():
    embedder = HuggingFaceSTEmbedder(
        model="nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
    )
    mock_st = MagicMock()
    mock_instance = MagicMock()
    mock_instance.get_sentence_embedding_dimension.return_value = 768
    mock_st.return_value = mock_instance

    with patch("sentence_transformers.SentenceTransformer", mock_st):
        embedder._load_model()

    mock_st.assert_called_once()
    _, kwargs = mock_st.call_args
    assert kwargs.get("trust_remote_code") is True
    assert kwargs.get("device") == "cpu"


def test_sentence_transformer_receives_revision_and_token():
    embedder = HuggingFaceSTEmbedder(
        model="org/model",
        trust_remote_code=True,
        revision="main",
        hf_token="hf_test_token",
    )
    mock_st = MagicMock()
    mock_instance = MagicMock()
    mock_instance.get_sentence_embedding_dimension.return_value = 384
    mock_st.return_value = mock_instance

    with patch("sentence_transformers.SentenceTransformer", mock_st):
        embedder._load_model()

    _, kwargs = mock_st.call_args
    assert kwargs.get("revision") == "main"
    assert kwargs.get("token") == "hf_test_token"
    assert kwargs.get("trust_remote_code") is True


def test_auto_tokenizer_receives_trust_remote_code_from_catalog():
    entry = EmbedderCatalogEntry(
        {
            "model_id": "nomic-ai/nomic-embed-text-v1.5",
            "provider": "huggingface",
            "tokenizer_class": "transformers.AutoTokenizer",
            "tokenizer_source": "hf_config",
            "tokenizer_family": "wordpiece",
            "dimension": 768,
            "distance_metric": "cosine",
            "is_normalized": True,
            "embed_max_tokens": 2048,
            "query_prefix": "search_query: ",
            "passage_prefix": "search_document: ",
            "requires_trust_remote_code": True,
            "verification_status": "verified",
        }
    )
    assert _hf_pretrained_kwargs(entry) == {"trust_remote_code": True}

    mock_tok = MagicMock()
    mock_tok.vocab_file = None
    mock_tok.cls_token = "[CLS]"
    mock_tok.sep_token = "[SEP]"
    mock_tok.pad_token = ""
    mock_tok.unk_token = "[UNK]"

    mock_cfg = MagicMock()
    mock_cfg.max_position_embeddings = 2048
    mock_cfg.model_type = "bert"

    with patch("transformers.AutoTokenizer") as mock_at, patch(
        "transformers.AutoConfig"
    ) as mock_ac:
        mock_at.from_pretrained.return_value = mock_tok
        mock_ac.from_pretrained.return_value = mock_cfg
        _resolve_hf_tokenizer(entry)

    mock_at.from_pretrained.assert_called_once_with(
        "nomic-ai/nomic-embed-text-v1.5",
        use_fast=True,
        trust_remote_code=True,
    )
    mock_ac.from_pretrained.assert_called_once_with(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
    )


def test_prompted_embedder_applies_prefixes_at_encode_time():
    base = MagicMock()
    base.embed_query.return_value = [0.1, 0.2]
    base.embed_documents.return_value = [[0.1, 0.2]]

    prompted = PromptedEmbedder(
        base=base,
        prompts=EmbeddingPrompts(
            query_prefix="search_query: ",
            document_prefix="search_document: ",
        ),
    )

    prompted.embed_query("hello")
    base.embed_query.assert_called_once_with("search_query: hello")

    prompted.embed_documents(["chunk one"])
    base.embed_documents.assert_called_once_with(["search_document: chunk one"])


def test_nomic_catalog_entry_loads_from_yaml():
    catalog = load_catalog(force_reload=True)
    entry = catalog.entries["nomic-ai/nomic-embed-text-v1.5"]
    assert entry.dimension == 768
    assert entry.embed_max_tokens == 2048
    assert entry.requires_trust_remote_code is True
    assert entry.query_prefix == "search_query: "
    assert entry.passage_prefix == "search_document: "
    assert entry.verification_status.value == "verified"


def test_catalog_entry_requires_trust_remote_code_defaults_false():
    entry = EmbedderCatalogEntry(
        {
            "model_id": "BAAI/bge-large-en-v1.5",
            "provider": "huggingface",
            "tokenizer_class": "transformers.AutoTokenizer",
            "tokenizer_source": "hf_config",
            "tokenizer_family": "wordpiece",
            "dimension": 1024,
            "distance_metric": "cosine",
            "is_normalized": True,
            "embed_max_tokens": 512,
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "verification_status": "verified",
        }
    )
    assert entry.requires_trust_remote_code is False
