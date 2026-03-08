from app.core.vectordb.milvus_v1 import (
    _build_milvus_filter_expr,
    _normalize_milvus_metadata,
    _sanitize_milvus_metadata,
)


def test_build_milvus_filter_expr_supports_operator_filters():
    expr = _build_milvus_filter_expr(
        {
            "semantic_hash": {"$ne": "abc123"},
            "tenant": {"$in": ["acme", "globex"]},
            "score": {"$gte": 0.8},
            "active": True,
        }
    )

    assert expr == (
        '_metadata["semantic_hash"] != "abc123" && '
        '_metadata["tenant"] in ["acme", "globex"] && '
        '_metadata["score"] >= 0.8 && '
        '_metadata["active"] == true'
    )


def test_normalize_milvus_metadata_supports_legacy_json_strings():
    meta = _normalize_milvus_metadata('{"semantic_hash":"abc123","rank":2}')

    assert meta == {"semantic_hash": "abc123", "rank": 2}


def test_sanitize_milvus_metadata_stringifies_non_json_values():
    class ValueObject:
        def __str__(self) -> str:
            return "value-object"

    meta = _sanitize_milvus_metadata({"obj": ValueObject(), "empty": None, "flag": True})

    assert meta == {"obj": "value-object", "empty": "", "flag": True}
