"""Unit tests for multi-step query token aggregation."""

from types import SimpleNamespace

from app.observability.rag_chat_trace import aggregate_query_token_usage, llm_usage_payload


def _resp(prompt: int, completion: int):
    total = prompt + completion
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        finish_reason="stop",
        raw=None,
    )


def test_aggregate_sums_available_steps_only():
    payload = aggregate_query_token_usage(
        [
            ("rewrite", _resp(10, 2)),
            ("hyde", None),
            ("answer", _resp(100, 50)),
        ]
    )
    assert payload["label"] == "total_query_tokens"
    assert payload["provider_usage_available"] is True
    assert payload["prompt_tokens"] == 110
    assert payload["completion_tokens"] == 52
    assert payload["total_tokens"] == 162
    assert len(payload["steps"]) == 3
    assert payload["steps"][1]["step"] == "hyde"
    assert payload["steps"][1]["provider_usage_available"] is False


def test_aggregate_empty_steps_unavailable():
    payload = aggregate_query_token_usage([])
    assert payload["provider_usage_available"] is False
    assert payload["prompt_tokens"] == 0
    assert payload["steps"] == []


def test_llm_usage_payload_ollama_raw_fallback():
    raw = {"prompt_eval_count": 30, "eval_count": 8}
    resp = SimpleNamespace(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        finish_reason="stop",
        raw=raw,
    )
    usage = llm_usage_payload(resp)
    assert usage["provider_usage_available"] is True
    assert usage["prompt_tokens"] == 30
    assert usage["completion_tokens"] == 8
    assert usage["total_tokens"] == 38
