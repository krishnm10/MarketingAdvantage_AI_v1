"""Unit tests for Ollama token count mapping."""

from types import SimpleNamespace

from app.core.llms.ollama_v1 import _extract_ollama_token_usage


def test_extract_from_prompt_eval_count_and_eval_count():
    response = {"prompt_eval_count": 42, "eval_count": 17}
    pt, ct, tt = _extract_ollama_token_usage(response)
    assert pt == 42
    assert ct == 17
    assert tt == 59


def test_extract_from_usage_dict():
    response = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    pt, ct, tt = _extract_ollama_token_usage(response)
    assert pt == 10
    assert ct == 5
    assert tt == 15


def test_extract_from_pydantic_like_object():
    response = SimpleNamespace(prompt_eval_count=100, eval_count=25)
    pt, ct, tt = _extract_ollama_token_usage(response)
    assert pt == 100
    assert ct == 25
    assert tt == 125


def test_extract_returns_zeros_when_missing():
    pt, ct, tt = _extract_ollama_token_usage({})
    assert pt == 0
    assert ct == 0
    assert tt == 0
