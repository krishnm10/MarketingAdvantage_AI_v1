"""
Tests for normalize_for_hash — ensures the L1 dedup hash preserves
semantically meaningful characters in numeric-heavy content while
still aggressively normalising formatting variations.
"""
import pytest
from app.services.ingestion.deduplication_engine_v2 import normalize_for_hash


# ── Basic normalisation ──────────────────────────────────────────────────────
class TestBasicNormalization:
    def test_case_and_whitespace(self):
        assert normalize_for_hash("Hello   World") == "hello world"

    def test_empty_and_none(self):
        assert normalize_for_hash("") == ""
        assert normalize_for_hash(None) == ""
        assert normalize_for_hash(123) == ""

    def test_strips_plain_punctuation(self):
        result = normalize_for_hash("hello, world! foo-bar")
        assert "," not in result
        assert "!" not in result
        assert "-" not in result


# ── Numeric/financial content preservation ───────────────────────────────────
class TestNumericPreservation:
    def test_percent_preserved(self):
        result = normalize_for_hash("revenue increased 15% in Q3 2024")
        assert "15%" in result

    def test_dollar_preserved(self):
        result = normalize_for_hash("$1,000.50 profit")
        assert "$1000.50" in result

    def test_comma_in_numbers_removed(self):
        result = normalize_for_hash("revenue was 1,000,000 units")
        assert "1000000" in result

    def test_decimal_preserved(self):
        result = normalize_for_hash("growth rate 3.5%")
        assert "3.5%" in result

    def test_different_values_produce_different_hashes(self):
        a = normalize_for_hash("revenue increased 15% in Q3 2024")
        b = normalize_for_hash("revenue increased 20% in Q3 2024")
        assert a != b

    def test_formatting_variants_produce_same_hash(self):
        a = normalize_for_hash("Revenue was $1,000.00")
        b = normalize_for_hash("revenue was $1000.00")
        assert a == b

    def test_decreased_vs_increased_different(self):
        a = normalize_for_hash("revenue increased 15% Q3 2024")
        b = normalize_for_hash("revenue decreased 15% Q3 2024")
        assert a != b
