"""
Lightweight BM25 keyword search over a list of text chunks.

Used by hybrid search (SearchMode.HYBRID) to combine keyword matches
with vector similarity via RRF fusion.

No external dependencies — uses a pure-Python BM25 Okapi implementation
so it works everywhere and doesn't add install requirements.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class BM25Hit:
    """One BM25 search result."""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


# ── Simple tokeniser (lowered, split on non-alphanum) ─────────────────────
_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t for t in _SPLIT_RE.split(text.lower()) if t]


class BM25Index:
    """
    In-memory BM25 Okapi index.

    Designed for small-to-medium collections (up to ~100k chunks).
    For larger corpora, consider Elasticsearch or a dedicated BM25 backend.

    Typical usage inside RAGPipeline::

        idx = BM25Index()
        idx.build(chunks)              # chunks = [{id, text, metadata}, ...]
        hits = idx.search("query", k=20)
    """

    # BM25 Okapi parameters
    k1: float = 1.5
    b: float = 0.75

    def __init__(self, *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[Dict[str, Any]] = []
        self._doc_tokens: List[List[str]] = []
        self._avg_dl: float = 0.0
        self._df: Dict[str, int] = {}   # document frequency per term
        self._N: int = 0

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build the index from a list of chunk dicts.
        Each chunk must have at least 'id' and 'text' keys.
        """
        self._docs = chunks
        self._doc_tokens = []
        self._df = {}
        self._N = len(chunks)
        total_len = 0

        for chunk in chunks:
            tokens = _tokenize(chunk.get("text", ""))
            self._doc_tokens.append(tokens)
            total_len += len(tokens)
            seen: set = set()
            for t in tokens:
                if t not in seen:
                    self._df[t] = self._df.get(t, 0) + 1
                    seen.add(t)

        self._avg_dl = total_len / self._N if self._N else 1.0

    def search(self, query: str, k: int = 20) -> List[BM25Hit]:
        """Return top-k BM25 hits for the given query string."""
        query_tokens = _tokenize(query)
        if not query_tokens or not self._N:
            return []

        scores: List[float] = [0.0] * self._N

        for qt in query_tokens:
            df = self._df.get(qt, 0)
            if df == 0:
                continue
            idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)
            for i, doc_tokens in enumerate(self._doc_tokens):
                tf = doc_tokens.count(qt)
                if tf == 0:
                    continue
                dl = len(doc_tokens)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                scores[i] += idf * numerator / denominator

        # Top-k by score descending
        ranked = sorted(range(self._N), key=lambda i: scores[i], reverse=True)[:k]
        results: List[BM25Hit] = []
        for i in ranked:
            if scores[i] <= 0:
                break
            doc = self._docs[i]
            results.append(BM25Hit(
                id=doc.get("id", ""),
                text=doc.get("text", ""),
                score=scores[i],
                metadata=doc.get("metadata", {}),
            ))
        return results
