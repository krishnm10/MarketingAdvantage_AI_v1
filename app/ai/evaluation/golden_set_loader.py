"""
Load and validate golden-set JSON files for RAG evaluation.

Golden files live under tests/golden_sets/ (not shipped as runtime config).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.ai.evaluation.rag_evaluator import GoldenExample

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
DEFAULT_GOLDEN_ROOT = Path(__file__).resolve().parents[3] / "tests" / "golden_sets"


@dataclass(frozen=True)
class GoldenCaseSpec:
    """One labeled case from JSON (superset of GoldenExample fields)."""

    question_id: str
    question: str
    relevant_chunk_ids: Set[str]
    relevant_file_ids: Set[str]
    expected_answer: Optional[str]
    expected_answer_substrings: List[str]
    forbidden_substrings: List[str]
    route_hint: str
    metadata: Dict[str, Any]

    def skip_retrieval_metrics_if_empty(self) -> bool:
        return bool(self.metadata.get("skip_retrieval_metrics_if_empty")) and not self.relevant_chunk_ids


@dataclass(frozen=True)
class GoldenSetFile:
    """Parsed golden-set file."""

    schema_version: str
    set_id: str
    domain: str
    tenant_id: str
    description: str
    corpus_fingerprint: str
    k_values: List[int]
    cases: List[GoldenCaseSpec]
    source_path: Path

    def to_golden_examples(self) -> List[GoldenExample]:
        out: List[GoldenExample] = []
        for case in self.cases:
            if case.skip_retrieval_metrics_if_empty():
                continue
            out.append(
                GoldenExample(
                    question_id=case.question_id,
                    question=case.question,
                    relevant_chunk_ids=set(case.relevant_chunk_ids),
                    expected_answer=case.expected_answer,
                    metadata={
                        **case.metadata,
                        "route_hint": case.route_hint,
                        "relevant_file_ids": sorted(case.relevant_file_ids),
                    },
                )
            )
        return out

    def cases_for_retrieval_eval(self) -> List[GoldenCaseSpec]:
        return [c for c in self.cases if not c.skip_retrieval_metrics_if_empty()]


def resolve_golden_set_path(set_ref: str, *, root: Optional[Path] = None) -> Path:
    """
    Resolve a set reference to an absolute path.

    Accepts:
      - invoice/vaidyanad_inv_1101.json
      - vaidyanad_inv_1101 (searches under root)
      - absolute path
    """
    base = (root or DEFAULT_GOLDEN_ROOT).resolve()
    ref = set_ref.strip().replace("\\", "/")
    candidate = Path(ref)
    if candidate.is_file():
        return candidate.resolve()

    if not ref.endswith(".json"):
        ref = f"{ref}.json"
    direct = base / ref
    if direct.is_file():
        return direct.resolve()

    matches = list(base.rglob(Path(ref).name))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Ambiguous golden set '{set_ref}': {matches}. Use a full relative path."
        )
    raise FileNotFoundError(f"Golden set not found: {set_ref} (searched under {base})")


def _parse_case(raw: Dict[str, Any]) -> GoldenCaseSpec:
    if "question_id" not in raw or "question" not in raw:
        raise ValueError("Each case requires question_id and question.")

    route_hint = str(raw.get("route_hint") or "any").strip().lower()
    allowed_routes = {"structured", "knowledge", "chitchat", "meta_help", "any"}
    if route_hint not in allowed_routes:
        raise ValueError(
            f"Invalid route_hint={route_hint!r} for question_id={raw.get('question_id')}"
        )

    return GoldenCaseSpec(
        question_id=str(raw["question_id"]),
        question=str(raw["question"]),
        relevant_chunk_ids={str(x) for x in raw.get("relevant_chunk_ids") or [] if str(x).strip()},
        relevant_file_ids={str(x) for x in raw.get("relevant_file_ids") or [] if str(x).strip()},
        expected_answer=raw.get("expected_answer"),
        expected_answer_substrings=list(raw.get("expected_answer_substrings") or []),
        forbidden_substrings=list(raw.get("forbidden_substrings") or []),
        route_hint=route_hint,
        metadata=dict(raw.get("metadata") or {}),
    )


def load_golden_set(
    set_ref: str,
    *,
    root: Optional[Path] = None,
) -> GoldenSetFile:
    """Load and minimally validate a golden-set JSON file."""
    path = resolve_golden_set_path(set_ref, root=root)
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Golden set must be a JSON object: {path}")

    version = str(data.get("schema_version", ""))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={version!r} in {path} (expected {SCHEMA_VERSION})"
        )

    cases_raw = data.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError(f"Golden set must include a non-empty cases array: {path}")

    k_values = data.get("k_values") or [1, 3, 5, 10]
    k_values = [int(k) for k in k_values if int(k) >= 1]

    cases = [_parse_case(c) for c in cases_raw if isinstance(c, dict)]

    return GoldenSetFile(
        schema_version=version,
        set_id=str(data["set_id"]),
        domain=str(data.get("domain") or "general"),
        tenant_id=str(data["tenant_id"]),
        description=str(data.get("description") or ""),
        corpus_fingerprint=str(data.get("corpus_fingerprint") or ""),
        k_values=k_values,
        cases=cases,
        source_path=path,
    )


def load_golden_set_from_path(path: Path) -> GoldenSetFile:
    return load_golden_set(str(path.resolve()), root=path.parent)
