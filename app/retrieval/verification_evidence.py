from __future__ import annotations

"""
Phase 6A — Verification evidence bundle builder.

Builds immutable evidence from the final post-sanitization LLM-visible context_str.
Raw top_chunks.text must never be used as the textual basis for verification.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

_SOURCE_BLOCK_SEP = "\n\n---\n\n"
_SOURCE_HEADER_RE = re.compile(
    r"^\[Source\s+(\d+)\]\s*\(score=[\d.]+\)\n(.*)\Z",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class Passage:
    """One LLM-visible source passage derived from sanitized context_str."""

    index: int  # 1-based [Source n]
    chunk_id: str
    file_id: Optional[str]
    text: str  # passage body only (header excluded)


@dataclass(frozen=True)
class VerificationEvidenceBundle:
    """Authoritative verifier input for KNOWLEDGE/docset routes."""

    route: str
    raw_query: str
    passages: Tuple[Passage, ...]
    alignment_valid: bool
    alignment_errors: Tuple[str, ...]


def build_verification_evidence_bundle(
    *,
    context_str: str,
    chunk_metas: Sequence[Tuple[str, Optional[str]]],
    route: str,
    raw_query: str,
) -> VerificationEvidenceBundle:
    """
    Parse [Source n] blocks from final post-sanitization context_str.

    chunk_metas[i] = (chunk_id, file_id) for source index i+1.
    Text is always taken from context_str segments, never from raw chunk text.
    """
    errors: List[str] = []
    expected_count = len(chunk_metas)
    context = context_str or ""

    if expected_count == 0:
        return VerificationEvidenceBundle(
            route=route,
            raw_query=raw_query,
            passages=(),
            alignment_valid=not context.strip(),
            alignment_errors=("no_chunks_expected",) if context.strip() else (),
        )

    if not context.strip():
        return VerificationEvidenceBundle(
            route=route,
            raw_query=raw_query,
            passages=(),
            alignment_valid=False,
            alignment_errors=("empty_context_str",),
        )

    blocks = context.split(_SOURCE_BLOCK_SEP)
    if len(blocks) != expected_count:
        errors.append(
            f"source_block_count_mismatch:expected={expected_count}:got={len(blocks)}"
        )

    passages: List[Passage] = []
    for i, block in enumerate(blocks):
        source_num = i + 1
        m = _SOURCE_HEADER_RE.match(block.strip())
        if not m:
            errors.append(f"source_header_parse_failed:source={source_num}")
            continue
        parsed_index = int(m.group(1))
        body = m.group(2)
        if parsed_index != source_num:
            errors.append(
                f"source_index_mismatch:expected={source_num}:got={parsed_index}"
            )
        if i >= len(chunk_metas):
            errors.append(f"chunk_meta_missing:source={source_num}")
            continue
        chunk_id, file_id = chunk_metas[i]
        passages.append(
            Passage(
                index=source_num,
                chunk_id=chunk_id,
                file_id=file_id,
                text=body,
            )
        )

    if len(passages) != expected_count:
        errors.append(
            f"passage_count_mismatch:expected={expected_count}:got={len(passages)}"
        )

    alignment_valid = not errors and len(passages) == expected_count
    return VerificationEvidenceBundle(
        route=route,
        raw_query=raw_query,
        passages=tuple(passages),
        alignment_valid=alignment_valid,
        alignment_errors=tuple(errors),
    )


def passage_by_index(
    bundle: VerificationEvidenceBundle,
    index: int,
) -> Optional[Passage]:
    """Return passage for 1-based [Source n] index, or None."""
    for p in bundle.passages:
        if p.index == index:
            return p
    return None
