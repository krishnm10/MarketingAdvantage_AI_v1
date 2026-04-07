# =============================================
# segmenter_elite_v2.py — Elite Enterprise-Advanced Semantic Chunker v2
#
# Upgrades over v1:
#   1. PROPOSITION-BASED CHUNKING (opt-in)
#      - LLM decomposes sentences into atomic independent facts
#      - Resolves dangling pronouns (it/they → explicit entities)
#      - Env: CHUNK_ELITE_PROPOSITIONS=true
#
#   2. SEMANTIC DYNAMIC OVERLAP (replaces fixed overlap)
#      - Uses boundary coherence scoring to decide IF overlap is needed
#      - Dynamically calculates overlap size based on coherence gap
#      - High-coherence boundaries get zero overlap (same topic, no waste)
#
#   3. AGENTIC SEMANTIC SPLITTING (opt-in)
#      - LLM identifies logical topic transition points
#      - Understands meaning, not just word overlap
#      - Env: CHUNK_ELITE_AGENTIC_SPLIT=true
#
#   4. RECURSIVE CONTEXTUAL ENRICHMENT (Global Gist)
#      - Every chunk carries a 1-sentence document summary
#      - LLM gist (default) or extractive fallback
#      - Chunks become globally self-describing in vector space
#
#   5. TF-IDF TOPIC SHIFT REFINEMENT (always on)
#      - Merges TF-IDF cosine + discourse cue detection with Jaccard
#      - Catches topic shifts that pure word-overlap misses
#
# 10-PHASE PIPELINE:
#   Phase 0: Text pre-processing
#   Phase 1: Advanced sentence tokenization (abbreviation-safe)
#   Phase 2A: Topic boundary detection (Jaccard default / Agentic opt-in)
#   Phase 2B: TF-IDF topic shift refinement (always on, merges with 2A)
#   Phase 3: Semantic segment assembly
#   Phase 4: Proposition decomposition (opt-in LLM)
#   Phase 5: Hierarchical multi-resolution chunking (parent + child)
#   Phase 6: Semantic dynamic overlap (coherence-gated)
#   Phase 7: Global gist injection (LLM default / extractive fallback)
#   Phase 8: Section context injection
#   Phase 9: Quality gate
#
# Registered strategies: "elite_v2", "enterprise_v3"
#
# Env vars (new — in addition to all CHUNK_ELITE_* from v1):
#   CHUNK_ELITE_PROPOSITIONS          — enable proposition decomposition (default false)
#   CHUNK_ELITE_AGENTIC_SPLIT         — enable LLM topic splitting (default false)
#   CHUNK_ELITE_LLM_PROVIDER          — LLM provider override (default → MAI_LLM → "ollama")
#   CHUNK_ELITE_GIST_MODE             — "llm" (default) or "extractive"
#   CHUNK_ELITE_COHERENCE_THRESHOLD   — dynamic overlap coherence gate (default 0.6)
# =============================================

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Dict, List, Optional

from app.core.chunking_stratagies.chunking_registry import Chunker, register_chunker

# ── Reuse from Elite v1 (no duplication) ────────────────────────────────────
from app.core.chunking_stratagies.segmenter_elite_v1 import (
    _elite_tokenize_sentences,
    _estimate_tokens,
    _detect_topic_boundaries,
    _assemble_segments,
    _build_hierarchical_chunks,
    _extract_section_title,
    _inject_section_context,
    _apply_quality_gate,
    PARENT_MAX_TOKENS,
    PARENT_MIN_TOKENS,
    CHILD_MAX_TOKENS,
    CHILD_MIN_TOKENS,
    COHESION_WINDOW,
    COHESION_THRESHOLD,
    EMIT_PARENTS,
    QUALITY_GATE_THRESHOLD,
    OVERLAP_SENTENCES,
    _env_int,
    _env_float,
    _apply_sentence_overlap,
)
from app.core.chunking_stratagies.segmenter_v2 import make_chunk_dict
from app.core.chunking_stratagies.text_preprocessor import preprocess_document_text
from app.utils.text_cleaner_v2 import clean_text
from app.utils.logger import log_info, log_warning

# ── Reuse from Structure-Aware v3 (directly importable module-level functions)
from app.core.chunking_stratagies.segmenter_structure_aware import (
    _score_boundary_coherence,
    _detect_topic_shifts,
)

# ── LLM provider system ────────────────────────────────────────────────────
from app.core.plugin_registry import llm_registry


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — v2-specific env vars
# ─────────────────────────────────────────────────────────────────────────────

ENABLE_PROPOSITIONS: bool = os.getenv("CHUNK_ELITE_PROPOSITIONS", "false").lower() == "true"
ENABLE_AGENTIC_SPLIT: bool = os.getenv("CHUNK_ELITE_AGENTIC_SPLIT", "false").lower() == "true"

GIST_MODE: str = os.getenv("CHUNK_ELITE_GIST_MODE", "llm").lower()  # "llm" or "extractive"
COHERENCE_OVERLAP_THRESHOLD: float = _env_float("CHUNK_ELITE_COHERENCE_THRESHOLD", 0.6, 0.1)

# LLM provider resolution chain: CHUNK_ELITE_LLM_PROVIDER → MAI_LLM → "ollama"
_LLM_PROVIDER: str = os.getenv(
    "CHUNK_ELITE_LLM_PROVIDER",
    os.getenv("MAI_LLM", "ollama"),
).lower()


# ─────────────────────────────────────────────────────────────────────────────
# LLM HELPER — configurable provider, fail-safe
# ─────────────────────────────────────────────────────────────────────────────

def _get_elite_llm():
    """
    Build an LLM instance for elite v2 features.
    Returns None if the provider/model is unavailable.
    """
    model_env = f"{_LLM_PROVIDER.upper()}_LLM_MODEL"
    api_key_env = f"{_LLM_PROVIDER.upper()}_API_KEY"

    model = os.getenv(model_env)
    api_key = os.getenv(api_key_env, "")

    if not model:
        return None

    try:
        import app.core.llms.register  # noqa: F401 — triggers registration
        kwargs = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        return llm_registry.build(_LLM_PROVIDER, **kwargs)
    except Exception as exc:
        log_warning(f"[EliteV2] Failed to build LLM provider '{_LLM_PROVIDER}': {exc}")
        return None


async def _llm_generate(llm, prompt: str, system_prompt: str = "", max_tokens: int = 512) -> str:
    """Fail-safe LLM generation. Returns empty string on any failure.
    Runs the sync BaseLLM.generate() in a thread executor to stay event-loop safe."""
    if llm is None:
        return ""
    try:
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=max_tokens,
            ),
        )
        return (resp.text or "").strip()
    except Exception as exc:
        log_warning(f"[EliteV2] LLM generation failed: {exc}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2A-ALT: AGENTIC SEMANTIC SPLITTING (opt-in)
#
# Uses an LLM to identify logical topic transition points.
# Understands meaning, not just word overlap — catches transitions like
# "Cattle population" → "Milk production" that share the word "Dairy"
# but are different topics.
# ─────────────────────────────────────────────────────────────────────────────

_AGENTIC_SPLIT_PROMPT = """You are a document structure analyst. Below are numbered sentences from a document.
Identify the sentence numbers where a NEW TOPIC begins (not the first sentence).
Return ONLY a comma-separated list of sentence numbers. Example: 4, 9, 15
If no clear topic changes exist, return: none

Sentences:
{numbered_sentences}

Topic transition indices:"""


async def _agentic_topic_split(
    sentences: List[str],
    llm,
) -> List[int]:
    """
    LLM-based topic boundary detection. Returns sentence indices where
    new topics begin. Fail-safe: returns empty list on any error.
    """
    if not sentences or len(sentences) < 4:
        return []

    # Cap at 80 sentences to avoid exceeding context window
    capped = sentences[:80]
    numbered = "\n".join(f"{i}: {s}" for i, s in enumerate(capped))
    prompt = _AGENTIC_SPLIT_PROMPT.format(numbered_sentences=numbered)

    result = await _llm_generate(llm, prompt, max_tokens=256)
    if not result or result.lower().strip() == "none":
        return []

    # Parse comma-separated integers
    boundaries: List[int] = []
    for token in result.replace(",", " ").split():
        token = token.strip().rstrip(".")
        if token.isdigit():
            idx = int(token)
            if 1 <= idx < len(sentences):
                boundaries.append(idx)

    return sorted(set(boundaries))


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2B: TF-IDF BOUNDARY MERGE
#
# Merges Jaccard/Agentic boundaries with TF-IDF topic shifts from
# structure_aware. Union of both, deduplicated, min-gap enforced.
# ─────────────────────────────────────────────────────────────────────────────

def _merge_boundaries(
    boundaries_a: List[int],
    boundaries_b: List[int],
    n_sentences: int,
    min_gap: int = 3,
) -> List[int]:
    """
    Merge two boundary lists. Deduplicate, sort, enforce minimum gap.
    """
    merged = sorted(set(boundaries_a + boundaries_b))

    # Filter: valid range only
    merged = [b for b in merged if 1 <= b < n_sentences]

    # Enforce minimum gap
    filtered: List[int] = []
    for b in merged:
        if not filtered or b - filtered[-1] >= min_gap:
            filtered.append(b)

    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: PROPOSITION DECOMPOSITION (opt-in)
#
# Breaks sentences into atomic independent propositions using an LLM.
# Every proposition is a standalone fact with no dangling pronouns.
#
# Example:
#   Input:  "Crisil, a company of S&P Global, released its dairy dashboard in Sep-2025"
#   Output: ["Crisil is a company of S&P Global.",
#            "Crisil released a dairy dashboard in September 2025."]
# ─────────────────────────────────────────────────────────────────────────────

_PROPOSITION_PROMPT = """Break each sentence below into atomic independent facts (propositions).
Rules:
- Each proposition must be a complete, standalone sentence
- Resolve ALL pronouns (it, they, he, she, this, that) to their explicit referent
- Each proposition should contain exactly one fact
- Preserve all proper nouns, numbers, and dates exactly
- Number each proposition

Sentences:
{sentences}

Propositions:"""

_PROPOSITION_BATCH_SIZE: int = 8


async def _decompose_propositions(
    sentences: List[str],
    llm,
) -> List[str]:
    """
    LLM-based proposition decomposition. Processes in batches.
    Fail-safe: returns original sentences on any error.
    """
    if not sentences or llm is None:
        return sentences

    all_propositions: List[str] = []

    for batch_start in range(0, len(sentences), _PROPOSITION_BATCH_SIZE):
        batch = sentences[batch_start:batch_start + _PROPOSITION_BATCH_SIZE]
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(batch))
        prompt = _PROPOSITION_PROMPT.format(sentences=numbered)

        result = await _llm_generate(llm, prompt, max_tokens=1024)

        if not result:
            # Fallback: keep original sentences for this batch
            all_propositions.extend(batch)
            continue

        # Parse numbered propositions from LLM output
        batch_props: List[str] = []
        for line in result.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip leading number/bullet: "1. ", "- ", "1) "
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
            cleaned = re.sub(r"^[-*]\s*", "", cleaned)
            cleaned = cleaned.strip()
            if cleaned and len(cleaned) > 5:
                batch_props.append(cleaned)

        if batch_props:
            all_propositions.extend(batch_props)
        else:
            # Parsing failed — keep originals
            all_propositions.extend(batch)

    return all_propositions if all_propositions else sentences


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6: SEMANTIC DYNAMIC OVERLAP
#
# Replaces fixed sentence overlap with coherence-gated overlap.
# Uses _score_boundary_coherence from structure_aware to decide:
#   - High coherence (>= threshold) → skip overlap (same topic)
#   - Low coherence (< threshold) → dynamic overlap size based on gap
# ─────────────────────────────────────────────────────────────────────────────

_MAX_DYNAMIC_OVERLAP_SENTS: int = 3


def _apply_semantic_dynamic_overlap(
    chunks: List[Dict[str, Any]],
    coherence_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Coherence-gated dynamic overlap between consecutive child chunks.

    For each boundary:
    - Score coherence between end of chunk[i-1] and start of chunk[i]
    - If coherence >= threshold: no overlap needed (same topic flows naturally)
    - If coherence < threshold: overlap size proportional to coherence gap

    Preserves semantic_hash for dedup identity.
    """
    if len(chunks) < 2:
        return chunks

    for i in range(1, len(chunks)):
        prev_text = chunks[i - 1].get("text", "")
        curr_text = chunks[i].get("text", "")

        if not prev_text or not curr_text:
            continue

        # Score the boundary
        coherence = _score_boundary_coherence(prev_text, curr_text)

        meta = chunks[i].setdefault("reasoning_ingestion", {})
        meta["boundary_coherence"] = round(coherence, 4)

        if coherence >= coherence_threshold:
            # High coherence — same topic, no overlap needed
            meta["overlap_mode"] = "skipped_high_coherence"
            continue

        # Low coherence — calculate dynamic overlap
        gap = coherence_threshold - coherence
        overlap_sents = max(1, min(
            _MAX_DYNAMIC_OVERLAP_SENTS,
            round(gap / coherence_threshold * 4),
        ))

        # Extract trailing sentences from previous chunk
        prev_sents = _elite_tokenize_sentences(prev_text)
        if len(prev_sents) <= overlap_sents:
            continue  # Previous chunk too small

        overlap_text = " ".join(prev_sents[-overlap_sents:])
        overlap_tokens = _estimate_tokens(overlap_text)

        # Prepend overlap
        original_text = chunks[i].get("text", "")
        original_hash = chunks[i].get("semantic_hash")

        chunks[i]["text"] = overlap_text + " " + original_text
        chunks[i]["cleaned_text"] = clean_text(chunks[i]["text"])
        chunks[i]["tokens"] = _estimate_tokens(chunks[i]["cleaned_text"])

        # Preserve dedup identity
        chunks[i]["semantic_hash"] = original_hash
        chunks[i]["normalized_hash"] = original_hash

        meta["overlap_mode"] = "dynamic"
        meta["overlap_sentences"] = overlap_sents
        meta["overlap_tokens"] = overlap_tokens

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7: GLOBAL GIST INJECTION
#
# Every embeddable chunk carries the document's DNA — a 1-sentence summary.
# This makes chunks globally self-describing in vector space.
#
# Example: Instead of "Revenue grew by 6%", chunk becomes:
#   "Document Gist: Sep-2025 Crisil Dairy Dashboard on global milk production.
#    section: milk trends. Revenue grew by 6%."
# ─────────────────────────────────────────────────────────────────────────────

_GIST_PROMPT = """Summarize the following document in exactly ONE sentence (max 30 words).
Capture: what the document is about, the time period (if any), and the main subject.
Do NOT use phrases like "This document discusses" — start directly with the subject.

Document (first 2000 chars):
{text_preview}

One-sentence summary:"""


async def _generate_global_gist(
    full_text: str,
    sentences: List[str],
    llm,
    mode: str,
) -> str:
    """
    Generate a 1-sentence document summary.

    LLM mode: single LLM call on first 2000 chars.
    Extractive mode: first meaningful sentence (>15 words, starts uppercase).
    LLM mode falls back to extractive on failure.
    """
    # Try LLM mode first
    if mode == "llm" and llm is not None:
        preview = full_text[:2000]
        prompt = _GIST_PROMPT.format(text_preview=preview)
        gist = await _llm_generate(llm, prompt, max_tokens=64)
        if gist and len(gist.split()) >= 5:
            # Clean trailing period if missing
            if gist[-1] not in ".!?":
                gist += "."
            return gist

    # Extractive fallback
    for sent in sentences[:20]:
        words = sent.split()
        if len(words) >= 15 and sent[0].isupper():
            # Truncate if too long
            if len(words) > 30:
                return " ".join(words[:30]) + "."
            return sent

    # Last resort: first sentence
    if sentences:
        return sentences[0]

    return ""


def _inject_global_gist(
    chunks: List[Dict[str, Any]],
    global_gist: str,
) -> List[Dict[str, Any]]:
    """
    Prepend the document gist to every embeddable chunk's cleaned_text.
    Makes each chunk globally unique and self-describing in vector space.
    """
    if not global_gist:
        return chunks

    prefix = f"Document Gist: {global_gist} "

    for chunk in chunks:
        meta = chunk.get("reasoning_ingestion", {})
        if not meta.get("embeddable", True):
            continue

        meta["document_gist"] = global_gist

        if "cleaned_text" in chunk:
            chunk["cleaned_text"] = prefix + chunk["cleaned_text"]
            chunk["tokens"] = _estimate_tokens(chunk["cleaned_text"])

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# MAIN STRATEGY: EliteChunkerV2
# ─────────────────────────────────────────────────────────────────────────────

@register_chunker("elite_v2")
@register_chunker("enterprise_v3")
class EliteChunkerV2(Chunker):
    """
    Elite Enterprise-Advanced Semantic Chunker v2.

    10-phase pipeline producing "Intelligence Atom" chunks:
    - Every chunk carries the document's global gist
    - Boundaries are coherence-gated (smart overlap, not fixed)
    - Optional LLM proposition decomposition for atomic facts
    - Optional LLM agentic topic splitting for meaning-based boundaries
    - TF-IDF cosine + discourse cue refinement always on

    Usage:
        chunker = get_chunker("elite_v2")
        chunks = await chunker.chunk(text, file_id=..., source_type=...)

    Or via env: CHUNKING_STRATEGY=elite_v2
    """

    async def chunk(
        self,
        text: str,
        *,
        db_session=None,
        file_id=None,
        business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not (text or "").strip():
            return []

        kw = dict(
            file_id=file_id, business_id=business_id,
            source_type=source_type, embedding_model=embedding_model,
        )

        # Build LLM once (shared across all LLM-dependent phases)
        llm = None
        needs_llm = ENABLE_PROPOSITIONS or ENABLE_AGENTIC_SPLIT or GIST_MODE == "llm"
        if needs_llm:
            llm = _get_elite_llm()

        # ── Phase 0: Pre-processing ──────────────────────────────────
        preprocessed = preprocess_document_text(text)
        if not preprocessed.strip():
            return []

        section_title = _extract_section_title(preprocessed)

        # ── Phase 1: Advanced sentence tokenization ──────────────────
        sentences = _elite_tokenize_sentences(preprocessed)
        if not sentences:
            return []

        # Fast path: very short text → single standalone chunk
        total_tokens = sum(_estimate_tokens(s) for s in sentences)
        if total_tokens <= CHILD_MAX_TOKENS:
            chunk = make_chunk_dict(" ".join(sentences), **kw)
            if not chunk:
                return []
            meta = chunk.setdefault("reasoning_ingestion", {})
            meta["chunking_strategy"] = "elite_v2"
            meta["chunk_resolution"] = "standalone"
            meta["embeddable"] = True
            if section_title:
                meta["section_context"] = section_title
            return [chunk]

        # ── Phase 2A: Topic boundary detection ───────────────────────
        if ENABLE_AGENTIC_SPLIT and llm is not None:
            boundaries_primary = await _agentic_topic_split(sentences, llm)
            boundary_method = "agentic"
            # Fallback to Jaccard if LLM returned nothing useful
            if not boundaries_primary:
                effective_window = min(COHESION_WINDOW, max(2, len(sentences) // 6))
                boundaries_primary = _detect_topic_boundaries(
                    sentences, window=effective_window, threshold=COHESION_THRESHOLD,
                )
                boundary_method = "jaccard_fallback"
        else:
            effective_window = min(COHESION_WINDOW, max(2, len(sentences) // 6))
            boundaries_primary = _detect_topic_boundaries(
                sentences, window=effective_window, threshold=COHESION_THRESHOLD,
            )
            boundary_method = "jaccard"

        # ── Phase 2B: TF-IDF topic shift refinement (always on) ─────
        tfidf_boundaries = _detect_topic_shifts(sentences)
        boundaries = _merge_boundaries(
            boundaries_primary, tfidf_boundaries, len(sentences),
        )

        # ── Phase 3: Semantic segment assembly ───────────────────────
        segments = _assemble_segments(sentences, boundaries)

        # ── Phase 4: Proposition decomposition (opt-in) ─────────────
        proposition_active = False
        if ENABLE_PROPOSITIONS and llm is not None:
            # Decompose each segment's sentences into propositions
            decomposed_segments: List[List[str]] = []
            for seg in segments:
                props = await _decompose_propositions(seg, llm)
                decomposed_segments.append(props)
            segments = decomposed_segments
            proposition_active = True

        # ── Phase 5: Hierarchical multi-resolution chunking ──────────
        parent_chunks, child_chunks = _build_hierarchical_chunks(
            segments, **kw,
        )

        if not child_chunks and not parent_chunks:
            return []

        # Tag all chunks with v2 strategy name
        for c in child_chunks + parent_chunks:
            meta = c.setdefault("reasoning_ingestion", {})
            meta["chunking_strategy"] = "elite_v2"

        # ── Phase 6: Semantic dynamic overlap ────────────────────────
        if len(child_chunks) > 1:
            child_chunks = _apply_semantic_dynamic_overlap(
                child_chunks, COHERENCE_OVERLAP_THRESHOLD,
            )

        # ── Phase 7: Section context injection ─────────────────────
        #    (runs before gist so gist ends up as outermost prefix)
        child_chunks = _inject_section_context(child_chunks, section_title)
        if EMIT_PARENTS:
            parent_chunks = _inject_section_context(parent_chunks, section_title)

        # ── Phase 8: Global gist injection ─────────────────────────
        #    (prepends on top → "Document Gist: ... section: ... <text>")
        global_gist = await _generate_global_gist(
            preprocessed, sentences, llm, GIST_MODE,
        )
        if global_gist:
            child_chunks = _inject_global_gist(child_chunks, global_gist)
            if EMIT_PARENTS:
                parent_chunks = _inject_global_gist(parent_chunks, global_gist)

        # ── Phase 9: Quality gate ────────────────────────────────────
        child_chunks = _apply_quality_gate(child_chunks, QUALITY_GATE_THRESHOLD)

        # ── Assemble final output ────────────────────────────────────
        result: List[Dict[str, Any]] = list(child_chunks)
        if EMIT_PARENTS:
            result.extend(parent_chunks)

        # ── Telemetry ────────────────────────────────────────────────
        n_parents = len(parent_chunks)
        n_children = len(child_chunks)
        n_flagged = sum(
            1 for c in result
            if c.get("reasoning_ingestion", {}).get("quality_gate_action") == "flagged"
        )
        n_with_overlap = sum(
            1 for c in result
            if c.get("reasoning_ingestion", {}).get("overlap_mode") == "dynamic"
        )
        n_skipped_overlap = sum(
            1 for c in result
            if c.get("reasoning_ingestion", {}).get("overlap_mode") == "skipped_high_coherence"
        )

        log_info(
            f"[EliteV2] {len(result)} chunks "
            f"({n_parents} parents, {n_children} children, "
            f"{n_flagged} flagged) | "
            f"{len(sentences)} sentences, {len(boundaries)} boundaries "
            f"(method={boundary_method}) | "
            f"overlap: {n_with_overlap} dynamic, {n_skipped_overlap} skipped | "
            f"gist={'yes' if global_gist else 'no'} "
            f"propositions={'on' if proposition_active else 'off'} | "
            f"file_id={file_id} | source={source_type}"
        )

        return result
