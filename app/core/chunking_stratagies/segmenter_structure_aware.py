# =============================================
# segmenter_structure_aware.py — Structure-Aware Document Chunker
#
# Enterprise-grade chunker that respects document structure:
#   - Detects markdown headings, code fences, tables, list runs
#   - Never splits across structural boundaries
#   - Code blocks kept intact up to configurable max; oversized split at line boundaries
#   - Tables kept whole up to configurable max; oversized split preserving header row
#   - Lists split at item boundaries
#   - Prose sections chunked semantically via recursive_semantic_chunk()
#   - Every chunk annotated with section_title, section_depth, content_type
#
# Registered names: "structure_aware", "document_aware"
#
# Env vars:
#   CHUNK_STRUCTURE_CODE_MAX   — max chars for a single code chunk (default 2000)
#   CHUNK_STRUCTURE_TABLE_MAX  — max chars for a single table chunk (default 3000)
#   CHUNK_STRUCTURE_PROSE_MAX  — max chars for prose (default 600, same as semantic)
#   CHUNK_STRUCTURE_PROSE_MIN  — min chars for prose merge threshold (default 150)
# =============================================

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from app.core.chunking_stratagies.chunking_registry import Chunker, register_chunker
from app.core.chunking_stratagies.segmenter_v2 import (
    make_chunk_dict,
    recursive_semantic_chunk,
    DEFAULT_MAX_CHUNK_LEN,
    DEFAULT_MIN_CHUNK_LEN,
)
from app.core.chunking_stratagies.text_preprocessor import preprocess_document_text
from app.utils.logger import log_info, log_warning


# ─────────────────────────────────────────────────────────────────────────────
# ENV HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_int_env(key: str, default: int, minimum: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value >= minimum else default
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL PATTERNS — compiled once at module load
# ─────────────────────────────────────────────────────────────────────────────

# Code fence: ``` or ~~~ (with optional language tag on the opening line)
_CODE_FENCE_RE = re.compile(r"^(`{3,}|~{3,})(\w*)\s*$")

# Markdown heading: # .. ######
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

# Pipe-delimited table row  |col1|col2|  or  | col1 | col2 |
_TABLE_ROW_RE = re.compile(r"^\|.+\|$")

# List item: -, *, +, or 1. / 1)
_LIST_ITEM_RE = re.compile(r"^[\s]*([-*+]|\d+[.)])\s+")


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class _Block:
    """Lightweight container for a parsed structural block."""
    __slots__ = ("type", "text", "heading_title", "heading_depth", "language")

    def __init__(
        self,
        block_type: str,
        text: str,
        heading_title: str = "",
        heading_depth: int = 0,
        language: str = "",
    ):
        self.type = block_type          # code | table | list | heading | prose
        self.text = text
        self.heading_title = heading_title
        self.heading_depth = heading_depth
        self.language = language


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT PARSER — line-by-line structural detection
# ─────────────────────────────────────────────────────────────────────────────

def _parse_structural_blocks(text: str) -> List[_Block]:
    """
    Walk lines and emit typed blocks in document order.

    Detection priority: code fence > heading > table > list > prose.
    The parser tracks the most recent heading so every block inherits
    its nearest-ancestor section context.
    """
    lines = text.split("\n")
    blocks: List[_Block] = []

    cur_heading_title = ""
    cur_heading_depth = 0
    prose_buf: List[str] = []
    i = 0

    def _flush_prose() -> None:
        nonlocal prose_buf
        if prose_buf:
            joined = "\n".join(prose_buf).strip()
            if joined:
                blocks.append(_Block(
                    "prose", joined,
                    heading_title=cur_heading_title,
                    heading_depth=cur_heading_depth,
                ))
            prose_buf = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Code fence ────────────────────────────────────────────────
        fence_m = _CODE_FENCE_RE.match(stripped)
        if fence_m:
            _flush_prose()
            fence_char = fence_m.group(1)[0]      # ` or ~
            fence_len = len(fence_m.group(1))      # e.g. 3
            language = fence_m.group(2) or ""
            code_lines: List[str] = [line]
            i += 1
            while i < len(lines):
                code_lines.append(lines[i])
                cl = lines[i].strip()
                # Closing fence: same char, >= same length, nothing else
                if cl and all(c == fence_char for c in cl) and len(cl) >= fence_len:
                    i += 1
                    break
                i += 1
            # Strip opening/closing fence lines — content_type + code_language
            # metadata already capture that info; fences are structural syntax,
            # not meaningful content to embed.
            inner = code_lines[1:-1] if len(code_lines) >= 2 else code_lines
            code_text = "\n".join(inner).strip()
            if not code_text:
                # Edge: empty code block (```\n```)
                i += 0  # already incremented
                continue
            blocks.append(_Block(
                "code", code_text,
                heading_title=cur_heading_title,
                heading_depth=cur_heading_depth,
                language=language,
            ))
            continue

        # ── Heading ───────────────────────────────────────────────────
        heading_m = _HEADING_RE.match(stripped)
        if heading_m:
            _flush_prose()
            cur_heading_depth = len(heading_m.group(1))
            cur_heading_title = heading_m.group(2).strip()
            blocks.append(_Block(
                "heading", stripped,
                heading_title=cur_heading_title,
                heading_depth=cur_heading_depth,
            ))
            i += 1
            continue

        # ── Table ─────────────────────────────────────────────────────
        if _TABLE_ROW_RE.match(stripped):
            _flush_prose()
            table_lines: List[str] = [line]
            i += 1
            while i < len(lines):
                ns = lines[i].strip()
                if _TABLE_ROW_RE.match(ns):
                    table_lines.append(lines[i])
                    i += 1
                elif not ns:
                    # blank line terminates table
                    break
                else:
                    break
            blocks.append(_Block(
                "table", "\n".join(table_lines),
                heading_title=cur_heading_title,
                heading_depth=cur_heading_depth,
            ))
            continue

        # ── List run ──────────────────────────────────────────────────
        if _LIST_ITEM_RE.match(line):
            _flush_prose()
            list_lines: List[str] = [line]
            i += 1
            while i < len(lines):
                nl = lines[i]
                ns = nl.strip()
                if _LIST_ITEM_RE.match(nl):
                    list_lines.append(nl)
                    i += 1
                elif ns and (nl.startswith("  ") or nl.startswith("\t")):
                    # indented continuation of previous item
                    list_lines.append(nl)
                    i += 1
                elif not ns:
                    # blank line — peek ahead for another list item
                    peek = i + 1
                    while peek < len(lines) and not lines[peek].strip():
                        peek += 1
                    if peek < len(lines) and _LIST_ITEM_RE.match(lines[peek]):
                        list_lines.append(nl)
                        i += 1
                    else:
                        break
                else:
                    break
            blocks.append(_Block(
                "list", "\n".join(list_lines),
                heading_title=cur_heading_title,
                heading_depth=cur_heading_depth,
            ))
            continue

        # ── Prose (default) ───────────────────────────────────────────
        prose_buf.append(line)
        i += 1

    _flush_prose()
    return blocks


def _merge_heading_into_content(blocks: List[_Block]) -> List[_Block]:
    """
    Prepend each standalone heading to the next content block so the
    heading text travels with its content (better for retrieval).

    If a heading has no following content block it becomes a small prose chunk.
    """
    merged: List[_Block] = []
    pending: Optional[_Block] = None

    for blk in blocks:
        if blk.type == "heading":
            if pending is not None:
                # previous heading had no content — emit as prose
                merged.append(_Block(
                    "prose", pending.text,
                    heading_title=pending.heading_title,
                    heading_depth=pending.heading_depth,
                ))
            pending = blk
        else:
            if pending is not None:
                blk.text = pending.text + "\n" + blk.text
                blk.heading_title = pending.heading_title
                blk.heading_depth = pending.heading_depth
                pending = None
            merged.append(blk)

    if pending is not None:
        merged.append(_Block(
            "prose", pending.text,
            heading_title=pending.heading_title,
            heading_depth=pending.heading_depth,
        ))

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

@register_chunker("structure_aware")
@register_chunker("document_aware")
class StructureAwareChunker(Chunker):
    """
    Structure-aware document chunker.

    Preserves headings, code, tables, and lists as coherent units.
    Prose sections are sub-chunked via the proven recursive_semantic_chunk
    engine.  Every chunk carries content_type, section_title, section_depth.
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

        code_max  = _safe_int_env("CHUNK_STRUCTURE_CODE_MAX", 2000, 200)
        table_max = _safe_int_env("CHUNK_STRUCTURE_TABLE_MAX", 3000, 200)
        prose_max = _safe_int_env("CHUNK_STRUCTURE_PROSE_MAX", DEFAULT_MAX_CHUNK_LEN, 100)
        prose_min = _safe_int_env("CHUNK_STRUCTURE_PROSE_MIN", DEFAULT_MIN_CHUNK_LEN, 50)
        merge_min_tokens = _safe_int_env("CHUNK_STRUCTURE_MERGE_MIN_TOKENS", 25, 5)

        # ── Phase 1: Pre-process raw text ─────────────────────────────────────
        # Strip page-break markers, OCR garbage, image stubs, LLM prompt
        # prefixes BEFORE structural parsing.  This benefits all block types.
        preprocessed = preprocess_document_text(text)
        if not preprocessed.strip():
            return []

        # ── Phase 2: Structural parsing ───────────────────────────────────────
        blocks = _parse_structural_blocks(preprocessed)
        blocks = _merge_heading_into_content(blocks)
        if not blocks:
            return []

        # ── Phase 3: Block-level chunking ─────────────────────────────────────
        all_chunks: List[Dict[str, Any]] = []
        for blk in blocks:
            blk_chunks = await self._process_block(
                blk,
                code_max=code_max,
                table_max=table_max,
                prose_max=prose_max,
                prose_min=prose_min,
                db_session=db_session,
                file_id=file_id,
                business_id=business_id,
                source_type=source_type,
                embedding_model=embedding_model,
            )
            all_chunks.extend(blk_chunks)

        # ── Phase 4: Post-process — merge tiny chunks with neighbours ─────────
        all_chunks = self._merge_tiny_chunks(all_chunks, merge_min_tokens)

        log_info(
            f"[StructureAware] {len(all_chunks)} chunks from "
            f"{len(blocks)} structural blocks | "
            f"file_id={file_id} | source={source_type}"
        )
        return all_chunks

    # ── Block dispatch ────────────────────────────────────────────────────────

    async def _process_block(
        self,
        blk: _Block,
        *,
        code_max: int,
        table_max: int,
        prose_max: int,
        prose_min: int,
        db_session=None,
        file_id=None,
        business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        kw = dict(file_id=file_id, business_id=business_id,
                   source_type=source_type, embedding_model=embedding_model)

        if blk.type == "code":
            return self._chunk_code(blk, code_max=code_max, **kw)
        if blk.type == "table":
            return self._chunk_table(blk, table_max=table_max, **kw)
        if blk.type == "list":
            return self._chunk_list(blk, max_len=prose_max, **kw)
        # prose (default)
        return await self._chunk_prose(
            blk, max_len=prose_max, min_len=prose_min,
            db_session=db_session, **kw,
        )

    # ── Metadata helper ───────────────────────────────────────────────────────

    @staticmethod
    def _annotate(chunk: Dict[str, Any], blk: _Block, content_type: str) -> None:
        meta = chunk.setdefault("reasoning_ingestion", {})
        meta["chunking_strategy"] = "structure_aware"
        meta["content_type"] = content_type
        meta["section_title"] = blk.heading_title or ""
        meta["section_depth"] = blk.heading_depth
        if blk.language:
            meta["code_language"] = blk.language

    # ── Post-processing: merge tiny chunks with neighbours ────────────────────

    @staticmethod
    def _merge_tiny_chunks(
        chunks: List[Dict[str, Any]],
        min_tokens: int,
    ) -> List[Dict[str, Any]]:
        """
        Merge chunks below min_tokens into their nearest neighbour.

        Rules:
          - A tiny chunk is merged into the NEXT chunk if same section,
            or into the PREVIOUS chunk if no suitable next exists.
          - Code/table chunks are never merged (even if tiny, they are
            structurally distinct).
          - The merged chunk's text concatenates with a newline separator.
          - The merged chunk inherits the larger sibling's metadata.
        """
        if not chunks or min_tokens < 5:
            return chunks

        # Tag indices that need merging
        tiny_indices = set()
        for i, ch in enumerate(chunks):
            meta = ch.get("reasoning_ingestion", {})
            ct = meta.get("content_type", "prose")
            if ct in ("code", "table"):
                continue  # never merge code/table
            if ch.get("tokens", 0) < min_tokens:
                tiny_indices.add(i)

        if not tiny_indices:
            return chunks

        result: List[Dict[str, Any]] = []
        skip = set()
        for i, ch in enumerate(chunks):
            if i in skip:
                continue
            if i in tiny_indices:
                # Try to merge forward
                merged = False
                if i + 1 < len(chunks) and (i + 1) not in skip:
                    next_ch = chunks[i + 1]
                    next_ch["text"] = ch["text"] + "\n" + next_ch["text"]
                    next_ch["cleaned_text"] = ch.get("cleaned_text", "") + " " + next_ch.get("cleaned_text", "")
                    next_ch["tokens"] = ch.get("tokens", 0) + next_ch.get("tokens", 0)
                    merged = True
                # Try to merge backward
                elif result:
                    prev = result[-1]
                    prev["text"] = prev["text"] + "\n" + ch["text"]
                    prev["cleaned_text"] = prev.get("cleaned_text", "") + " " + ch.get("cleaned_text", "")
                    prev["tokens"] = prev.get("tokens", 0) + ch.get("tokens", 0)
                    merged = True
                if not merged:
                    result.append(ch)  # nowhere to merge — keep as-is
            else:
                result.append(ch)

        return result

    # ── Code blocks ───────────────────────────────────────────────────────────

    def _chunk_code(
        self,
        blk: _Block,
        *,
        code_max: int,
        file_id=None,
        business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        text = blk.text
        kw = dict(file_id=file_id, business_id=business_id,
                   source_type=source_type, embedding_model=embedding_model)

        if len(text) <= code_max:
            c = make_chunk_dict(text, **kw)
            if c:
                self._annotate(c, blk, "code")
                return [c]
            return []

        # Oversized → split at line boundaries keeping each segment ≤ code_max
        lines = text.split("\n")
        segments: List[str] = []
        buf: List[str] = []
        buf_len = 0
        for ln in lines:
            if buf_len + len(ln) + 1 > code_max and buf:
                segments.append("\n".join(buf))
                buf = [ln]
                buf_len = len(ln)
            else:
                buf.append(ln)
                buf_len += len(ln) + 1
        if buf:
            segments.append("\n".join(buf))

        out: List[Dict[str, Any]] = []
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            c = make_chunk_dict(seg, **kw)
            if c:
                self._annotate(c, blk, "code")
                out.append(c)
        return out

    # ── Tables ────────────────────────────────────────────────────────────────

    def _chunk_table(
        self,
        blk: _Block,
        *,
        table_max: int,
        file_id=None,
        business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        text = blk.text
        kw = dict(file_id=file_id, business_id=business_id,
                   source_type=source_type, embedding_model=embedding_model)

        if len(text) <= table_max:
            c = make_chunk_dict(text, **kw)
            if c:
                self._annotate(c, blk, "table")
                return [c]
            return []

        # Oversized → split rows, repeating header+separator in each segment
        lines = text.split("\n")
        header_lines = lines[:2] if len(lines) >= 2 else lines[:1]
        data_lines = lines[len(header_lines):]
        header_text = "\n".join(header_lines)
        header_len = len(header_text) + 1

        segments: List[str] = []
        data_buf: List[str] = []
        data_len = header_len
        for dl in data_lines:
            if data_len + len(dl) + 1 > table_max and data_buf:
                segments.append(header_text + "\n" + "\n".join(data_buf))
                data_buf = [dl]
                data_len = header_len + len(dl)
            else:
                data_buf.append(dl)
                data_len += len(dl) + 1
        if data_buf:
            segments.append(header_text + "\n" + "\n".join(data_buf))

        out: List[Dict[str, Any]] = []
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            c = make_chunk_dict(seg, **kw)
            if c:
                self._annotate(c, blk, "table")
                out.append(c)
        return out

    # ── Lists ─────────────────────────────────────────────────────────────────

    def _chunk_list(
        self,
        blk: _Block,
        *,
        max_len: int,
        file_id=None,
        business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        text = blk.text
        kw = dict(file_id=file_id, business_id=business_id,
                   source_type=source_type, embedding_model=embedding_model)

        if len(text) <= max_len:
            c = make_chunk_dict(text, **kw)
            if c:
                self._annotate(c, blk, "list")
                return [c]
            return []

        # Parse individual items, then group into segments ≤ max_len
        items: List[str] = []
        cur_item: List[str] = []
        for ln in text.split("\n"):
            if _LIST_ITEM_RE.match(ln):
                if cur_item:
                    items.append("\n".join(cur_item))
                cur_item = [ln]
            else:
                cur_item.append(ln)
        if cur_item:
            items.append("\n".join(cur_item))

        segments: List[str] = []
        group: List[str] = []
        group_len = 0
        for item in items:
            if group_len + len(item) + 1 > max_len and group:
                segments.append("\n".join(group))
                group = [item]
                group_len = len(item)
            else:
                group.append(item)
                group_len += len(item) + 1
        if group:
            segments.append("\n".join(group))

        out: List[Dict[str, Any]] = []
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            c = make_chunk_dict(seg, **kw)
            if c:
                self._annotate(c, blk, "list")
                out.append(c)
        return out

    # ── Prose ─────────────────────────────────────────────────────────────────

    async def _chunk_prose(
        self,
        blk: _Block,
        *,
        max_len: int,
        min_len: int,
        db_session=None,
        file_id=None,
        business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        sub_chunks = await recursive_semantic_chunk(
            blk.text,
            max_chunk_len=max_len,
            min_chunk_len=min_len,
            db_session=db_session,
            file_id=file_id,
            business_id=business_id,
            source_type=source_type,
            embedding_model=embedding_model,
        )
        for c in sub_chunks:
            self._annotate(c, blk, "prose")
        return sub_chunks
