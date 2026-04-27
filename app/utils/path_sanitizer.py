"""
Path sanitization utilities for client IDs, config paths, and slugs.

Prevents path traversal, directory escape, and injection attacks by enforcing
strict character allowlists and containment checks on all filesystem paths
derived from user-supplied identifiers.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_SAFE_ID_RE = re.compile(r"[^a-z0-9_\-]")
_SLUG_RE = re.compile(r"^[a-z0-9_\-]+$")


def sanitize_client_id(client_id: str | None) -> str:
    """
    Normalize a client ID into a filesystem-safe, lowercase slug.

    Raises ValueError for traversal attempts containing "..", "/" or "\\".
    Returns "default" for empty/None inputs.
    Max 64 characters after sanitization.
    """
    if client_id is None or not str(client_id).strip():
        return "default"

    raw = str(client_id).strip()

    if ".." in raw or "/" in raw or "\\" in raw:
        logger.warning(
            "Path traversal attempt rejected: %.20s...", raw[:20],
        )
        raise ValueError("Invalid client_id: path traversal characters detected")

    safe = os.path.basename(raw)
    safe = _SAFE_ID_RE.sub("", safe.lower())[:64]

    if not safe:
        logger.warning(
            "client_id sanitized to empty string, using default: %.20s...",
            raw[:20],
        )
        return "default"

    return safe


def safe_config_path(
    base_dir: Path, client_id: str, extension: str = "json",
) -> Path:
    """
    Build a safe config file path from base_dir + sanitized client_id + extension.

    Raises ValueError if the resolved path escapes base_dir (containment check).
    """
    clean_id = sanitize_client_id(client_id)
    result = (base_dir / f"{clean_id}.{extension}").resolve()
    base_resolved = base_dir.resolve()

    if not str(result).startswith(str(base_resolved)):
        logger.warning(
            "Path containment violation for client_id=%.20s...", str(client_id)[:20],
        )
        raise ValueError("Resolved path escapes base directory")

    return result


def validate_slug(slug: str, max_length: int = 64) -> str:
    """
    Validate a slug (template ID, tag, etc.) against a strict allowlist.

    Only [a-z0-9_-] allowed, non-empty, up to max_length characters.
    """
    if not slug or not slug.strip():
        raise ValueError("Slug must be non-empty")

    value = slug.strip().lower()

    if len(value) > max_length:
        logger.warning(
            "Slug exceeds max length (%d): %.20s...", max_length, value[:20],
        )
        raise ValueError(f"Slug exceeds maximum length of {max_length}")

    if not _SLUG_RE.match(value):
        logger.warning("Slug contains invalid characters: %.20s...", value[:20])
        raise ValueError(
            "Slug contains invalid characters (only a-z, 0-9, _, - allowed)"
        )

    return value
