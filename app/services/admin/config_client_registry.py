"""
Enumerate client/tenant IDs from on-disk config files (authoritative for admin dashboard).
Skips template/prompt subdirectories and non-client JSON roots.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Set

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIRS = [
    _REPO_ROOT / "app" / "core" / "configs",
    _REPO_ROOT / "configs",
]


def get_config_search_paths() -> List[str]:
    """Resolved directory paths used for client JSON discovery."""
    return [str(p.resolve()) for p in _CONFIG_DIRS if p.is_dir()]

def list_config_client_stems() -> List[str]:
    """
    Collect unique client config stems (filenames without extension) from
    top-level ``*.json`` files in known config dirs (not under
    ``pipeline_templates/`` or ``prompts/`` — those paths are never matched by
    a top-level glob).
    """
    found: Set[str] = set()
    for base in _CONFIG_DIRS:
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*.json")):
            stem = path.stem
            if not stem or stem.startswith("."):
                continue
            found.add(stem)

    return sorted(found)
