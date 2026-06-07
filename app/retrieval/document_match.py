from __future__ import annotations

"""
Shared, domain-neutral document match contract for debug-only analysis.

Adapters may project domain-specific signals into the attributes dict and
expose lightweight evidence such as sample chunk ids. The analysis core
only relies on file_id and the ability to serialize to a JSON-safe dict.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class DocumentMatch:
    file_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    sample_chunk_ids: List[str] = field(default_factory=list)

    def to_debug_dict(self) -> Dict[str, Any]:
        """
        Normalize to a compact, JSON-serializable payload.

        Phase 2/3: top-level keys remain adapter-defined; the core requires
        only file_id, while attributes may contain domain-specific signals
        like 'overdue' or 'tax_gt_threshold', and sample_chunk_ids provide
        minimal evidence for debugging.
        """
        out: Dict[str, Any] = {"file_id": self.file_id}
        out.update(self.attributes or {})
        if self.sample_chunk_ids:
            out["sample_chunk_ids"] = list(self.sample_chunk_ids)
        return out

