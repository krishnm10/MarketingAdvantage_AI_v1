"""
Tenant configuration storage abstraction (Phase 2).

Decouples config resolution from direct filesystem reads and provides a
version key for distributed cache invalidation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from app.utils.path_sanitizer import sanitize_client_id

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_ID = "default"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG_DIRS: List[Path] = [
    _REPO_ROOT / "app" / "core" / "configs",
    _REPO_ROOT / "configs",
]


def _canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, default=str)


def compute_config_version(data: Dict[str, Any]) -> str:
    """Stable content hash used as a cache invalidation key."""
    digest = hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()
    return digest[:16]


@runtime_checkable
class ConfigStore(Protocol):
    """Storage backend for tenant JSON configuration."""

    def load_raw(self, client_id: str) -> Dict[str, Any]:
        """Return merged default+client raw dict (pre-validation)."""
        ...

    def get_version(self, client_id: str) -> str:
        """Return a cache invalidation key for the tenant's effective config."""
        ...

    def save_raw(self, client_id: str, data: Dict[str, Any]) -> str:
        """Persist a client override file and return the new version."""
        ...


class FileSystemConfigStore:
    """
    Filesystem-backed ConfigStore.

    Resolution order matches the legacy resolver: ``default.json`` deep-merged
    with ``{client_id}.json`` when present.
    """

    def __init__(self, config_dirs: Optional[List[Path]] = None) -> None:
        self._config_dirs = list(config_dirs or _DEFAULT_CONFIG_DIRS)

    @property
    def config_dirs(self) -> List[Path]:
        return list(self._config_dirs)

    def find_config_path(self, client_id: str) -> Optional[Path]:
        """Return the first matching config file path for *client_id*, if any."""
        safe_id = sanitize_client_id(client_id)
        for base in self._config_dirs:
            if not base.is_dir():
                continue
            resolved_base = base.resolve()
            for ext in ("json", "yaml", "yml"):
                candidate = (base / f"{safe_id}.{ext}").resolve()
                if not str(candidate).startswith(str(resolved_base)):
                    logger.warning(
                        "[ConfigStore] Path traversal blocked for client_id=%.30s",
                        client_id[:30],
                    )
                    continue
                if candidate.is_file():
                    return candidate
        return None

    def has_client_override(self, client_id: str) -> bool:
        return (
            client_id != DEFAULT_CONFIG_ID
            and self.find_config_path(client_id) is not None
        )

    def load_source_label(self, client_id: str) -> str:
        if self.has_client_override(client_id):
            return f"default+{client_id}"
        return "default"

    def _read_file(self, path: Path) -> Dict[str, Any]:
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "PyYAML required for YAML configs: pip install pyyaml"
                ) from exc
            return yaml.safe_load(text) or {}
        return json.loads(text)

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(base)
        for key, override_val in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(override_val, dict)
            ):
                merged[key] = self._deep_merge(merged[key], override_val)
            else:
                merged[key] = deepcopy(override_val)
        return merged

    def _merged_raw(self, client_id: str) -> Dict[str, Any]:
        default_path = self.find_config_path(DEFAULT_CONFIG_ID)
        if default_path is None:
            searched = [str(d) for d in self._config_dirs]
            raise FileNotFoundError(
                f"Default config not found. Searched: {searched}"
            )

        base_raw = self._read_file(default_path)
        client_path = self.find_config_path(client_id)
        if client_path is not None and client_id != DEFAULT_CONFIG_ID:
            client_raw = self._read_file(client_path)
            return self._deep_merge(base_raw, client_raw)
        return deepcopy(base_raw)

    def load_raw(self, client_id: str) -> Dict[str, Any]:
        safe_id = sanitize_client_id(client_id)
        return self._merged_raw(safe_id)

    def get_version(self, client_id: str) -> str:
        safe_id = sanitize_client_id(client_id)
        merged = self._merged_raw(safe_id)
        merged["client_id"] = safe_id
        return compute_config_version(merged)

    def save_raw(self, client_id: str, data: Dict[str, Any]) -> str:
        safe_id = sanitize_client_id(client_id)
        if safe_id == DEFAULT_CONFIG_ID:
            raise ValueError("Refusing to overwrite default config via save_raw.")

        target_dir = self._config_dirs[0]
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = (target_dir / f"{safe_id}.json").resolve()
        if not str(out_path).startswith(str(target_dir.resolve())):
            raise ValueError("Refusing to write outside config directory.")

        payload = deepcopy(data)
        payload["client_id"] = safe_id
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(out_path)

        version = self.get_version(safe_id)
        logger.info(
            "[ConfigStore] Saved client config client_id=%r version=%s path=%s",
            safe_id,
            version,
            out_path,
        )

        from app.core.config.config_change_bus import get_config_change_bus

        get_config_change_bus().publish(safe_id, version)
        return version


_default_store: Optional[ConfigStore] = None


def get_config_store() -> ConfigStore:
    global _default_store
    if _default_store is None:
        _default_store = FileSystemConfigStore()
    return _default_store


def set_config_store(store: Optional[ConfigStore]) -> None:
    """Replace the process-wide ConfigStore (tests / future DI wiring)."""
    global _default_store
    _default_store = store
