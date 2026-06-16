#!/usr/bin/env python3
"""
Phase 7 — Migrate legacy tenant JSON configs to strict v2 schemas.

Transforms:
  - Injects ``secrets_backend.provider = env`` when missing.
  - Maps ``api_key_env`` / ``token_env`` / ``password_env`` → ``secret_ref.uri`` (env://…).
  - Removes legacy env key fields after migration.

Usage (from repository root):

  # Preview changes without writing
  python scripts/migrations/migrate_tenant_configs_v2.py --dry-run

  # Migrate production configs with timestamped backups
  python scripts/migrations/migrate_tenant_configs_v2.py --backup

  # Point at an isolated test directory (copy default.json + tenants first)
  python scripts/migrations/migrate_tenant_configs_v2.py \\
      --config-dir /tmp/tenant-config-test --dry-run

Requires: repository root on PYTHONPATH (script inserts it automatically).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pydantic import ValidationError

from app.core.config.client_config_resolver import (  # noqa: E402
    IssueSeverity,
    validate_config_compatibility,
)
from app.core.config.client_config_schema import ClientConfig  # noqa: E402
from app.core.config.secret_ref import migrate_legacy_secret_fields  # noqa: E402

DEFAULT_CONFIG_DIR = _REPO_ROOT / "app" / "core" / "configs"
DEFAULT_SECRETS_BACKEND: Dict[str, str] = {"provider": "env"}


@dataclass
class MigrationResult:
    path: Path
    client_id: str
    changed: bool = False
    dry_run: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _migrate_tree(node: Any) -> Any:
    """Recursively migrate legacy secret fields in nested dicts."""
    if isinstance(node, dict):
        children = {k: _migrate_tree(v) for k, v in node.items()}
        return migrate_legacy_secret_fields(children)
    if isinstance(node, list):
        return [_migrate_tree(item) for item in node]
    return node


def transform_tenant_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Apply Phase 7 transformations to a raw tenant JSON dict."""
    data = _migrate_tree(dict(raw))

    backend = data.get("secrets_backend")
    if not isinstance(backend, dict) or not backend.get("provider"):
        data["secrets_backend"] = dict(DEFAULT_SECRETS_BACKEND)

    client_id = data.get("client_id")
    if isinstance(client_id, str) and client_id.strip():
        data["client_id"] = client_id.strip()

    return data


def validate_tenant_dict(data: Dict[str, Any]) -> tuple[ClientConfig, List[str]]:
    """Parse through ClientConfig and collect compatibility warnings."""
    try:
        config = ClientConfig.from_dict(data)
    except ValidationError as exc:
        raise ValueError(f"Pydantic validation failed: {exc}") from exc

    warnings: List[str] = []
    for issue in validate_config_compatibility(config):
        msg = f"[{issue.component}] {issue.message}"
        if issue.severity == IssueSeverity.ERROR:
            raise ValueError(msg)
        warnings.append(msg)
    return config, warnings


def _canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, default=str)


def migrate_file(
    path: Path,
    *,
    dry_run: bool,
    backup_dir: Optional[Path],
) -> MigrationResult:
    result = MigrationResult(path=path, client_id="", dry_run=dry_run)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result.errors.append(f"read failed: {exc}")
        return result

    if not isinstance(raw, dict):
        result.errors.append("top-level JSON must be an object")
        return result

    result.client_id = str(raw.get("client_id") or path.stem)

    before = _canonical_json(raw)
    try:
        transformed = transform_tenant_dict(raw)
        config, warnings = validate_tenant_dict(transformed)
        result.warnings.extend(warnings)
        result.client_id = config.client_id
    except ValueError as exc:
        result.errors.append(str(exc))
        return result

    after = _canonical_json(transformed)
    result.changed = before != after

    if not result.changed:
        return result

    if dry_run:
        return result

    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = backup_dir / f"{path.stem}.{stamp}.json.bak"
        shutil.copy2(path, backup_path)

    payload = json.dumps(transformed, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".migrate_tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return result


def discover_json_files(config_dir: Path) -> Sequence[Path]:
    return sorted(config_dir.glob("*.json"))


def run_migration(
    config_dir: Path,
    *,
    dry_run: bool,
    backup: bool,
) -> List[MigrationResult]:
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    backup_dir = config_dir / "_migrations_backup" if backup and not dry_run else None
    results: List[MigrationResult] = []

    for path in discover_json_files(config_dir):
        results.append(
            migrate_file(path, dry_run=dry_run, backup_dir=backup_dir)
        )
    return results


def _print_report(results: Sequence[MigrationResult]) -> int:
    changed = [r for r in results if r.changed and not r.errors]
    failed = [r for r in results if r.errors]
    unchanged = [r for r in results if not r.changed and not r.errors]

    print(f"Processed {len(results)} file(s).")
    print(f"  Changed:   {len(changed)}")
    print(f"  Unchanged: {len(unchanged)}")
    print(f"  Failed:    {len(failed)}")

    for r in changed:
        mode = "would update" if r.dry_run else "updated"
        print(f"  [{mode}] {r.path.name} (client_id={r.client_id})")
        for w in r.warnings:
            print(f"    warn: {w}")

    for r in failed:
        print(f"  [FAILED] {r.path.name}: {'; '.join(r.errors)}")

    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate tenant JSON configs to Phase 7 strict schemas.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help=f"Directory containing tenant JSON files (default: {DEFAULT_CONFIG_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report changes without writing files.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Copy originals to <config-dir>/_migrations_backup/ before writing.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.backup and args.dry_run:
        print("Note: --backup is ignored when --dry-run is set.")

    try:
        results = run_migration(
            args.config_dir.resolve(),
            dry_run=args.dry_run,
            backup=args.backup,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return _print_report(results)


if __name__ == "__main__":
    raise SystemExit(main())
