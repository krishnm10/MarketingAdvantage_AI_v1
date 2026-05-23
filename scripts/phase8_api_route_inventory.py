#!/usr/bin/env python3
"""
Phase 8 — API surface inventory for tenant and auth review.

Run from the repository root inside the normal application virtualenv (full
requirements.txt) so `app.main` can import:

    ENVIRONMENT=test JWT_SECRET_KEY=<strong-secret> python scripts/phase8_api_route_inventory.py

Output: TSV on stdout (method, path, route name, tags, tenant_heuristic).
The tenant_heuristic column flags routes that likely require explicit tenant
context based on path keywords — manual review still required.

This script does not start the server and does not call external services.
"""

from __future__ import annotations

import argparse
import os
import sys


def _tenant_heuristic(path: str) -> str:
    p = path.lower()
    keywords = (
        "ingest",
        "retrieve",
        "rag",
        "client",
        "tenant",
        "business",
        "admin",
        "chunk",
        "file",
        "pipeline",
        "embedding",
        "alignment",
        "config",
        "customer",
    )
    if any(k in p for k in keywords):
        return "review_tenant"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="List FastAPI routes for Phase 8 audit.")
    parser.add_argument(
        "--format",
        choices=("tsv", "md"),
        default="tsv",
        help="Output format (default: tsv)",
    )
    args = parser.parse_args()

    # JWT and environment must satisfy app.auth.generate_token import guards.
    os.environ.setdefault(
        "JWT_SECRET_KEY",
        "phase8_route_inventory_only_" + ("z" * 48),
    )
    os.environ.setdefault("ENVIRONMENT", "test")

    try:
        from fastapi.routing import APIRoute

        from app.main import app
    except Exception as exc:  # pragma: no cover - environment-specific
        print(
            "ERROR: Could not import FastAPI app. Use the project venv and full deps.\n",
            exc,
            file=sys.stderr,
        )
        return 1

    rows: list[tuple[str, str, str, str, str]] = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        tags = ",".join(sorted(route.tags)) if route.tags else ""
        name = route.name or ""
        for method in sorted(route.methods):
            if method == "HEAD":
                continue
            hint = _tenant_heuristic(route.path)
            rows.append((method, route.path, name, tags, hint))

    rows.sort(key=lambda r: (r[1], r[0]))

    if args.format == "md":
        print("| Method | Path | Name | Tags | Tenant hint |")
        print("| --- | --- | --- | --- | --- |")
        for method, path, name, tags, hint in rows:
            print(f"| {method} | `{path}` | {name} | {tags} | {hint or '—'} |")
    else:
        print("method\tpath\tname\ttags\ttenant_heuristic")
        for row in rows:
            print("\t".join(row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
