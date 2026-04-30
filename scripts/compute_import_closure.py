#!/usr/bin/env python3
"""
Compute a static import-reachable closure from app/main.py over the ``app.*`` package.

Output: docs/import_closure_from_main.md

Limitations:
  * Dynamic imports are invisible.
  * ``from pkg import *`` does not expand ``pkg``'s submodules.
  * Stdlib / site-packages are recorded by top-level name only (not expanded).
"""
from __future__ import annotations

import ast
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"


@dataclass
class ClosureStats:
    app_modules_seen: Set[str] = field(default_factory=set)
    external_roots: Set[str] = field(default_factory=set)
    parse_errors: Dict[str, str] = field(default_factory=dict)


def path_to_app_module(py_path: Path) -> Optional[str]:
    try:
        rel = py_path.resolve().relative_to(APP_DIR.resolve())
    except ValueError:
        return None
    parts = list(rel.parts)
    if "__pycache__" in parts:
        return None
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        assert parts[-1].endswith(".py")
        parts[-1] = parts[-1][:-3]
    return "app" if not parts else "app." + ".".join(parts)


def resolve_app_module(mod: str) -> Optional[Path]:
    if mod != "app" and not mod.startswith("app."):
        return None
    if mod == "app":
        ai = APP_DIR / "__init__.py"
        return ai if ai.is_file() else None

    suf = mod[4:]
    segs = suf.split(".")
    p = APP_DIR.joinpath(*segs)
    f1, f2 = p.with_suffix(".py"), p / "__init__.py"
    if f1.is_file():
        return f1
    if f2.is_file():
        return f2
    return None


def import_context_package(current_module: str) -> Optional[str]:
    """__package__-style dotted name for the file implementing ``current_module``."""
    path = resolve_app_module(current_module)
    if path is None:
        return None
    if path.name == "__init__.py":
        return path_to_app_module(path)
    rel_dirs = path.parent.relative_to(APP_DIR).parts
    return "app" if not rel_dirs else "app." + ".".join(rel_dirs)


def relative_anchor(current_module: str, level: int, rel_mod: Optional[str]) -> Optional[str]:
    """
    Anchor module for PEP 328 ``from `.` …`` / ``from `..`` …``.
    """
    pkg = import_context_package(current_module)
    if pkg is None or level <= 0:
        return None
    base_parts = pkg.split(".")
    up = level - 1
    if len(base_parts) < up:
        return None
    anchor_parts = base_parts[:-up] if up else base_parts[:]
    if not anchor_parts or anchor_parts[0] != "app":
        return None
    head = ".".join(anchor_parts)
    if not rel_mod:
        return head
    tail = ".".join([head, rel_mod])
    return tail if tail.startswith("app") else None


def iter_import_edges(tree: ast.AST, current_module: str) -> Iterable[Tuple[str, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                nm = alias.name
                if nm == "app" or nm.startswith("app."):
                    yield ("app", nm)
                else:
                    yield ("ext", nm.split(".", 1)[0])
            continue

        if not isinstance(node, ast.ImportFrom):
            continue

        lvl = int(node.level or 0)
        mod = node.module

        if lvl == 0:
            if not mod:
                continue
            if mod == "app" or mod.startswith("app."):
                yield ("app", mod)
                for alias in node.names:
                    if alias.name != "*":
                        cand = f"{mod}.{alias.name}"
                        if resolve_app_module(cand):
                            yield ("app", cand)
            else:
                yield ("ext", mod.split(".", 1)[0])
            continue

        anchor = relative_anchor(current_module, lvl, mod)
        if anchor is None:
            continue
        if mod:
            if resolve_app_module(anchor):
                yield ("app", anchor)
            for alias in node.names:
                if alias.name == "*":
                    continue
                cand = f"{anchor}.{alias.name}"
                if resolve_app_module(cand):
                    yield ("app", cand)
        else:
            for alias in node.names:
                if alias.name == "*":
                    yield ("app", anchor)
                    continue
                cand = f"{anchor}.{alias.name}"
                if resolve_app_module(cand):
                    yield ("app", cand)


def parse_file(py_path: Path) -> Tuple[Optional[ast.AST], Optional[str]]:
    try:
        text = py_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = py_path.read_text(encoding="utf-8", errors="replace")
    try:
        return ast.parse(text, filename=str(py_path)), None
    except SyntaxError as e:
        return None, f"{type(e).__name__}: {e}"


def bfs_closure(entry_module: str) -> ClosureStats:
    stats = ClosureStats()
    q: deque[str] = deque()
    seen: Set[str] = set()

    def enqueue(mod: str) -> None:
        if mod in seen:
            return
        if mod != "app" and not mod.startswith("app."):
            return
        if resolve_app_module(mod) is None:
            return
        seen.add(mod)
        stats.app_modules_seen.add(mod)
        q.append(mod)

    enqueue(entry_module)

    while q:
        mod = q.popleft()
        path = resolve_app_module(mod)
        if path is None:
            continue
        tree, err = parse_file(path)
        if err:
            stats.parse_errors[mod] = err
            continue
        assert tree is not None

        for kind, pay in iter_import_edges(tree, mod):
            if kind == "ext":
                stats.external_roots.add(pay)
            elif kind == "app":
                enqueue(pay)

    return stats


def list_all_app_modules() -> Set[str]:
    out: Set[str] = set()
    for p in APP_DIR.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        m = path_to_app_module(p)
        if m:
            out.add(m)
    return out


def main() -> int:
    out_md = REPO_ROOT / "docs" / "import_closure_from_main.md"

    stats = bfs_closure("app.main")

    universe = list_all_app_modules()
    reachable = stats.app_modules_seen
    orphans = sorted(universe - reachable)

    lines = [
        "# Import closure from `app.main` (static AST)",
        "",
        f"- **Repo root:** `{REPO_ROOT}`",
        "- **Entry:** `app.main`",
        f"- **`app.*` reachable (closure):** {len(reachable)}",
        f"- **All `app/**/*.py` on disk:** {len(universe)}",
        f"- **On disk − closure:** {len(orphans)}",
        "",
        "## Interpreting “orphans”",
        "",
        "Modules listed below are reachable **only if** imported from `main.py` indirectly by ordinary `import` / `from` statements in the Python source of the closure. They may still be required when:",
        "",
        "- Imported via **reflective loaders** (`importlib`), dynamic strings, Celery/autodiscover.",
        "- Only used from **alternate entrypoints** (CLI scripts, Jupyter, workers launched without importing `main`).",
        "- **Tests** importing them.",
        "",
        "Treat orphans as review candidates—not automatic deletion targets.",
        "",
        "## Top-level externals touched from closure",
        "",
        *[f"- `{n}`" for n in sorted(stats.external_roots)],
        "",
        "## Parse errors within traversed modules",
        "",
        *(
            [f"- `{m}`: {t}" for m, t in sorted(stats.parse_errors.items())]
            if stats.parse_errors
            else ["(none)"]
        ),
        "",
        "---",
        "",
        "## `app.*` in closure",
        "",
        *[f"- `{m}`" for m in sorted(reachable)],
        "",
        "---",
        "",
        "## `app.*` on disk not in closure",
        "",
        *([f"- `{m}`" for m in orphans] if orphans else ["(closure covers all files seen)"]),
        "",
        "---",
        "",
        "*Generated by `scripts/compute_import_closure.py`.*",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    rc = int(bool(stats.parse_errors))
    print(
        f"Wrote {out_md.relative_to(REPO_ROOT)} — "
        f"reachable {len(reachable)} / universe {len(universe)} ({len(orphans)} orphans)",
        flush=True,
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
