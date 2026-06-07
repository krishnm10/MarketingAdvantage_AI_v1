#!/usr/bin/env python3
"""
Run a golden-set evaluation against a live MAI API.

Example:
  python scripts/run_golden_set_eval.py --set invoice/vaidyanad_inv_1101.json --paths chat,rag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ai.evaluation.golden_set_runner import GoldenSetRunner, check_thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description="MAI golden-set evaluation runner")
    parser.add_argument(
        "--set",
        required=True,
        help="Golden set path relative to tests/golden_sets/ or absolute",
    )
    parser.add_argument(
        "--paths",
        default="chat,rag",
        help="Comma-separated: chat,rag",
    )
    parser.add_argument(
        "--no-faithfulness",
        action="store_true",
        help="Skip LLM-as-judge faithfulness (retrieval + substring only)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Write JSON report to this file",
    )
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit 1 if GOLDEN_MIN_* env thresholds are violated",
    )
    args = parser.parse_args()

    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    runner = GoldenSetRunner()
    report = runner.run(
        args.set,
        paths=paths,  # type: ignore[arg-type]
        run_faithfulness=not args.no_faithfulness,
    )
    payload = report.to_dict()
    text = json.dumps(payload, indent=2)
    print(text)

    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")

    failed = 0
    for case in report.case_results:
        if case.errors:
            failed += 1
        if case.substring_checks and not case.substring_checks.passed:
            failed += 1
        if case.route_check and not case.route_check.passed:
            failed += 1

    if args.fail_on_threshold:
        violations = check_thresholds(report)
        if violations:
            print("Threshold violations:", file=sys.stderr)
            for v in violations:
                print(f"  - {v}", file=sys.stderr)
            failed += len(violations)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
