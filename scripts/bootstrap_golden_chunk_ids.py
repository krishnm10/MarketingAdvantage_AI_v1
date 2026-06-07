#!/usr/bin/env python3
"""
Bootstrap relevant_chunk_ids for a golden-set case via live chat retrieval.

Example:
  python scripts/bootstrap_golden_chunk_ids.py --set invoice/vaidyanad_inv_1101.json --case inv_1101_total --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ai.evaluation.golden_set_loader import load_golden_set, resolve_golden_set_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", required=True)
    parser.add_argument("--case", required=True, help="question_id to update")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--write", action="store_true", help="Patch JSON file in place")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--admin-user", default="admin")
    parser.add_argument("--admin-pass", default="admin")
    args = parser.parse_args()

    golden = load_golden_set(args.set)
    path = resolve_golden_set_path(args.set)

    case = next((c for c in golden.cases if c.question_id == args.case), None)
    if case is None:
        print(f"Case not found: {args.case}", file=sys.stderr)
        return 1

    base = args.base_url.rstrip("/")
    with httpx.Client() as client:
        auth = client.post(
            f"{base}/api/v2/auth/token",
            data={"username": args.admin_user, "password": args.admin_pass},
        )
        if auth.status_code != 200:
            print(f"Auth failed: {auth.status_code}", file=sys.stderr)
            return 1
        headers = {"Authorization": f"Bearer {auth.json()['access_token']}"}
        r = client.post(
            f"{base}/api/v2/retrieve/chat",
            json={
                "session_id": str(uuid.uuid4()),
                "messages": [{"role": "user", "content": case.question}],
                "client_id": golden.tenant_id,
                "top_k": args.top_k,
                "generate_answer": False,
            },
            headers=headers,
            timeout=120.0,
        )
        if r.status_code != 200:
            print(f"Chat failed: {r.status_code} {r.text[:400]}", file=sys.stderr)
            return 1
        body = r.json()

    chunk_ids = [str(row["chunk_id"]) for row in body.get("results") or [] if row.get("chunk_id")]
    print(f"question_id={args.case}")
    print(f"ranked_chunk_ids ({len(chunk_ids)}):")
    for cid in chunk_ids:
        print(f"  {cid}")

    if args.write:
        data = json.loads(path.read_text(encoding="utf-8"))
        for raw in data.get("cases") or []:
            if raw.get("question_id") == args.case:
                raw["relevant_chunk_ids"] = chunk_ids
                meta = raw.get("metadata") or {}
                meta["skip_retrieval_metrics_if_empty"] = False
                raw["metadata"] = meta
                break
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
