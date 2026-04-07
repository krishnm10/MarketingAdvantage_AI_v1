"""
Test All Chunking Strategies — One by One
==========================================
Runs each registered chunking strategy against multiple document types
(narrative, data-heavy, mixed structure, short text) and reports:
  • chunk count & token stats
  • quality score distribution
  • required field validation
  • pass / fail per strategy

Usage:
    python test_all_chunking_strategies.py                 # test ALL strategies
    python test_all_chunking_strategies.py semantic overlap # test specific ones
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

# ── ensure project root on path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.chunking_stratagies.chunking_registry import (
    clear_chunker_cache,
    get_chunker,
    list_chunking_strategies,
)

# ── disable LLM-dependent features for offline testing ───────────────────────
os.environ.setdefault("CHUNK_ELITE_PROPOSITIONS", "false")
os.environ.setdefault("CHUNK_ELITE_AGENTIC_SPLIT", "false")


# ═══════════════════════════════════════════════════════════════════════════════
#  Sample Documents
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_NARRATIVE = """
Marketing Strategy Report — Q4 2025

Executive Summary
Our Q4 campaign outperformed all targets. Social media engagement grew by 42 percent,
email open rates climbed to 28 percent, and the cost per acquisition dropped below $12
for the first time. This positions us well for aggressive expansion in 2026.

Channel Performance
Paid search delivered the highest ROAS at 5.2x, followed by programmatic display at 3.8x.
Organic social continued its upward trend with a 15 percent increase in follower count
across all major platforms. The new TikTok content series generated 2.3 million views
within its first month, exceeding the target by 180 percent.

Email campaigns targeting lapsed customers achieved a 19 percent re-engagement rate,
contributing $340K in recovered revenue. The segmentation model introduced in October
proved its value by lifting conversion rates 22 percent over the prior quarter.

Recommendations
1. Increase paid search budget by 30 percent to capture the spring seasonal uplift.
2. Expand the TikTok content series with a weekly cadence.
3. Deploy the next-generation segmentation model across all email campaigns.
4. Pilot an influencer micro-partnership programme in the wellness vertical.

Budget Implications
The proposed changes require an incremental budget of $120K per quarter. Given the
projected ROAS improvement, the programme should become self-funding within two quarters.
Finance has pre-approved the expenditure contingent on maintaining CPA below $15.
""".strip()

SAMPLE_DATA_HEAVY = """
Quarterly Revenue Data — FY 2025

| Region       | Q1 ($M) | Q2 ($M) | Q3 ($M) | Q4 ($M) | YoY Growth |
|-------------|---------|---------|---------|---------|------------|
| North America| 42.3    | 45.1    | 48.7    | 54.2    | +18%       |
| EMEA         | 28.1    | 29.4    | 31.2    | 35.8    | +22%       |
| APAC         | 19.7    | 21.3    | 23.9    | 27.1    | +25%       |
| LATAM        | 8.4     | 9.1     | 10.2    | 12.5    | +31%       |

Key Metrics:
- Total revenue: $456.0M (up 21% YoY)
- Gross margin: 68.4% (up 2.1pp)
- Operating margin: 24.7% (up 3.3pp)
- Net revenue retention: 118%
- Customer count: 4,271 (up 340 net new)

Top performing SKUs by revenue contribution:
1. Enterprise Platform License — $189.2M (41.5%)
2. Professional Services — $87.3M (19.1%)
3. Data Analytics Add-on — $72.8M (16.0%)
4. API Gateway Tier 3 — $54.1M (11.9%)
5. Compliance Module — $52.6M (11.5%)

Cash flow from operations: $112.4M. Free cash flow: $89.7M. Cash & equivalents: $215.3M.
Board has authorized a $50M share repurchase programme effective January 2026.
""".strip()

SAMPLE_MIXED_STRUCTURE = """
# API Integration Guide v3.2

## Authentication

All requests must include a Bearer token in the `Authorization` header.
Tokens are obtained via the OAuth 2.0 client credentials flow.

```python
import requests

resp = requests.post(
    "https://auth.example.com/oauth/token",
    json={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    },
)
token = resp.json()["access_token"]
```

## Endpoints

### POST /api/v3/campaigns

Create a new marketing campaign.

**Request Body:**
| Field        | Type   | Required | Description                  |
|-------------|--------|----------|------------------------------|
| name         | string | yes      | Campaign name (max 120 chars)|
| budget       | number | yes      | Daily budget in USD          |
| start_date   | string | yes      | ISO-8601 date                |
| end_date     | string | no       | ISO-8601 date (optional)     |
| targeting    | object | yes      | Audience targeting rules     |

**Response:** `201 Created`
```json
{
  "id": "camp_abc123",
  "status": "draft",
  "created_at": "2025-11-15T10:30:00Z"
}
```

### GET /api/v3/campaigns/{id}/metrics

Returns real-time performance metrics for a campaign.

**Query Parameters:**
- `granularity` — `hourly`, `daily`, or `weekly` (default: `daily`)
- `start` — ISO-8601 start date
- `end` — ISO-8601 end date

## Rate Limits

| Tier       | Requests/min | Burst |
|-----------|-------------|-------|
| Free       | 60          | 10    |
| Pro        | 600         | 100   |
| Enterprise | 6000        | 1000  |

Exceeding limits returns `429 Too Many Requests` with a `Retry-After` header.

## Error Codes

All errors follow RFC 7807 Problem Details format:
```json
{
  "type": "https://api.example.com/errors/rate-limit",
  "title": "Rate Limit Exceeded",
  "status": 429,
  "detail": "You have exceeded 60 requests per minute.",
  "instance": "/api/v3/campaigns"
}
```
""".strip()

SAMPLE_SHORT = """
Quick reminder: the brand guidelines document has been updated. Please review
the new colour palette and typography choices before the Friday all-hands.
The Figma link is pinned in the #design Slack channel.
""".strip()

SAMPLES = {
    "narrative":  SAMPLE_NARRATIVE,
    "data_heavy": SAMPLE_DATA_HEAVY,
    "mixed":      SAMPLE_MIXED_STRUCTURE,
    "short":      SAMPLE_SHORT,
}

# Required top-level keys every chunk must have
REQUIRED_KEYS = {"text", "semantic_hash", "tokens", "reasoning_ingestion"}


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_header(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_subheader(title: str) -> None:
    print(f"\n  ── {title} {'─' * max(1, 50 - len(title))}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-Strategy Test Runner
# ═══════════════════════════════════════════════════════════════════════════════

async def test_single_strategy(
    strategy_name: str,
) -> Dict[str, Any]:
    """
    Run one strategy against all sample documents.
    Returns a result dict with pass/fail info.
    """
    result: Dict[str, Any] = {
        "strategy": strategy_name,
        "passed": True,
        "errors": [],
        "samples": {},
    }

    try:
        clear_chunker_cache()
        chunker = get_chunker(strategy_name)
    except Exception as exc:
        result["passed"] = False
        result["errors"].append(f"Failed to get chunker: {exc}")
        return result

    for sample_name, sample_text in SAMPLES.items():
        sample_result: Dict[str, Any] = {
            "chunk_count": 0,
            "total_tokens": 0,
            "avg_tokens": 0.0,
            "avg_quality": 0.0,
            "min_quality": None,
            "max_quality": None,
            "missing_keys": [],
            "error": None,
            "elapsed_ms": 0.0,
        }

        t0 = time.perf_counter()
        try:
            chunks = await chunker.chunk(
                sample_text,
                file_id=f"test-{strategy_name}-{sample_name}",
                source_type="txt",
                embedding_model="test-embed-model",
            )

            elapsed = (time.perf_counter() - t0) * 1000
            sample_result["elapsed_ms"] = round(elapsed, 1)

            if not isinstance(chunks, list):
                raise TypeError(f"Expected list, got {type(chunks).__name__}")

            sample_result["chunk_count"] = len(chunks)

            if len(chunks) == 0 and sample_name != "short":
                result["errors"].append(
                    f"[{sample_name}] Produced 0 chunks for non-trivial input"
                )
                result["passed"] = False

            token_counts: List[int] = []
            quality_scores: List[float] = []

            for idx, chunk in enumerate(chunks):
                # ── field validation ──────────────────────────────────
                for key in REQUIRED_KEYS:
                    if key not in chunk:
                        sample_result["missing_keys"].append(f"chunk[{idx}].{key}")

                # text must be non-empty string
                text = chunk.get("text", "")
                if not isinstance(text, str) or len(text.strip()) == 0:
                    result["errors"].append(
                        f"[{sample_name}] chunk[{idx}] has empty/missing text"
                    )
                    result["passed"] = False

                # token count
                tok = chunk.get("tokens", 0)
                if isinstance(tok, (int, float)) and tok > 0:
                    token_counts.append(int(tok))

                # quality score
                ri = chunk.get("reasoning_ingestion", {})
                qs = ri.get("chunk_quality_score")
                if isinstance(qs, (int, float)):
                    quality_scores.append(float(qs))

                # check chunking_strategy tag
                cs = ri.get("chunking_strategy")
                if cs is None:
                    result["errors"].append(
                        f"[{sample_name}] chunk[{idx}] missing reasoning_ingestion.chunking_strategy"
                    )
                    # non-fatal — don't fail, just warn

            sample_result["total_tokens"] = sum(token_counts)
            sample_result["avg_tokens"] = round(_avg(token_counts), 1)
            sample_result["avg_quality"] = round(_avg(quality_scores), 3)
            sample_result["min_quality"] = round(min(quality_scores), 3) if quality_scores else None
            sample_result["max_quality"] = round(max(quality_scores), 3) if quality_scores else None

        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            sample_result["elapsed_ms"] = round(elapsed, 1)
            sample_result["error"] = f"{type(exc).__name__}: {exc}"
            result["errors"].append(f"[{sample_name}] {sample_result['error']}")
            result["passed"] = False
            traceback.print_exc()

        if sample_result["missing_keys"]:
            result["errors"].append(
                f"[{sample_name}] missing keys: {sample_result['missing_keys']}"
            )
            result["passed"] = False

        result["samples"][sample_name] = sample_result

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Pretty Printer
# ═══════════════════════════════════════════════════════════════════════════════

def print_strategy_result(r: Dict[str, Any]) -> None:
    status = "PASS ✓" if r["passed"] else "FAIL ✗"
    _print_header(f"Strategy: {r['strategy']}  [{status}]")

    for sample_name, sr in r["samples"].items():
        _print_subheader(f"Sample: {sample_name}")
        if sr["error"]:
            print(f"    ERROR: {sr['error']}")
            continue
        print(f"    Chunks : {sr['chunk_count']}")
        print(f"    Tokens : total={sr['total_tokens']}  avg={sr['avg_tokens']}")
        print(
            f"    Quality: avg={sr['avg_quality']}  "
            f"min={sr['min_quality']}  max={sr['max_quality']}"
        )
        print(f"    Time   : {sr['elapsed_ms']} ms")

    if r["errors"]:
        print("\n  ⚠ Issues:")
        for e in r["errors"]:
            print(f"    • {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Summary Table
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results: List[Dict[str, Any]]) -> None:
    _print_header("SUMMARY")

    hdr = f"  {'Strategy':<22} {'Status':<8} {'Chunks':<8} {'AvgTok':<8} {'AvgQ':<8} {'Time(ms)':<10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        total_chunks = sum(sr["chunk_count"] for sr in r["samples"].values())
        avg_tok_all = _avg(
            [sr["avg_tokens"] for sr in r["samples"].values() if sr["avg_tokens"] > 0]
        )
        avg_q_all = _avg(
            [sr["avg_quality"] for sr in r["samples"].values() if sr["avg_quality"] > 0]
        )
        total_time = sum(sr["elapsed_ms"] for sr in r["samples"].values())
        print(
            f"  {r['strategy']:<22} {status:<8} {total_chunks:<8} "
            f"{avg_tok_all:<8.1f} {avg_q_all:<8.3f} {total_time:<10.1f}"
        )

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n  Result: {passed}/{total} strategies passed")
    if passed == total:
        print("  All chunking strategies are working — ready to move to Retrieval phase!")
    else:
        failed = [r["strategy"] for r in results if not r["passed"]]
        print(f"  Fix needed: {', '.join(failed)}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

CANONICAL_STRATEGIES = [
    "semantic",
    "overlap",
    "rust",
    "smart_check",
    "structure_aware",
    "recursive_overlap",
    "elite",
    "elite_v2",
]


async def main() -> int:
    # Allow filtering from CLI args
    requested = [a.lower() for a in sys.argv[1:]] or CANONICAL_STRATEGIES

    all_strategies = set(list_chunking_strategies())
    selected = []
    for name in requested:
        if name not in all_strategies:
            print(f"WARNING: '{name}' is not a registered strategy — skipping")
        else:
            selected.append(name)

    if not selected:
        print("No valid strategies to test.")
        return 1

    print(f"\nTesting {len(selected)} chunking strategies: {', '.join(selected)}")
    print(f"Against {len(SAMPLES)} sample documents: {', '.join(SAMPLES.keys())}")

    results: List[Dict[str, Any]] = []
    for strategy in selected:
        r = await test_single_strategy(strategy)
        print_strategy_result(r)
        results.append(r)

    print_summary(results)

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
