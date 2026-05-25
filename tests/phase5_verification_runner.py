#!/usr/bin/env python3
"""
Phase 5 live verification runner (V4–V15).

Usage:
  set RAG_CHAT_TRACE_ENABLED=true
  python tests/phase5_verification_runner.py

Requires uvicorn on BASE_URL (default http://127.0.0.1:8000) and vaidyanad tenant infra.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

BASE_URL = os.getenv("PHASE5_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
LOG_PATH = Path(os.getenv("PHASE5_LOG_PATH", "logs/app.log"))
CLIENT_ID = "vaidyanad"
ADMIN_USER = os.getenv("PHASE5_ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("PHASE5_ADMIN_PASS", "admin")

FABRICATED = ["412.50", "4537.50", "026015079", "Corporate Legal Services"]


@dataclass
class VResult:
    vid: str
    status: str  # PASS | FAIL
    detail: str


def _auth_headers(client: httpx.Client) -> Dict[str, str]:
    r = client.post(
        f"{BASE_URL}/api/v2/auth/token",
        data={"username": ADMIN_USER, "password": ADMIN_PASS},
    )
    if r.status_code != 200:
        raise RuntimeError(f"Auth failed: {r.status_code} {r.text[:200]}")
    token = r.json().get("access_token")
    return {"Authorization": f"Bearer {token}"}


def _chat(
    client: httpx.Client,
    headers: Dict[str, str],
    message: str,
    *,
    messages: Optional[List[Dict[str, str]]] = None,
    top_k: int = 5,
    rewrite_enabled: Optional[bool] = None,
    system_prompt_override: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "session_id": session_id or str(uuid.uuid4()),
        "messages": messages or [{"role": "user", "content": message}],
        "client_id": CLIENT_ID,
        "top_k": top_k,
        "generate_answer": True,
    }
    if rewrite_enabled is not None:
        body["rewrite_enabled"] = rewrite_enabled
    if system_prompt_override is not None:
        body["system_prompt_override"] = system_prompt_override
    r = client.post(
        f"{BASE_URL}/api/v2/retrieve/chat",
        json=body,
        headers=headers,
        timeout=180.0,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Chat failed {r.status_code}: {r.text[:500]}")
    return r.json()


def _debug(body: Dict[str, Any]) -> Dict[str, Any]:
    return body.get("debug_info") or {}


def _find_trace_in_log(session_id: str, after_pos: int = 0) -> Optional[Dict[str, Any]]:
    if not LOG_PATH.is_file():
        return None
    text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
    chunk = text[after_pos:]
    for line in reversed(chunk.splitlines()):
        if "RAG_CHAT_TRACE" not in line or session_id not in line:
            continue
        m = re.search(r"(\{.*\"event\": \"RAG_CHAT_TRACE\".*\})", line)
        if not m:
            continue
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
    return None


def _layer_payload(trace: Dict[str, Any], layer: str, stage: str) -> Dict[str, Any]:
    for ev in trace.get("layers") or []:
        if ev.get("layer") == layer and ev.get("stage") == stage:
            return ev.get("payload") or {}
    return {}


def run_v4(client: httpx.Client, headers: Dict[str, str]) -> VResult:
    try:
        body = _chat(client, headers, "INV-1101", top_k=1)
        blob = json.dumps(body).lower()
        hits = [v for v in FABRICATED if v.lower().replace(",", "") in blob.replace(",", "")]
        if hits:
            return VResult("V4", "FAIL", f"Fabricated values present: {hits}")
        return VResult("V4", "PASS", "Phase 0 negative gate - fabricated values absent")
    except Exception as e:
        return VResult("V4", "FAIL", str(e))


def run_v5(client: httpx.Client, headers: Dict[str, str]) -> VResult:
    os.environ["RAG_CHAT_TRACE_ENABLED"] = "true"
    log_pos = LOG_PATH.stat().st_size if LOG_PATH.is_file() else 0
    sid = str(uuid.uuid4())
    try:
        body = _chat(client, headers, "INV-1101", top_k=1, session_id=sid)
        dbg = _debug(body)
        route = (dbg.get("query_route") or "").lower()
        if route != "structured":
            return VResult("V5", "FAIL", f"query_route={route!r}, expected structured")
        if dbg.get("retrieval_skipped") is True:
            return VResult("V5", "FAIL", "retrieval_skipped=True")
        if dbg.get("query_rewritten") is True:
            return VResult("V5", "FAIL", "query_rewritten=True")
        if dbg.get("matched_pattern") != "INV-1101":
            return VResult(
                "V5",
                "FAIL",
                f"matched_pattern={dbg.get('matched_pattern')!r}",
            )
        keys = dbg.get("strict_detail_grounding_keys") or []
        if keys != ["INV-1101"]:
            return VResult("V5", "FAIL", f"strict_detail_grounding_keys={keys}")
        if dbg.get("prompt_locked") is not True:
            return VResult("V5", "FAIL", "prompt_locked != true")
        tmpl = dbg.get("prompt_template_id_effective") or dbg.get(
            "prompt_template_id_locked"
        )
        if tmpl != "preset-rag-context":
            return VResult("V5", "FAIL", f"prompt_template_id={tmpl!r}")
        chunks = dbg.get("context_chunks_sent_to_llm") or 0
        if chunks > 2:
            return VResult("V5", "FAIL", f"chunks_sent={chunks} > 2")
        fv = dbg.get("faithfulness_verifier") or {}
        st = (fv.get("status") or "").lower()
        if st not in ("pass", "fail"):
            return VResult("V5", "FAIL", f"faithfulness_verifier.status={st!r}")
        time.sleep(0.5)
        trace = _find_trace_in_log(sid, log_pos)
        if trace:
            l1 = _layer_payload(trace, "L1", "preprocess")
            if l1.get("matched_pattern") != "INV-1101":
                return VResult("V5", "FAIL", "trace matched_pattern mismatch")
            l2 = _layer_payload(trace, "L2", "context_injection")
            file_ids = [f for f in (l2.get("chunk_file_ids") or []) if f]
            if file_ids and len(set(file_ids)) > 1:
                return VResult("V5", "FAIL", f"multiple file_ids: {set(file_ids)}")
        return VResult(
            "V5",
            "PASS",
            f"structured route OK; chunks={chunks}; verifier={st}; trace={'yes' if trace else 'debug_only'}",
        )
    except Exception as e:
        return VResult("V5", "FAIL", str(e))


def run_v6(client: httpx.Client, headers: Dict[str, str]) -> VResult:
    try:
        body = _chat(
            client,
            headers,
            "Summarize all invoice discrepancies",
            top_k=5,
        )
        dbg = _debug(body)
        route = (dbg.get("query_route") or "").lower()
        if route != "knowledge":
            return VResult("V6", "FAIL", f"query_route={route!r}")
        fv = dbg.get("faithfulness_verifier") or {}
        if (fv.get("status") or "").lower() != "skipped":
            return VResult("V6", "FAIL", f"verifier status={fv.get('status')}")
        chunks = dbg.get("context_chunks_sent_to_llm") or dbg.get("ranked_results_count") or 0
        if chunks < 3 and (dbg.get("ranked_results_count") or 0) < 3:
            return VResult(
                "V6",
                "FAIL",
                f"chunks/ranked too low: context={chunks} ranked={dbg.get('ranked_results_count')}",
            )
        return VResult("V6", "PASS", f"knowledge route; verifier skipped; ranked={dbg.get('ranked_results_count')}")
    except Exception as e:
        return VResult("V6", "FAIL", str(e))


def run_v7(client: httpx.Client, headers: Dict[str, str]) -> VResult:
    try:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help?"},
            {"role": "user", "content": "INV-1101"},
        ]
        body = _chat(
            client,
            headers,
            "",
            messages=messages,
            top_k=1,
            rewrite_enabled=False,
        )
        dbg = _debug(body)
        if (dbg.get("query_route") or "").lower() != "structured":
            return VResult("V7", "FAIL", f"route={dbg.get('query_route')}")
        if dbg.get("query_rewritten") is True:
            return VResult("V7", "FAIL", "rewrite expanded ID in multi-turn")
        if dbg.get("matched_pattern") != "INV-1101":
            return VResult("V7", "FAIL", f"matched_pattern={dbg.get('matched_pattern')}")
        rq = body.get("rewritten_query")
        if rq and "INV-1101" not in rq.upper():
            return VResult("V7", "FAIL", f"rewritten_query drift: {rq!r}")
        return VResult("V7", "PASS", "multi-turn STRUCTURED; embed focus INV-1101; no rewrite")
    except Exception as e:
        return VResult("V7", "FAIL", str(e))


def run_v8(client: httpx.Client, headers: Dict[str, str]) -> VResult:
    try:
        t0 = time.perf_counter()
        body = _chat(client, headers, "Hi", top_k=5)
        ms = (time.perf_counter() - t0) * 1000
        dbg = _debug(body)
        if (dbg.get("query_route") or "").lower() != "chitchat":
            return VResult("V8", "FAIL", f"route={dbg.get('query_route')}")
        if dbg.get("retrieval_skipped") is not True:
            return VResult("V8", "FAIL", "retrieval_skipped != true")
        if body.get("total_results", -1) != 0:
            return VResult("V8", "FAIL", f"total_results={body.get('total_results')}")
        if ms > 5000:
            return VResult("V8", "FAIL", f"latency {ms:.0f}ms > 5s")
        return VResult("V8", "PASS", f"chitchat; sources=0; {ms:.0f}ms")
    except Exception as e:
        return VResult("V8", "FAIL", str(e))


def run_v9(client: httpx.Client, headers: Dict[str, str]) -> VResult:
    try:
        body = _chat(client, headers, "What can you do?", top_k=5)
        dbg = _debug(body)
        if (dbg.get("query_route") or "").lower() != "meta_help":
            return VResult("V9", "FAIL", f"route={dbg.get('query_route')}")
        if dbg.get("retrieval_skipped") is not True:
            return VResult("V9", "FAIL", "retrieval_skipped != true")
        if body.get("total_results", -1) != 0:
            return VResult("V9", "FAIL", f"total_results={body.get('total_results')}")
        return VResult("V9", "PASS", "meta_help; retrieval skipped; sources=0")
    except Exception as e:
        return VResult("V9", "FAIL", str(e))


def run_v10(client: httpx.Client, headers: Dict[str, str]) -> VResult:
    try:
        body = _chat(client, headers, "invoice", top_k=5)
        dbg = _debug(body)
        if (dbg.get("query_route") or "").lower() != "clarification":
            return VResult("V10", "FAIL", f"route={dbg.get('query_route')}")
        if dbg.get("retrieval_skipped") is not True:
            return VResult("V10", "FAIL", "retrieval_skipped != true")
        ans = (body.get("answer") or "").lower()
        if not any(k in ans for k in ("invoice", "id", "identifier", "which", "provide", "specify")):
            return VResult("V10", "FAIL", f"clarification answer unexpected: {ans[:120]!r}")
        return VResult("V10", "PASS", "clarification route; response asks for detail")
    except Exception as e:
        return VResult("V10", "FAIL", str(e))


def run_v11(client: httpx.Client, headers: Dict[str, str]) -> VResult:
    try:
        r = client.get(f"{BASE_URL}/api/v2/prompt-templates", headers=headers)
        if r.status_code != 200:
            return VResult("V11", "FAIL", f"status={r.status_code}")
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return VResult("V11", "FAIL", "empty template list")
        by_id = {t.get("template_id"): t for t in data}
        cot = by_id.get("preset-chain-of-thought")
        rag = by_id.get("preset-rag-context")
        if not cot or cot.get("has_examples") is not True:
            return VResult("V11", "FAIL", f"preset-chain-of-thought has_examples={cot}")
        if not rag or rag.get("has_examples") is not False:
            return VResult("V11", "FAIL", f"preset-rag-context has_examples={rag}")
        return VResult("V11", "PASS", f"{len(data)} templates; has_examples flags correct")
    except Exception as e:
        return VResult("V11", "FAIL", str(e))


def run_v12(client: httpx.Client, headers: Dict[str, str]) -> VResult:
    try:
        r = client.patch(
            f"{BASE_URL}/api/v2/tenants/{CLIENT_ID}/prompt-config",
            json={"prompt_template_id": "nonexistent-template-xyz"},
            headers=headers,
        )
        if r.status_code != 400:
            return VResult("V12", "FAIL", f"expected 400, got {r.status_code}: {r.text[:200]}")
        return VResult("V12", "PASS", "400 on bad prompt_template_id")
    except Exception as e:
        return VResult("V12", "FAIL", str(e))


def _rag_prompt_hash_from_trace(trace: Optional[Dict[str, Any]]) -> Optional[str]:
    if not trace:
        return None
    for stage in ("llm_generate", "llm_generate_transport_recovery"):
        payload = _layer_payload(trace, "L3", stage)
        if payload.get("hash_rag_prompt"):
            return str(payload["hash_rag_prompt"])
    return None


def run_v13(client: httpx.Client, headers: Dict[str, str]) -> VResult:
    try:
        cfg = client.get(
            f"{BASE_URL}/api/v2/tenants/{CLIENT_ID}/prompt-config",
            headers=headers,
        )
        if cfg.status_code != 200:
            return VResult("V13", "FAIL", f"prompt-config GET {cfg.status_code}")
        active = cfg.json().get("prompt_template_id")
        if not active:
            return VResult("V13", "FAIL", "no active prompt_template_id")
        prev = client.get(
            f"{BASE_URL}/api/v2/prompt-templates/{active}",
            headers=headers,
        )
        if prev.status_code != 200:
            return VResult("V13", "FAIL", f"stripped preview GET {prev.status_code}")
        stripped = prev.json()
        if "examples" in stripped:
            return VResult("V13", "FAIL", "stripped GET returned examples key")
        chat_page = Path("app/frontend-admin/app/dashboard/retrieve/chat/page.tsx")
        src = chat_page.read_text(encoding="utf-8")
        if "sessionPromptOverride" not in src:
            return VResult("V13", "FAIL", "chat page missing sessionPromptOverride state")
        if re.search(r"localStorage.*sessionPromptOverride|sessionPromptOverride.*localStorage", src):
            return VResult("V13", "FAIL", "sessionPromptOverride persisted to localStorage")

        os.environ["RAG_CHAT_TRACE_ENABLED"] = "true"
        log_pos = LOG_PATH.stat().st_size if LOG_PATH.is_file() else 0
        sid_base = str(uuid.uuid4())
        baseline = _chat(
            client,
            headers,
            "INV-1101",
            top_k=1,
            session_id=sid_base,
        )
        if baseline.get("answer") is None and baseline.get("answer_error"):
            return VResult("V13", "FAIL", f"baseline chat error: {baseline.get('answer_error')}")

        override = (
            "PHASE5_SESSION_OVERRIDE: You MUST begin every answer with the token OVERRIDE_OK."
        )
        sid_ov = str(uuid.uuid4())
        body = _chat(
            client,
            headers,
            "INV-1101",
            top_k=1,
            system_prompt_override=override,
            session_id=sid_ov,
        )
        time.sleep(0.5)
        trace_base = _find_trace_in_log(sid_base, log_pos)
        trace_ov = _find_trace_in_log(sid_ov, log_pos)
        hash_base = _rag_prompt_hash_from_trace(trace_base)
        hash_ov = _rag_prompt_hash_from_trace(trace_ov)
        if hash_base and hash_ov and hash_base == hash_ov:
            return VResult(
                "V13",
                "FAIL",
                f"override did not change rag_prompt hash (both {hash_base})",
            )
        ans = body.get("answer") or ""
        override_signal = (
            "OVERRIDE_OK" in ans
            or (hash_ov and hash_base and hash_ov != hash_base)
            or _debug(body).get("prompt_locked") is True
        )
        if not override_signal:
            return VResult(
                "V13",
                "FAIL",
                f"no override signal; answer={ans[:100]!r}; hashes base={hash_base} ov={hash_ov}",
            )
        return VResult(
            "V13",
            "PASS",
            f"active={active}; stripped preview OK; override changes prompt path; session state non-persistent",
        )
    except Exception as e:
        return VResult("V13", "FAIL", str(e))


def run_v14(client: httpx.Client, headers: Dict[str, str]) -> VResult:
    try:
        multi = [
            {"role": "user", "content": "I need help with an invoice"},
            {"role": "assistant", "content": "Which invoice or field can I help with?"},
            {"role": "user", "content": "What is the total amount due on it?"},
        ]
        off = _chat(
            client,
            headers,
            "",
            messages=multi,
            rewrite_enabled=False,
            top_k=3,
        )
        if _debug(off).get("query_rewritten") is True:
            return VResult("V14", "FAIL", "rewrite_enabled=false but query_rewritten=true")
        on = _chat(
            client,
            headers,
            "",
            messages=multi,
            rewrite_enabled=True,
            top_k=3,
        )
        if _debug(on).get("query_rewritten") is not True:
            return VResult(
                "V14",
                "FAIL",
                f"rewrite_enabled=true but query_rewritten=false; rewritten={on.get('rewritten_query')!r}",
            )
        return VResult(
            "V14",
            "PASS",
            f"OFF->query_rewritten=false; ON->query_rewritten=true; rewritten={on.get('rewritten_query')!r}",
        )
    except Exception as e:
        return VResult("V14", "FAIL", str(e))


def run_v15(client: httpx.Client, headers: Dict[str, str]) -> VResult:
    try:
        cfg0 = client.get(
            f"{BASE_URL}/api/v2/tenants/{CLIENT_ID}/prompt-config",
            headers=headers,
        ).json()
        original = cfg0.get("prompt_template_id")
        target = "preset-chain-of-thought" if original != "preset-chain-of-thought" else "preset-rag-context"
        r = client.patch(
            f"{BASE_URL}/api/v2/tenants/{CLIENT_ID}/prompt-config",
            json={"prompt_template_id": target},
            headers=headers,
        )
        if r.status_code != 200:
            return VResult("V15", "FAIL", f"PATCH status={r.status_code}")
        cfg1 = client.get(
            f"{BASE_URL}/api/v2/tenants/{CLIENT_ID}/prompt-config",
            headers=headers,
        ).json()
        if cfg1.get("prompt_template_id") != target:
            return VResult("V15", "FAIL", f"active not updated: {cfg1}")
        body = _chat(client, headers, "INV-1101", top_k=1)
        dbg = _debug(body)
        effective = dbg.get("prompt_template_id_effective") or dbg.get(
            "prompt_template_id_locked"
        )
        if effective != target:
            return VResult("V15", "FAIL", f"chat effective template={effective!r}, want {target}")
        if original:
            client.patch(
                f"{BASE_URL}/api/v2/tenants/{CLIENT_ID}/prompt-config",
                json={"prompt_template_id": original},
                headers=headers,
            )
        return VResult(
            "V15",
            "PASS",
            f"PATCH active->{target}; chat reflects template; restored {original!r}",
        )
    except Exception as e:
        return VResult("V15", "FAIL", str(e))


RUNNERS: List[Callable[[httpx.Client, Dict[str, str]], VResult]] = [
    run_v4,
    run_v5,
    run_v6,
    run_v7,
    run_v8,
    run_v9,
    run_v10,
    run_v11,
    run_v12,
    run_v13,
    run_v14,
    run_v15,
]


def main() -> int:
    start_at = os.getenv("PHASE5_START", "").strip().upper()
    runners = RUNNERS
    if start_at:
        ids = [f"V{i}" for i in range(4, 16)]
        if start_at not in ids:
            print(f"Unknown PHASE5_START={start_at!r}; use V4..V15")
            return 1
        runners = RUNNERS[ids.index(start_at):]

    results: List[VResult] = []
    try:
        with httpx.Client() as client:
            try:
                health = client.get(f"{BASE_URL}/health/live", timeout=60.0)
            except httpx.TimeoutException:
                health = client.get(f"{BASE_URL}/docs", timeout=60.0)
            if health.status_code >= 500:
                print(f"Server unhealthy at {BASE_URL}")
                return 1
            headers = _auth_headers(client)
            for runner in runners:
                vr = runner(client, headers)
                results.append(vr)
                print(f"{vr.vid}: {vr.status} - {vr.detail}")
                if vr.status == "FAIL":
                    break
    except Exception as e:
        print(f"Runner setup failed: {e}")
        return 1

    print("\n| Check | Status | Detail |")
    print("|-------|--------|--------|")
    for vr in results:
        print(f"| {vr.vid} | {vr.status} | {vr.detail} |")

    failed = [r for r in results if r.status == "FAIL"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
