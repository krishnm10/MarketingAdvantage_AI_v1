---
name: RAG Chat Hallucination Fix
overview: "FINAL gap-resolved master plan: INV-1101 hallucination fix, L0 router extensions, prompt hardening, faithfulness verifier, UI controls. Pre-read complete; deviations locked. Execute P0 → P2 → P3 → P1 → P4 → P5."
todos:
  - id: preread-complete
    content: Mandatory 12-file pre-read findings written and user-confirmed
    status: completed
  - id: phase0-config
    content: app/core/configs/vaidyanad.json — preset-rag-context, enable_hyde false, top_k_retrieval/final (reranker.top_k unchanged)
    status: completed
  - id: phase2-core
    content: file_id on RetrievalCandidate+RankedResult (runtime.py); STRUCTURED embed/filter/chunk cap; library_loader stripping; ssot wiring; rewrite_allowed=False
    status: completed
  - id: phase3-verifier
    content: app/services/faithfulness_verifier.py + wire after raw_answer + L5 trace
    status: completed
  - id: phase1-api
    content: Extend prompt-templates + tenants prompt-config; ChatRetrieveRequest fields
    status: completed
  - id: phase4-ui
    content: Chat prompt panel, auto-rewrite toggle, Prompt Library Set active
    status: completed
  - id: phase5-verify
    content: pytest V1-V3 + E2E V4-V15
    status: completed
isProject: false
---

# RAG Chat Hallucination Fix and L0 Router — FINAL Master Plan

**Version:** FINAL (gap-resolved, self-consistent)  
**Scope:** Chat path only (`/api/v2/retrieve/chat`). Do not touch `retrieve_api.py` or `rag_api.py`.  
**Pre-read:** Complete. Deviations approved and locked.  
**Execution order:** Phase 0 → Phase 2 → Phase 3 → Phase 1 → Phase 4 → Phase 5

---

## Incident summary

Query `INV-1101` returned hallucinated figures (tax $412.50, total $4,537.50, routing 026015079) from **few-shot Example C** in InvoiceIQ templates plus **5 chunks from 5 invoices** blended into context. No post-generation verifier ran.

**Three compounding failures:** (1) prompt leakage, (2) multi-chunk cross-invoice blend, (3) no L5 faithfulness check.

---

## Approved deviations (locked — canonical)

| # | Rule |
|---|------|
| 1 | `file_id`: add to `RetrievalCandidate` first; propagate at every **live** `RankedResult(...)` site in [runtime.py](app/retrieval/runtime.py) L146, L167–173. **Do NOT** patch [repository.py](app/retrieval/repository.py) for `RankedResult`. Also patch [retrieval_orchestrator.py](app/retrieval/retrieval_orchestrator.py) L125, L140 if on live path; mark dead paths explicitly. |
| 2 | Paths: tenant JSON → `app/core/configs/`; loader → `app/core/prompts/library_loader.py`; template JSON → `app/core/configs/prompts/`. Never `app/config/` or `app/prompts/`. |
| 3 | L0 router **already at** [retrieve_chat_api.py](app/api/v2/retrieve_chat_api.py) **L682** — **do not rebuild**. Extend STRUCTURED only: `embed_text=matched_pattern`, `max_context_chunks=2`, strict keys from `matched_pattern`, verifier after `raw_answer`. |
| 4 | APIs: extend `/api/v2/prompt-templates` and `/api/v2/rag-config/pipeline-pluggable`. Optional thin `GET/PATCH /api/v2/tenants/{client_id}/prompt-config`. **No** `/api/v2/prompts/templates` tree. |
| 5 | HyDE: UI sets `enable_hyde` from tenant runtime ([chat/page.tsx](app/frontend-admin/app/dashboard/retrieve/chat/page.tsx) L143). Phase 0 `enable_hyde: false` fixes config + UI after reload. |
| 6 | Phase 0 gate: **negative assertions only** (fabricated values absent). Full V5 needs Phase 2+3. Do not block on `chunks_sent <= 2` until Phase 2. |
| 7 | `preset-rag-context` only; `matched_pattern` not `focus_entity`; `rewrite_allowed=False`; fail-closed C1/C2/C4/C5; global strip in `library_loader`; 20 hard constraints apply. |

---

## Non-negotiable architecture decisions (D1–D12)

- **D1:** Safe template = `preset-rag-context` only (not `querysystem.json` — same Example C as CoT).
- **D2:** Use `RouteDecision.matched_pattern`; no `focus_entity`.
- **D3:** `RankedResult.file_id` optional, **last** field; `frozen=True` — never mutate after construction.
- **D4:** STRUCTURED: `rewrite_allowed=False` + `embed_text=matched_pattern` before embed.
- **D5:** STRUCTURED: `effective_max_context_chunks=2` in code (not config alone).
- **D6:** Strip examples in `library_loader.py` at runtime; no `few_shots_enabled` JSON flag.
- **D7:** `prompt_profile` functional via [ssot.py](app/core/prompts/ssot.py) `resolve_prompt_ssot(config)` — extend to accept override template id if needed; strip **after** ssot output.
- **D8:** `enable_hyde=false` in tenant config; verify UI does not force true.
- **D9:** Verifier: C1/C2/C4/C5 fail-closed; C3 warning-only.
- **D10–D12:** Tenant-level template id; session `system_prompt_override` text only; prompt locked per request; strip all templates before LLM.

---

## Mismatch corrections (M1–M9)

| ID | Action |
|----|--------|
| M1 | `preset-rag-context` only |
| M2 | `structured_decision`: `rewrite_allowed=False`; force `embed_text=matched_pattern` |
| M3 | No `focus_entity` |
| M4 | `file_id` on `RetrievalCandidate` + all live `RankedResult` sites in `runtime.py` (and orchestrator if live) |
| M5 | Runtime strip only — no JSON flags |
| M6 | Wire ssot + strip; do not bypass ssot |
| M7 | Cap chunks in code for STRUCTURED |
| M8 | Config + UI HyDE alignment |
| M9 | Add `retrieval.top_k_retrieval: 3`, `top_k_final: 1`; leave `reranker.top_k: 5` |

---

## Pre-read findings (completed — do not re-read unless code drift)

### Backend anchors

| Item | Finding |
|------|---------|
| ChatRetrieveRequest | `session_id`, `messages`, `client_id`, `llm_provider`, `llm_model`, `reranker`, `intent`, `top_k`, `search_mode`, `similarity_threshold`, `enable_hyde`, `max_context_chunks`, `generate_answer` |
| embed_text | L752 assign; HyDE may set L774; embed L800 |
| context slice | L1030 `results[:req.max_context_chunks]` |
| strict keys | L754 |
| raw LLM | `raw_answer` L1209 |
| L5 trace | L1293 `focus_fallback` |
| HyDE guard | L758 `if _hyde_enabled and req.enable_hyde` |
| L0 router | **L682** already present |
| RankedResult | `chunk_id`, `text`, `score`, `explanation`, `trust_decision`; frozen; **no file_id** |
| RankedResult sites | `runtime.py` L146, L167–173 (**chat live path**); `retrieval_orchestrator.py` L125, L140 (verify live) |
| structured rewrite | `rewrite_allowed=True` today (types.py L174) |
| matched_pattern | Yes; set `id_m.group(0)` rule_router L166 |
| Templates dir | `app/core/configs/prompts/` (loader already uses `parent.parent / configs / prompts`) |
| Safe template | `preset-rag-context` — no dollar examples |
| Dangerous | `preset-chain-of-thought`, `querysystem` — Example C 4,537.50 / 026015079 |
| ssot | `resolve_prompt_ssot(config)` in [ssot.py](app/core/prompts/ssot.py); selector = `config.retrieval.prompt_template_id` |
| faithfulness_verifier | **NOT FOUND** — create `app/services/faithfulness_verifier.py` |
| vaidyanad (pre Phase 0) | `prompt_template_id`: preset-chain-of-thought; `enable_hyde`: true; no top_k_retrieval/final in file |

### Frontend anchors

| Item | Finding |
|------|---------|
| Chat POST body | `session_id`, `messages`, `client_id`, `llm_provider`, `reranker`, `intent`, `top_k`, `search_mode`, `enable_hyde`, `generate_answer` |
| enable_hyde UI | Sent L291; defaulted from runtime L143 |
| rewrite toggle | **No** |
| Prompt Library | Edit, Delete, Refresh, New Template; **no** Set active; **no** tenant prompt fetch |
| APIs | `/api/v2/prompt-templates` exists; no `/tenants/.../prompt-config` yet; `rag-config/pipeline-pluggable` has `prompt_template_id` |

### Already implemented (extend only)

- L0 router + chitchat/meta/structured/blocked rules in [rule_router.py](app/services/query_routing/rule_router.py)
- Early return when `retrieval_allowed=False`
- Tests: [tests/test_query_routing.py](tests/test_query_routing.py), [tests/test_chat_routing_integration.py](tests/test_chat_routing_integration.py)

---

## Phase 0 — Config only

**File:** [app/core/configs/vaidyanad.json](app/core/configs/vaidyanad.json) — `retrieval` section only.

```diff
-    "enable_hyde": true,
+    "enable_hyde": false,
     ...
-    "prompt_template_id": "preset-chain-of-thought"
+    "top_k_retrieval": 3,
+    "top_k_final": 1,
+    "prompt_template_id": "preset-rag-context"
```

**Do not change** `reranker.top_k` (stays 5).

### Phase 0 gate (checkpoint — negative only)

POST `/api/v2/retrieve/chat` with `INV-1101`, `client_id=vaidyanad`, `top_k=1`.

**Assert ABSENT from answer:** `412.50`, `4,537.50`, `4537.50`, `026015079`, `Corporate Legal Services`, `Corporate Legal`.

If fabricated values remain → config reload or template resolution broken; stop before Phase 2.

**Do not require:** `chunks_sent <= 2`, positive tax/total/routing until Phase 2–3.

---

## Phase 2 — L0 extensions + core fixes (after Phase 0 gate)

### 2.0 — Extend L0 (do not rebuild)

At L682 router already runs. Add per-route behavior and L1 trace: `prompt_locked`, `prompt_template_id`, `prompt_source`, `retrieval_skipped`.

| Route | Retrieval | Notes |
|-------|-----------|-------|
| CHITCHAT / META_HELP / CLARIFICATION / BLOCKED | Skipped | No embed/retrieve/rerank |
| STRUCTURED | Narrow | See 2.2–2.5 |
| KNOWLEDGE | Full RAG | No file filter |

### 2.1 — file_id plumbing

1. [types_retrieve.py](app/retrieval/types_retrieve.py): `RetrievalCandidate.file_id: Optional[str] = None` (last field).
2. [repository.py](app/retrieval/repository.py): set `file_id=str(content.file_id)` on `RetrievalCandidate` in `_build_candidate` (~L673).
3. [runtime.py](app/retrieval/runtime.py): `file_id=candidate.file_id` on both `RankedResult` constructions (L146, L167).
4. [retrieval_orchestrator.py](app/retrieval/retrieval_orchestrator.py): same if live; else `# DEAD PATH` comment.

### 2.2 — STRUCTURED rewrite + embed

- [types.py](app/services/query_routing/types.py): `rewrite_allowed=False` in `structured_decision()`.
- [retrieve_chat_api.py](app/api/v2/retrieve_chat_api.py) before embed:
  - `req.rewrite_enabled is False` → `embed_text = raw_query`
  - `STRUCTURED` + `matched_pattern` → `embed_text = matched_pattern`
  - `req.rewrite_enabled is True` + multi-turn → existing rewrite
  - **Guard:** if STRUCTURED and `matched_pattern is None` → log warning; `embed_text = raw_query`

### 2.3 — Strict grounding keys

If keys empty and route STRUCTURED and `matched_pattern`: `_strict_detail_grounding_keys = [matched_pattern]`.

### 2.4 — Dominant file filter + chunk cap

Add `_filter_to_dominant_file(chunks, focus_id, max_chunks=2)` using `RankedResult.file_id` and `RankedResult.text` (not `.content` — fix pseudocode from master prompt).

STRUCTURED only, after rerank, before context assembly:

- `effective_max = 2`
- Replace `results[:req.max_context_chunks]` with filtered `top_chunks`

### 2.5 — Amount conflict warning

STRUCTURED only: if disjoint dollar sets across chunks, prepend warning to context string.

### 2.6 — Example stripping

[library_loader.py](app/core/prompts/library_loader.py):

- Confirm `PROMPTS_DIR` → `app/core/configs/prompts` (already correct via Path).
- Add `assert os.path.isdir(PROMPTS_DIR)` at import if not present.
- Add `strip_examples_from_instructions(text)` (+ optional `strip_and_verify` with instruction-preservation fallback per master prompt).
- Test against `preset-chain-of-thought` content before wiring.

### 2.7 — prompt_profile + ssot

At request start: `_active_template_id = tenant_config.retrieval.prompt_template_id`.

```python
_ps = resolve_prompt_ssot(_cfg_chat)
system_instructions = strip_examples_from_instructions(_ps.instructions)
```

If `req.system_prompt_override`: replace instruction text only; template id unchanged for trace.

Extend ssot only if needed to accept explicit template override kwarg — prefer tenant config as today.

### 2.8 — Prompt lock trace (L1)

`prompt_locked: true`, `prompt_template_id`, `prompt_source: tenant_config | session_override_text`.

---

## Phase 3 — Faithfulness verifier

**Create:** [app/services/faithfulness_verifier.py](app/services/faithfulness_verifier.py)

- Deterministic regex; no LLM; STRUCTURED only.
- `verify_or_refuse(answer, source_texts, focus_id=matched_pattern, route=...)`
- Fail-closed C1/C2/C4/C5; C3 warning-only.
- Wire after `raw_answer` (~L1209); add L5 event `faithfulness_verifier` (keep existing `focus_fallback` event).

Use `chunk.text` for source texts (not `.content`).

---

## Phase 1 — API extensions

| Endpoint | Action |
|----------|--------|
| `GET /api/v2/prompt-templates` | Add `has_examples` per template (dollar regex) |
| `GET /api/v2/prompt-templates/{id}` | Return **stripped** `system_instructions` + `has_examples` |
| `GET/PATCH /api/v2/tenants/{client_id}/prompt-config` | Thin wrapper; delegate to existing config persistence / rag-config |
| `ChatRetrieveRequest` | Add `rewrite_enabled: Optional[bool]`, `system_prompt_override: Optional[str]`; **no** `prompt_template_id` in body |

---

## Phase 4 — UI

**Chat** ([chat/page.tsx](app/frontend-admin/app/dashboard/retrieve/chat/page.tsx)):

- Collapsible Prompt panel (stripped preview, session override, admin save default).
- Auto-rewrite toggle (default OFF) → `rewrite_enabled` in body.
- Stop sending `enable_hyde` unless user explicitly enables (or rely on tenant false after Phase 0).

**Prompt Library** ([prompt-builder/page.tsx](app/frontend-admin/app/settings/prompt-builder/page.tsx)):

- Add "Set as active for tenant" → PATCH prompt-config.
- Card badge for templates with examples.
- Do not duplicate in Pipeline Builder.

---

## Phase 5 — Tests and verification

| File | Cases |
|------|-------|
| `tests/test_faithfulness_verifier.py` | 5 cases (fail/pass/skipped/warning/frozen) |
| `tests/test_example_stripping.py` | CoT stripped; rag-context preserved |
| Extend `tests/test_query_routing.py` | Hi, Thanks, meta, invoice, INV-1101, knowledge, blocked |

**Run order:** V1 pytest routing → V2 verifier → V3 stripping → V4 Phase 0 negative gate → V5–V15 E2E/trace/UI.

---

## Hard constraints (20)

1. No `retrieve_api.py` / `rag_api.py` changes this pass  
2. No delete examples from JSON — runtime strip only  
3. No `few_shots_enabled` flag  
4. Do not modify `reranker.top_k`  
5. Do not mutate `RankedResult` after construction  
6. No `focus_entity`  
7. No persist `system_prompt_override`  
8. No dominant file filter on KNOWLEDGE  
9. No LLM in verifier  
10. No inline imports in async handlers  
11. `prompt_profile` must be functional (via ssot + trace)  
12. Fail-closed C1/C2/C4/C5  
13. Pass `prompt_profile` at RouteDecision construction if frozen later  
14. No `prompt_template_id` in chat request body  
15. Prompt identity locked per request  
16. Global strip before LLM  
17. CHITCHAT/META/CLARIFICATION/BLOCKED: no rewrite/embed/retrieve/rerank  
18. No invoice context for chitchat  
19. Invoice prompt for KNOWLEDGE/STRUCTURED only  
20. Match existing auth, trace, response patterns  

---

## Definition of done

See master prompt checklist: Config, Routing, Backend, API, UI, Tests — all items must pass V1–V15.

---

## Execution style

1. Pre-read: **done** — proceed without re-reading unless drift suspected.  
2. **Phase 0:** Apply `vaidyanad.json` diff only → run negative gate → stop.  
3. **Phase 2 → 3 → 1 → 4 → 5** in order.  
4. Document any deviation from this plan before coding.  
5. Prefer **Agent mode** to edit non-markdown files (Plan mode blocks JSON/Python).

---

## Repo-specific notes (gap resolution)

- **Section 2.1:** Use Option B — patch `runtime.py` (+ `RetrievalCandidate` in repository), not repository `RankedResult`.  
- **Gap 2 removed:** Pre-read lists all `RankedResult` sites; no separate gap section needed.  
- **Master prompt path typos:** Use `app/core/configs/vaidyanad.json` and `app/core/prompts/library_loader.py` everywhere.  
- **Filter helper:** Use `r.text` and `r.file_id` on `RankedResult`, not `.content`.  
- **L0 “no pre-retrieval router” in incident text:** Outdated — router exists; task is **extend** and fix downstream grounding.

*End of FINAL master plan.*
