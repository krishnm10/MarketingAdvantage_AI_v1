# Chunking: Fallback Tokenizer UX + Dry Run Fix

**Scope:** Small, UI-first pass. No ingestion behavior changes unless explicitly noted as Phase 2.

**Problem**

1. **Tokenizer UI** — "Default tokenizer backend" implies it is the primary chunking tokenizer. With `use_model_native_tokenizer_for_chunking: true` (default), Gemini/OpenAI native tokenizers are primary; BERT/HF is fallback only. Worse: if native injection fails at ingest, the backend **silently** falls back to the factory tokenizer (log-only). Customers see successful ingestion but chunk boundaries may not match the embedder → retrieval quality drops with no visible signal.

2. **Dry run** — `TestSplitterPanel` pre-fills placeholder text as `useState` default, so it always shows 1 chunk on load. It uses client-side word-split + `chars/4` approximation; strategy label is display-only. Users think production chunking is broken.

---

## Goals

| # | Goal | Success criteria |
|---|------|------------------|
| G1 | Correct tokenizer labeling | Chunking block explains native vs fallback clearly |
| G2 | Surface alignment risk | Pipeline shows embedder/tokenizer binding status for active tenant |
| G3 | Fix dry-run UX | Empty default, no auto-preview on placeholder, honest "approximate only" copy |
| G4 | Optional explicit run | User clicks "Preview" to run dry run (not on every keystroke by default) |

**Non-goals (this pass):** New backend chunk-preview API, `fail_on_native_tokenizer_error` schema flag, ingestion job metadata stamps.

---

## Phase 1 — UI only (~2–3 files)

### 1A. Relabel tokenizer section (`TenantProcessingSettingsBlock.tsx`)

**Changes**

- Rename **"Default tokenizer backend"** → **"Fallback tokenizer backend"**
- Subtitle: *Used only when embedder-native chunking is unavailable (bundle missing, native injection failed, or native mode off).*
- Rename **"HuggingFace tokenizer model"** → **"Fallback HuggingFace model"**
- Dim/disable fallback fields when `use_model_native_tokenizer_for_chunking === true` (still editable when native off)
- Add amber info callout when native is on:

  > Production ingestion uses your **embedder's native tokenizer** (tiktoken, SentencePiece, etc.). Fallback settings below apply only if native chunking cannot run. If that happens, ingestion currently continues silently — check server logs or the alignment status below.

- Checkbox label: **"Use embedder-native tokenizer for chunking"** (keep JSON key unchanged)

**Files:** `app/frontend-admin/app/settings/pipeline/components/TenantProcessingSettingsBlock.tsx`

---

### 1B. Tokenizer alignment status card (new small component)

**New:** `TokenizerAlignmentCard.tsx` in `components/`

- On mount (chunking block + wizard step 2): `GET /api/v2/embedding-alignment?client_id={clientId}` (route exists in `lib/apiRoutes.ts`)
- Show compact badge from `component_checks` where `component === "tokenizer"`:
  - `ok` → green: "Native {family} bound"
  - `warning` / `error` → amber/red with `message` + `detail`
- Link to Dashboard → Health or embedding-alignment docs for ops
- Loading / error states (don't block rest of form)

**Wire into:**

- `page.tsx` — chunking wizard step (~line 1821) and chunking block view (~line 1440), above `TestSplitterPanel`

**Files:** new component + `page.tsx` (2 import sites)

---

### 1C. Fix TestSplitterPanel dry run (`TestSplitterPanel.tsx`)

**Changes**

| Issue | Fix |
|-------|-----|
| Placeholder treated as input | `useState("")` — empty default |
| Auto 1-chunk on load | Only compute chunks when `sample.trim().length > 0` **and** user has clicked **Preview chunks** |
| Strategy misleading | Banner: *"Approximate preview only — does not call backend. Strategy `{strategy}` is not simulated. Production uses ingestion chunking engine + embedder tokenizer."* |
| Short text confusion | Show hint when `estimateTokens(sample) < chunkSize * 0.1`: *"Text shorter than one chunk at size {chunkSize} — expect 1 chunk."* |
| Token estimate | Keep `~` prefix; add tooltip/note: *"Latin-script estimate (~4 chars/token). Not embedder-native."* |

**Optional props** (if `page.tsx` has embedder context handy):

```ts
embedderModel?: string;
useNativeTokenizer?: boolean;
```

Show in header: *"Embedder: {model} · counting: approximate (not native)"*

**Files:** `TestSplitterPanel.tsx`, minor prop pass in `page.tsx`

---

## Phase 2 — Backend observability (defer unless requested)

Small schema + ingest flag; **not in Phase 1**.

| Item | Description |
|------|-------------|
| `fail_on_native_tokenizer_error` | Tenant JSON `tokenization` — fail job instead of silent fallback |
| Ingest metadata | `chunking_tokenizer_mode: native \| factory \| degraded` on job result |
| Alignment API | Distinguish *configured* native vs *last run* actual (needs ingest audit store) |
| Chunk preview API | `POST /api/v2/ingestion/chunk-preview` — real backend split for dry run |

---

## Implementation order

```
1. TestSplitterPanel fixes          (immediate user-visible dry-run fix)
2. TenantProcessingSettingsBlock    (relabel + callout + conditional dim)
3. TokenizerAlignmentCard           (wire embedding-alignment into chunking views)
4. Smoke: tsc + manual chunking block
```

---

## Test plan

### Dry run

- [ ] Open Pipeline → Chunking block: textarea empty, no chunk list
- [ ] Paste 2–3 sentences (< chunk_size): click Preview → 1 chunk, hint visible
- [ ] Paste long doc (5k+ chars): Preview → multiple chunks
- [ ] Change strategy semantic → recursive: chunk boundaries unchanged (proves strategy not applied)
- [ ] Wizard step 2: same behavior

### Tokenizer UX

- [ ] Native checkbox on: fallback fields visually secondary; callout visible
- [ ] Native checkbox off: fallback fields fully enabled
- [ ] Alignment card loads for tenant with embedder in catalog → green native binding
- [ ] Tenant without catalog entry → error/warning on alignment card
- [ ] Apply Configuration still PATCHes same `tokenization` keys (no schema change)

### Regression

- [ ] `npx tsc --noEmit` in `frontend-admin`
- [ ] No backend test changes required for Phase 1

---

## Effort estimate

| Phase | Effort |
|-------|--------|
| Phase 1 (UI) | ~2–4 hours |
| Phase 2 (backend) | ~1–2 days |

---

## Open question for implementer

Pass `clientId` + selected embedder model from `page.tsx` into `TestSplitterPanel` / `TokenizerAlignmentCard` so alignment reflects the **draft** embedder before Apply, or only **saved** tenant config? Recommendation: use saved config via `client_id` API (simpler); note in UI if draft embedder differs from saved.
