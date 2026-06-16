# Security — JWT storage (admin UI)

## Decision (Phase 5)

**Default recommendation:** migrate JWT from `localStorage` to **httpOnly cookies** when the admin UI is reachable beyond localhost or when SSO/OIDC is planned.

## Current compensating controls (if deferring migration)

- Admin UI intended for **internal-network-only** deployments
- Chat output sanitized via `rehype-sanitize` on the retrieve/chat page
- No third-party scripts in the admin bundle (verify before each release)
- Backend APIs require `Authorization: Bearer` on all sensitive routes (`require_role`)

## Accepted risk documentation

If httpOnly cookies are deferred, record here:

| Field | Value |
|-------|-------|
| Decision date | |
| Owner | |
| Network scope | internal-only / staging / public |
| Review date | |

## LLM provider naming

- **`grok`** (request-time override or display alias) routes to **xAI** (`https://api.x.ai/v1`), not Groq inference.
- Use **`groq`** for Groq LPU cloud inference (e.g. `llama-3.1-8b-instant`).
- Tenant JSON stores `llm.single.type` as an enum value (`xai`, `groq`, etc.); `grok` is not a persisted config type.

## Related controls implemented

- `AUTH_USERS` required in non-dev backend startup
- Frontend middleware redirects unauthenticated users on protected routes
- Default JWT role claim falls back to `viewer`, not `admin`
