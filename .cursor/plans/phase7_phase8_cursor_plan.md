# Phase 7 and Ongoing Phase 8 Plan for Cursor

## Context
The implementation work is already complete for the tenant-sensitive backend and frontend paths, and the remaining work is primarily validation, hardening, and operational verification across Phase 7 and the ongoing Phase 8 effort.[cite:conversation_history] This plan is therefore structured as a world-class testing, audit, and release-readiness program rather than a new coding roadmap.[cite:conversation_history][cite:1][cite:8]

## Objectives
The main objective for Phase 7 is to prove tenant correctness, security, and regression safety across all affected backend and frontend flows through a complete test matrix, strong negative testing, and traceable quality gates.[cite:1][cite:2][cite:3][cite:8] The main objective for Phase 8 is to continuously audit the full API surface, validate production-like observability, and detect drift, shadow behavior, or cross-tenant leakage in real environments before it affects customers.[cite:7][cite:10][cite:13][cite:15]

## Delivery principles
The testing program should follow a risk-based strategy with explicit scope, traceability, production-like environments, and layered validation across unit, integration, system, security, and UAT levels.[cite:1][cite:5][cite:8] Multi-tenant isolation must be treated as both a security requirement and a functional requirement, so every positive path must have matching cross-tenant negative paths, symmetric role checks, and tenant-boundary assertions.[cite:6][cite:9][cite:12][cite:15]

## Exit criteria
Phase 7 should only be considered complete when all tenant-sensitive screens and APIs pass the approved matrix for two or more real tenants, all critical and high defects are closed, and regression suites are green in CI on repeated runs.[cite:1][cite:5][cite:8] Phase 8 remains ongoing and should be treated as an operational audit loop with periodic endpoint review, log/alert verification, and continuous evidence that tenant context is enforced and observable end to end.[cite:7][cite:10][cite:13][cite:15]

## Phase 7 scope
The Phase 7 test scope should cover ingestion public and admin APIs, retrieval, query and chat flows, health checks, dedup paths, retrieval repositories and runtime behavior, pipeline cache behavior, tenant validation, and all frontend screens that read or propagate the active tenant context.[cite:conversation_history] The UI scope should explicitly include the global tenant selector and context, dashboard, ingestion list, ingestion detail, ingestion upload, retrieval console, chat, settings, pipeline builder, reranking, and multi-customer dashboards.[cite:conversation_history]

## Phase 7 workstreams

### 1. Requirements traceability
Create a requirements-to-tests matrix that maps each tenant-sensitive endpoint, repository filter, frontend screen, and state transition to positive, negative, authorization, observability, and regression test cases.[cite:1][cite:3][cite:8] Each item should include expected tenant inputs, expected storage UUID filtering behavior, expected denial behavior, and linked evidence artifacts from manual or automated execution.[cite:1][cite:8]

### 2. Environment and data setup
Prepare at least two real tenants plus one synthetic adversarial tenant, with isolated users, roles, datasets, documents, ingestion history, retrieval artifacts, and dashboard data to support both functional and hostile test paths.[cite:9][cite:15] The test environment should mirror production configuration as closely as possible, including auth, tenant validator wiring, caching behavior, background jobs, logging, and monitoring integrations.[cite:1][cite:5][cite:11]

### 3. Backend testing
Execute a deep backend matrix across unit, integration, contract, and end-to-end API layers.[cite:1][cite:8]

- Positive tests: authenticated tenant requests return only that tenant’s resources, writes stay within the authenticated tenant, and storage UUID filters are always applied.[cite:9][cite:15]
- Negative tests: missing client_id or business_id, mismatched tenant identity, forged identifiers, stale or invalid tenant context, and direct object reference attempts must fail with the expected denial semantics and no data exposure.[cite:6][cite:9][cite:10][cite:15]
- Symmetric isolation tests: Tenant A must not read or mutate Tenant B data, and Tenant B must not read or mutate Tenant A data, even when users hold equivalent roles or permissions.[cite:9][cite:12]
- Background path tests: dedup jobs, retrieval runtime, cache helpers, and repository calls must preserve tenant scoping under retries, parallel execution, and cache hits or misses.[cite:9][cite:15]
- Contract tests: every documented request and response shape should be validated for tenant-bearing fields, auth expectations, and failure payload consistency.[cite:13]

### 4. Frontend testing
Validate that the active tenant selected in the global context is consistently reflected in all UI reads, mutations, navigation transitions, deep links, and reloads.[cite:conversation_history] The test matrix should cover tenant switching mid-session, stale tabs, back-button navigation, refresh behavior, concurrent windows, empty states, partial failures, and unauthorized or cross-tenant links pasted directly into the browser.[cite:2][cite:3][cite:9]

Recommended frontend coverage:

- Component tests for TenantContext, Navbar selector, and hooks that compose tenant-aware API calls.[cite:conversation_history]
- Screen tests for dashboard, ingestion list and detail, upload, retrieval console, chat, settings, pipeline builder, reranking, and multi-customer dashboards under at least two tenants.[cite:conversation_history]
- UX validation that tenant identity is visible enough to prevent operator error, especially before upload, deletion, reranking, or settings changes.[cite:2][cite:3]
- Negative UI tests where APIs deny access or return empty tenant-scoped datasets, ensuring no misleading mixed-tenant state is shown.[cite:9][cite:15]

### 5. Security testing
Run explicit multi-tenant authorization and object access testing focused on tenant isolation, broken access control, and unsafe direct object access patterns.[cite:6][cite:9][cite:10][cite:15] Every sensitive endpoint should be challenged with replay attempts, modified path parameters, modified query parameters, modified request bodies, and cross-user or cross-tenant token combinations.[cite:6][cite:9][cite:12]

Security checks should include:

- BOLA and IDOR-style attempts against ingestion resources, retrieval resources, chats, dashboards, settings, and pipeline artifacts.[cite:6][cite:10]
- Cross-tenant replay of valid requests using another tenant’s credentials or browser state.[cite:9][cite:12]
- Validation that logs and audit trails capture allow and deny decisions with tenant context, request identifiers, and actor identity without leaking secrets or raw tokens.[cite:7][cite:10][cite:15]

### 6. Performance and resilience
Add tenant-aware load and concurrency scenarios to confirm that tenant filters hold under parallel execution, retries, cache contention, burst traffic, and large ingestion or retrieval workloads.[cite:1][cite:5][cite:10] Include soak tests for long-running retrieval and chat sessions, and confirm that no cache pollution or response bleed occurs across tenants during sustained traffic.[cite:9][cite:15]

### 7. Automation and CI
Promote the most valuable matrix cases into CI so that tenant isolation, tenant propagation, and regression safety are continuously enforced.[cite:2][cite:5][cite:10] At minimum, CI should block merges on unit, integration, contract, API regression, UI smoke, cross-tenant negative, and logging assertion suites for tenant-sensitive areas.[cite:1][cite:5][cite:8]

## Phase 7 test matrix

| Area | Positive coverage | Negative coverage | Evidence |
|---|---|---|---|
| Ingestion APIs | Correct tenant create/read/update/list behavior.[cite:conversation_history][cite:9] | Missing or forged tenant identifiers, cross-tenant access, invalid role.[cite:6][cite:9] | API logs, automated suite, response captures.[cite:7][cite:10] |
| Retrieval/query/chat APIs | Only tenant-owned results and chats returned.[cite:conversation_history][cite:9] | Cross-tenant prompts, resource IDs, history access, replay attempts.[cite:6][cite:9][cite:12] | API tests, audit logs, trace IDs.[cite:7][cite:15] |
| Dedup/runtime/cache | Tenant-safe filtering and cache behavior.[cite:conversation_history] | Cache pollution, stale tenant context, retry crossover.[cite:9][cite:15] | Parallel test logs, metrics, traces.[cite:10][cite:15] |
| Frontend screens | Correct tenant selector propagation and rendering.[cite:conversation_history] | Deep-link leakage, stale tab, mid-session switch, unauthorized view.[cite:2][cite:3][cite:9] | Playwright/Cypress runs, screenshots, HAR files.[cite:2][cite:5] |
| Logging/monitoring | Tenant context in logs and alerts.[cite:7][cite:15] | Missing tenant tag, cross-tenant alert blind spot, secret leakage in logs.[cite:7][cite:10][cite:15] | Structured logs, alert screenshots, SIEM traces.[cite:7][cite:10] |

## Phase 7 manual validation script
Use a strict two-tenant walkthrough for every high-risk flow because real operator behavior often exposes defects that automation misses.[cite:2][cite:11] The following manual script should be executed in Cursor and documented with screenshots, request IDs, and timestamps for auditability.[cite:7][cite:10]

1. Log in as Tenant A, complete ingestion, retrieval, chat, settings, and dashboard actions, and capture the tenant identifier visible in the UI and logs.[cite:conversation_history][cite:7]
2. Switch to Tenant B and confirm that Tenant A data, counts, artifacts, and history are fully absent across all affected screens and APIs.[cite:9][cite:15]
3. Paste direct URLs, resource IDs, or copied request payloads from Tenant A into a Tenant B session and confirm denial with no partial leakage.[cite:6][cite:9][cite:12]
4. Repeat the same tests in reverse order and then under concurrent browser tabs for both tenants.[cite:9][cite:12]
5. Re-run after cache warm-up, browser refresh, and partial API failure simulation to verify state recovery and tenant correctness.[cite:5][cite:9]

## Defect severity model
Tenant leakage, cross-tenant mutation, or missing tenant validation should be treated as Sev-0 or Sev-1 defects and should block release immediately.[cite:6][cite:9][cite:15] Logging gaps, inconsistent denial payloads, and stale-UI tenant confusion should be treated as high severity when they reduce containment or operator safety in production.[cite:7][cite:10][cite:15]

## Ongoing Phase 8 scope
Phase 8 should focus on API surface audit, observability hardening, runtime drift detection, and production-like verification rather than new feature implementation.[cite:conversation_history][cite:10][cite:13] The goal is to continuously prove that every reachable endpoint, job, and integration still honors tenant boundaries as the platform evolves.[cite:7][cite:10][cite:15]

## Phase 8 workstreams

### 1. API inventory and surface audit
Build and maintain a living inventory of all active endpoints, including public APIs, admin APIs, internal service calls, background jobs, health endpoints, and any undocumented or legacy paths.[cite:7][cite:10][cite:13] For each item, record auth model, tenant source of truth, required tenant fields, allowed roles, data classification, expected denial modes, and monitoring coverage.[cite:7][cite:13]

### 2. Drift and shadow endpoint detection
Compare code routes, gateway routes, deployed routes, OpenAPI documentation, and runtime logs to detect undocumented, shadow, or zombie endpoints that bypass the expected tenant model.[cite:7][cite:10][cite:13] Any endpoint missing explicit tenant behavior or audit coverage should be treated as an audit finding until proven safe.[cite:7][cite:10]

### 3. Runtime observability
Structured logs should include request ID, actor identity, tenant identity, endpoint, authorization decision, and high-level outcome so that any cross-tenant attempt can be traced quickly.[cite:7][cite:10][cite:15] Alerts should be added for cross-tenant denial spikes, missing tenant metadata, unexpected null tenant values, repeated authorization failures, and abnormal access patterns by endpoint or actor.[cite:10][cite:15]

### 4. Continuous security review
Schedule recurring multi-user and multi-tenant abuse tests against production-like environments, especially after auth, caching, routing, repository, or UI context changes.[cite:6][cite:9][cite:12] This should include symmetric and asymmetric role patterns, token replay attempts, object reference fuzzing, and tenant-switch flows in browser sessions.[cite:9][cite:12]

### 5. Production validation cadence
Run a lightweight but disciplined audit cadence: daily smoke checks for critical tenant flows, weekly API inventory drift review, sprint-end full tenant isolation regression, and monthly observability and alert review.[cite:5][cite:10][cite:13] Every release should include a signed checkpoint that confirms no tenant-sensitive endpoint or screen shipped without updated tests and monitoring evidence.[cite:1][cite:7]

## Phase 8 checklist

| Audit item | Expected standard |
|---|---|
| Endpoint inventory | Every reachable endpoint is cataloged with auth and tenant requirements.[cite:7][cite:13] |
| Route drift | No undocumented or shadow tenant-sensitive paths remain in deployment.[cite:7][cite:10] |
| Logs | Every allow or deny decision carries tenant context and request ID.[cite:7][cite:15] |
| Alerts | Cross-tenant attempts and missing tenant metadata trigger alerts.[cite:10][cite:15] |
| Regression | Critical tenant tests run on every release path.[cite:2][cite:5] |
| Evidence | Screenshots, traces, logs, and automated reports are stored for audit and rollback decisions.[cite:7][cite:10] |

## Recommended automation stack for Cursor
Cursor should be used to drive a disciplined workflow that combines code-assisted test authoring, repeatable automation, and reviewable evidence capture.[cite:conversation_history] A strong practical stack is Playwright for browser and multi-tab flows, pytest for backend and contract tests, schema and OpenAPI validation for API conformance, and centralized logging dashboards for runtime verification.[cite:1][cite:5][cite:10]

Suggested test layers:

- Pytest unit and integration suites for tenant validator, repository filters, dedup logic, runtime scoping, and cache helpers.[cite:conversation_history]
- API tests with tenant-aware fixtures for positive and hostile paths.[cite:6][cite:9]
- Playwright end-to-end suites for tenant selector propagation and cross-screen isolation.[cite:conversation_history]
- Security regression scripts for BOLA, IDOR, replay, and cross-tenant object access attempts.[cite:6][cite:10]
- Log assertions and dashboard checks to validate observability and alerting behaviors.[cite:7][cite:15]

## Suggested delivery sequence

### Week 1
Freeze the Phase 7 scope, finalize the requirements-to-tests traceability matrix, prepare the multi-tenant test data, and baseline the current automated suites.[cite:1][cite:5][cite:8]

### Week 2
Execute the backend and frontend tenant matrix, automate the highest-risk negative paths, and triage defects with immediate fixes for any isolation or authorization issue.[cite:6][cite:9][cite:12]

### Week 3
Run performance, concurrency, cache, and observability validation, then complete the two-real-tenant manual walkthrough and capture sign-off evidence.[cite:7][cite:10][cite:15]

### Week 4 and ongoing
Move into Phase 8 operational cadence with API inventory review, route drift detection, release-gate enforcement, and recurring real-environment tenant isolation checks.[cite:7][cite:10][cite:13]

## Definition of done
The work is done when tenant-sensitive behavior is not only implemented but also continuously provable through automated tests, manual adversarial validation, production-like telemetry, and release gates that prevent regression.[cite:1][cite:5][cite:10] In practice, this means no cross-tenant leakage, no unaudited endpoint drift, complete tenant context in logs, and a repeatable evidence trail for every release affecting tenant-aware flows.[cite:7][cite:10][cite:15]

---

## Repo implementation pointers (Phase 7 / 8 automation in this codebase)

The following artifacts support the plan without changing production behavior:

| Artifact | Purpose |
| -------- | ------- |
| [phase7_requirements_traceability.md](phase7_requirements_traceability.md) | Requirements-to-tests matrix and manual evidence checklist |
| `tests/test_tenant_validator_phase7.py` | Fast CI: validator, UUID derivation, enforcement modes, `enforce_tenant_access` |
| `tests/conftest.py` + `pytest.ini` | Phase 7 pytest defaults (`pythonpath`, safe JWT for future imports) |
| `tests/tenant_isolation/test_cross_tenant_isolation.py` | Existing vector mock isolation suite — run with full venv: `pytest tests/tenant_isolation/ -v` |
| `scripts/phase8_api_route_inventory.py` | Phase 8 route inventory (TSV/Markdown); requires full app deps |
| `.github/workflows/tenant-isolation-tests.yml` | CI gate: minimal pytest install + `test_tenant_validator_phase7` only |

