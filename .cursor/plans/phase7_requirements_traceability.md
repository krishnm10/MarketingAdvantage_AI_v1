# Phase 7 — Requirements-to-tests traceability

This matrix maps tenant-sensitive behaviors from the implementation plan to **automated tests** and **manual / operational** evidence. Update this file when you add endpoints or tests.

## Automated (CI — minimal job)

GitHub Actions workflow `tenant-isolation-tests.yml` runs only `tests/test_tenant_validator_phase7.py` with **pytest alone** (no full `requirements.txt`) so PRs get a fast tenant-boundary signal.

## Automated (full venv — run locally or in release pipeline)

| Requirement ID | Behavior | Test artifact | Notes |
| ---------------- | -------- | ------------- | ----- |
| T-ISO-VEC-01 | Cross-tenant retrieval / vector metadata isolation (mock DB) | `tests/tenant_isolation/test_cross_tenant_isolation.py` | 68+ cases; needs project deps |
| T-STO-01 | Stable storage UUID from slug helper | `tests/test_tenant_storage_uuid.py` | Complements validator UUID tests |
| T-VAL-01 | Valid tenant slug is normalized (case, trim) | `tests/test_tenant_validator_phase7.py::TestValidateTenantId::test_normalize_lowercase_and_strip` | |
| T-VAL-02 | Missing/empty tenant rejected when `allow_default=False` | `TestValidateTenantId::test_rejects_empty_when_no_default` | |
| T-VAL-03 | Optional default tenant for legacy `allow_default=True` | `TestValidateTenantId::test_allow_default_maps_to_default` | |
| T-VAL-04 | Path traversal, reserved names, wildcards rejected | `TestValidateTenantId::test_rejects_illegal_values` | Parametrized |
| T-UUID-01 | Storage UUID is deterministic per slug | `TestStorageUuidDerivation::test_deterministic_per_slug` | uuid5 namespace |
| T-UUID-02 | Different slugs → different storage UUIDs | `TestStorageUuidDerivation::test_different_slugs_different_uuids` | Isolation |
| T-UUID-03 | UUID matches `MAI_TENANT_NAMESPACE` + `mai:tenant:{slug}` | `TestStorageUuidDerivation::test_namespace_and_string_form` | |
| T-MODE-01 | `TENANT_ENFORCEMENT_MODE=strict` rejects missing tenant | `TestValidateTenantIdStrictEnforcementMode::test_strict_missing_raises` | |
| T-MODE-02 | `TENANT_ENFORCEMENT_MODE=off` allows default fallback | `TestValidateTenantIdStrictEnforcementMode::test_off_missing_defaults_without_raise` | |
| T-MODE-03 | Invalid env value falls back to `warn` | `TestValidateTenantIdStrictEnforcementMode::test_warn_invalid_env_falls_back_to_warn` | |
| T-MODE-04 | Valid body under strict validates normally | `TestValidateTenantIdStrictEnforcementMode::test_provided_id_validates_same_as_validate_tenant_id` | |
| T-ACL-01 | Admin role can access any tenant | `TestEnforceTenantAccess::test_admin_always_allowed` | |
| T-ACL-02 | No `allowed_tenants` → backward-compatible allow | `TestEnforceTenantAccess::test_no_allowed_tenants_list_allows_compat` | |
| T-ACL-03 | User allowed list must include tenant | `TestEnforceTenantAccess::test_allowed_list_match` / `test_allowed_list_mismatch_denies` | |

## Manual / integration (two-tenant script)

Execute the walkthrough in `phase7_phase8_cursor_plan.md` § “Phase 7 manual validation script”:

1. Tenant A: ingestion, retrieval, chat, settings, dashboard — capture UI tenant badge and request IDs.
2. Tenant B: confirm Tenant A data absent everywhere.
3. Cross-tenant URL / ID / payload paste → denial, no leakage.
4. Reverse order + concurrent tabs.
5. After cache warm, refresh, simulated partial API failure — tenant remains correct.

**Evidence:** screenshots, HAR, or structured logs with request ID + tenant slug.

## Phase 8 — API inventory

| Requirement ID | Behavior | Artifact |
| ---------------- | -------- | -------- |
| P8-INV-01 | Enumerate all FastAPI routes with methods | `scripts/phase8_api_route_inventory.py` |
| P8-INV-02 | Flag paths needing tenant review | Same script (`tenant_heuristic` column) |

## Frontend (recommended next automation)

Not yet in CI (requires Playwright/Cypress or Vitest + Next mocks):

- TenantContext init order: URL → `localStorage` → default.
- Navbar switch updates API calls (ingestion, retrieve, chat).
- Deep link `?tenant=` on ingestion detail.

Track in backlog or add a dedicated workflow when the stack is chosen.

## Definition of done (excerpt)

- [x] Core validator, UUID, enforcement modes, and access helper covered by automated tests.
- [ ] Full matrix + two-tenant manual sign-off (operator).
- [ ] Phase 8 inventory reviewed per release for new routes.
