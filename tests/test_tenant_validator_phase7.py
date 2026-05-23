"""
Phase 7 — tenant validator unit tests (no DB, no FastAPI app import).

Maps to plan workstream 3 (backend testing): positive/negative validation,
storage UUID determinism, enforcement modes, and access enforcement.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from app.utils.tenant_validator import (
    MAI_TENANT_NAMESPACE,
    TenantAccessDeniedError,
    TenantEnforcementMode,
    TenantValidationError,
    enforce_tenant_access,
    get_enforcement_mode,
    get_storage_uuid,
    get_storage_uuid_str,
    validate_tenant_id,
    validate_tenant_id_strict,
)


class TestValidateTenantId:
    def test_normalize_lowercase_and_strip(self) -> None:
        ctx = validate_tenant_id("  Acme_Corp  ", source="body", endpoint="t", allow_default=False)
        assert ctx.tenant_id == "acme_corp"

    def test_single_char_alphanumeric(self) -> None:
        assert validate_tenant_id("a", source="body", endpoint="t", allow_default=False).tenant_id == "a"

    def test_rejects_empty_when_no_default(self) -> None:
        with pytest.raises(TenantValidationError):
            validate_tenant_id("", source="body", endpoint="t", allow_default=False)
        with pytest.raises(TenantValidationError):
            validate_tenant_id(None, source="body", endpoint="t", allow_default=False)

    def test_allow_default_maps_to_default(self) -> None:
        ctx = validate_tenant_id(None, source="body", endpoint="t", allow_default=True)
        assert ctx.tenant_id == "default"

    @pytest.mark.parametrize(
        "bad",
        [
            "../x",
            "a/b",
            "a\\b",
            "__admin__",
            "admin",
            "*",
            "",
            "__all__",
        ],
    )
    def test_rejects_illegal_values(self, bad: str) -> None:
        with pytest.raises(TenantValidationError):
            validate_tenant_id(bad, source="body", endpoint="t", allow_default=False)


class TestStorageUuidDerivation:
    def test_deterministic_per_slug(self) -> None:
        ctx = validate_tenant_id("tenant_a", source="query", endpoint="t", allow_default=False)
        u1 = get_storage_uuid(ctx)
        u2 = get_storage_uuid(ctx)
        assert u1 == u2
        assert isinstance(u1, uuid.UUID)

    def test_different_slugs_different_uuids(self) -> None:
        a = get_storage_uuid(
            validate_tenant_id("tenant_a", source="body", endpoint="t", allow_default=False)
        )
        b = get_storage_uuid(
            validate_tenant_id("tenant_b", source="body", endpoint="t", allow_default=False)
        )
        assert a != b

    def test_namespace_and_string_form(self) -> None:
        ctx = validate_tenant_id("acme", source="body", endpoint="t", allow_default=False)
        expected = uuid.uuid5(MAI_TENANT_NAMESPACE, "mai:tenant:acme")
        assert get_storage_uuid(ctx) == expected
        assert get_storage_uuid_str(ctx) == str(expected)


class TestValidateTenantIdStrictEnforcementMode:
    def test_strict_missing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TENANT_ENFORCEMENT_MODE", "strict")
        assert get_enforcement_mode() == TenantEnforcementMode.STRICT
        with pytest.raises(TenantValidationError):
            validate_tenant_id_strict(None, source="body", endpoint="retrieve")

    def test_off_missing_defaults_without_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TENANT_ENFORCEMENT_MODE", "off")
        ctx = validate_tenant_id_strict(None, source="body", endpoint="retrieve")
        assert ctx.tenant_id == "default"

    def test_invalid_env_falls_back_to_strict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TENANT_ENFORCEMENT_MODE", "not-a-real-mode")
        assert get_enforcement_mode() == TenantEnforcementMode.STRICT

    def test_default_enforcement_mode_is_strict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TENANT_ENFORCEMENT_MODE", raising=False)
        assert get_enforcement_mode() == TenantEnforcementMode.STRICT

    def test_provided_id_validates_same_as_validate_tenant_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TENANT_ENFORCEMENT_MODE", "strict")
        ctx = validate_tenant_id_strict("sales_demo", source="body", endpoint="ingest")
        assert ctx.tenant_id == "sales_demo"


class TestEnforceTenantAccess:
    def test_admin_always_allowed(self) -> None:
        user = SimpleNamespace(sub="admin1", role="admin")
        ctx = validate_tenant_id("any_tenant", source="body", endpoint="t", allow_default=False)
        enforce_tenant_access(user, ctx, endpoint="t")

    def test_no_allowed_tenants_list_allows_compat(self) -> None:
        user = SimpleNamespace(sub="u1", role="viewer", allowed_tenants=None)
        ctx = validate_tenant_id("x", source="body", endpoint="t", allow_default=False)
        enforce_tenant_access(user, ctx, endpoint="t")

    def test_allowed_list_match(self) -> None:
        user = SimpleNamespace(sub="u1", role="viewer", allowed_tenants=["a", "b"])
        ctx = validate_tenant_id("b", source="body", endpoint="t", allow_default=False)
        enforce_tenant_access(user, ctx, endpoint="t")

    def test_allowed_list_mismatch_denies(self) -> None:
        user = SimpleNamespace(sub="u1", role="viewer", allowed_tenants=["tenant_b"])
        ctx = validate_tenant_id("tenant_a", source="body", endpoint="t", allow_default=False)
        with pytest.raises(TenantAccessDeniedError):
            enforce_tenant_access(user, ctx, endpoint="admin")


def test_enforcement_mode_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENANT_ENFORCEMENT_MODE", "warn")
    assert get_enforcement_mode() == TenantEnforcementMode.WARN
