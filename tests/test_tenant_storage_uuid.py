import uuid

from app.utils.tenant_storage_uuid import storage_business_uuid_for_tenant


def test_slug_produces_stable_uuid():
    u1 = storage_business_uuid_for_tenant("acme_corp", None)
    u2 = storage_business_uuid_for_tenant("acme_corp", None)
    assert isinstance(u1, uuid.UUID)
    assert u1 == u2


def test_literal_uuid_preserved():
    u = uuid.uuid4()
    out = storage_business_uuid_for_tenant("anything", str(u))
    assert out == u
