from __future__ import annotations

from fastapi import Request
from fastapi.testclient import TestClient

from app.main import app


@app.get("/__obs_middleware_check")
async def _obs_middleware_check(req: Request):
    return {
        "has_request_id": bool(getattr(req.state, "request_id", None)),
        "seen_by_tracing": bool(
            getattr(req.state, "_observability_tracing_saw_request_id", False)
        ),
    }


def test_metrics_endpoint_exposes_prometheus_text() -> None:
    with TestClient(app, raise_server_exceptions=True) as client:
        # Hit a simple health endpoint to ensure at least one request is recorded.
        r_health = client.get("/health")
        assert r_health.status_code == 200
        body = r_health.json()
        assert isinstance(body, dict)
        assert "status" in body

        r = client.get("/metrics")
        assert r.status_code == 200
        content_type = r.headers.get("content-type", "")
        assert "text/plain" in content_type.lower()
        text = r.text
        # At minimum, our custom HTTP metric family should be present.
        assert "mai_request_total" in text


def test_metrics_endpoint_trailing_slash() -> None:
    """
    Ensure /metrics/ works reasonably (200 or redirect) so operators
    don't accidentally get a broken scrape path.
    """
    with TestClient(app, raise_server_exceptions=True) as client:
        r = client.get("/metrics/")
        assert r.status_code in (200, 301, 302, 307, 308)


def test_tracing_middleware_sees_request_id() -> None:
    """
    Ensure tracing middleware runs after RequestIDMiddleware and can see the
    populated request_id on the same Request object.
    """
    with TestClient(app, raise_server_exceptions=True) as client:
        rid = "test-request-id-123"
        r = client.get("/__obs_middleware_check", headers={"X-Request-ID": rid})
        assert r.status_code == 200
        data = r.json()
        assert data["has_request_id"] is True
        assert data["seen_by_tracing"] is True

