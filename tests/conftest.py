"""
Shared pytest configuration for Phase 7 tenant-isolation and safety tests.

Keeps a non-placeholder JWT for any test module that eventually imports app.auth.
Minimal CI jobs install only pytest; they do not need the full requirements.txt.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _phase7_test_environment() -> None:
    # Must not match blocked placeholder secrets in app.auth.generate_token
    os.environ.setdefault(
        "JWT_SECRET_KEY",
        "pytest_phase7_only_" + ("k" * 48),
    )
    os.environ.setdefault("ENVIRONMENT", "test")
