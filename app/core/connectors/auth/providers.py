"""
Concrete auth providers — one class per authentication strategy.
File: app/core/connectors/auth/providers.py

All production secrets come from environment variables. NEVER hardcoded.

Layer-5 Fix: OAuth2Auth + AzureADAuth — threading.Lock() prevents
             double token fetch under concurrent requests.

Layer-6 Fix: FormBasedAuth — SESSION_TTL_SECONDS tracks session
             expiry and triggers auto re-login when session expires.
"""
from __future__ import annotations

import base64
import os
import time
import threading
from typing import Any, Dict, Optional

import httpx

from app.core.connectors.auth.base_auth import BaseAuthProvider


# ─────────────────────────────────────────────────────────────────────
# 1. NoAuth — public endpoints, no credentials needed
# Example: BBC RSS, public web pages, open REST APIs
# ─────────────────────────────────────────────────────────────────────
class NoAuth(BaseAuthProvider):
    """Public endpoints — no authentication needed."""

    def get_headers(self) -> Dict[str, str]:
        return {}

    def is_valid(self) -> bool:
        return True


# ─────────────────────────────────────────────────────────────────────
# 2. APIKeyAuth — single API key in a request header
# Example: NewsAPI ("X-Api-Key: abc123"), RapidAPI ("X-RapidAPI-Key: …")
#
# Usage:
#   auth = APIKeyAuth(header_name="X-Api-Key", api_key="abc123")
#   auth = APIKeyAuth.from_env("NEWSAPI_KEY", header_name="X-Api-Key")
# ─────────────────────────────────────────────────────────────────────
class APIKeyAuth(BaseAuthProvider):

    def __init__(self, header_name: str, api_key: str):
        if not api_key:
            raise ValueError("APIKeyAuth: api_key cannot be empty.")
        self._header_name = header_name
        self._api_key     = api_key

    @classmethod
    def from_env(cls, env_var: str,
                 header_name: str = "X-Api-Key") -> "APIKeyAuth":
        key = os.environ.get(env_var, "").strip()
        if not key:
            raise EnvironmentError(
                f"APIKeyAuth: environment variable '{env_var}' is not set."
            )
        return cls(header_name=header_name, api_key=key)

    def get_headers(self) -> Dict[str, str]:
        return {self._header_name: self._api_key}

    def is_valid(self) -> bool:
        return bool(self._api_key)


# ─────────────────────────────────────────────────────────────────────
# 3. BasicAuth — username + password, Base64 encoded
# Example: internal APIs, legacy systems, protected RSS feeds
#
# Usage:
#   auth = BasicAuth.from_env("INTERNAL_USER", "INTERNAL_PASS")
# ─────────────────────────────────────────────────────────────────────
class BasicAuth(BaseAuthProvider):

    def __init__(self, username: str, password: str):
        if not username or not password:
            raise ValueError("BasicAuth: username and password are required.")
        raw           = f"{username}:{password}".encode("utf-8")
        self._encoded = base64.b64encode(raw).decode("utf-8")

    @classmethod
    def from_env(cls, user_env: str, pass_env: str) -> "BasicAuth":
        user = os.environ.get(user_env, "").strip()
        pwd  = os.environ.get(pass_env, "").strip()
        if not user or not pwd:
            raise EnvironmentError(
                f"BasicAuth: env vars '{user_env}' or '{pass_env}' not set."
            )
        return cls(username=user, password=pwd)

    def get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Basic {self._encoded}"}

    def is_valid(self) -> bool:
        return bool(self._encoded)


# ─────────────────────────────────────────────────────────────────────
# 4. OAuth2Auth — Client Credentials flow with auto token refresh
# Example: Salesforce, HubSpot, Google APIs, any OAuth2 server
#
# Layer-5 Fix: threading.Lock() prevents race condition where two
# concurrent requests both see an expired token and both call
# _fetch_token() simultaneously — only one fetch happens, the
# second waits and reuses the result.
#
# Usage:
#   auth = OAuth2Auth(
#       token_url="https://login.salesforce.com/services/oauth2/token",
#       client_id=os.environ["SF_CLIENT_ID"],
#       client_secret=os.environ["SF_CLIENT_SECRET"],
#   )
# ─────────────────────────────────────────────────────────────────────
class OAuth2Auth(BaseAuthProvider):

    def __init__(self, token_url: str, client_id: str,
                 client_secret: str, scope: str = ""):
        self._token_url     = token_url
        self._client_id     = client_id
        self._client_secret = client_secret
        self._scope         = scope
        self._access_token: Optional[str] = None
        self._expires_at:   float         = 0.0
        # ✅ Layer-5: thread lock prevents concurrent double-fetch
        self._lock = threading.Lock()

    def _fetch_token(self) -> None:
        """Fetch a new token. MUST be called inside self._lock."""
        data: Dict[str, str] = {
            "grant_type":    "client_credentials",
            "client_id":     self._client_id,
            "client_secret": self._client_secret,
        }
        if self._scope:
            data["scope"] = self._scope

        resp = httpx.post(self._token_url, data=data, timeout=15)
        resp.raise_for_status()
        payload = resp.json()

        self._access_token = payload["access_token"]
        expires_in         = int(payload.get("expires_in", 3600))
        # 60-second safety margin — refresh before actual expiry
        self._expires_at   = time.time() + expires_in - 60

    def get_headers(self) -> Dict[str, str]:
        # ✅ Layer-5: acquire lock before checking + fetching
        # Only one thread fetches — others wait and reuse the result
        with self._lock:
            if not self._is_valid_unsafe():
                self._fetch_token()
        return {"Authorization": f"Bearer {self._access_token}"}

    def _is_valid_unsafe(self) -> bool:
        """Check validity WITHOUT acquiring lock — call only inside lock."""
        return bool(self._access_token) and time.time() < self._expires_at

    def is_valid(self) -> bool:
        """Thread-safe validity check for external callers."""
        with self._lock:
            return self._is_valid_unsafe()

    def refresh(self) -> None:
        with self._lock:
            self._fetch_token()


# ─────────────────────────────────────────────────────────────────────
# 5. AzureADAuth — Microsoft Azure Active Directory (MSAL-style)
# Example: SharePoint, Microsoft Graph API, Azure Blob Storage
#
# Layer-5 Fix: Same threading.Lock() pattern as OAuth2Auth —
# prevents concurrent double token fetch under async load.
#
# Usage:
#   auth = AzureADAuth(
#       tenant_id=os.environ["AZURE_TENANT_ID"],
#       client_id=os.environ["AZURE_CLIENT_ID"],
#       client_secret=os.environ["AZURE_CLIENT_SECRET"],
#       scope="https://graph.microsoft.com/.default",
#   )
# ─────────────────────────────────────────────────────────────────────
class AzureADAuth(BaseAuthProvider):

    def __init__(self, tenant_id: str, client_id: str,
                 client_secret: str, scope: str):
        self._tenant_id     = tenant_id
        self._client_id     = client_id
        self._client_secret = client_secret
        self._scope         = scope
        self._access_token: Optional[str] = None
        self._expires_at:   float         = 0.0
        # ✅ Layer-5: thread lock prevents concurrent double-fetch
        self._lock = threading.Lock()

    def _fetch_token(self) -> None:
        """Fetch a new token. MUST be called inside self._lock."""
        url = (
            f"https://login.microsoftonline.com/"
            f"{self._tenant_id}/oauth2/v2.0/token"
        )
        data = {
            "grant_type":    "client_credentials",
            "client_id":     self._client_id,
            "client_secret": self._client_secret,
            "scope":         self._scope,
        }
        resp = httpx.post(url, data=data, timeout=15)
        resp.raise_for_status()
        payload = resp.json()

        self._access_token = payload["access_token"]
        expires_in         = int(payload.get("expires_in", 3600))
        # 60-second safety margin
        self._expires_at   = time.time() + expires_in - 60

    def get_headers(self) -> Dict[str, str]:
        # ✅ Layer-5: acquire lock before checking + fetching
        with self._lock:
            if not self._is_valid_unsafe():
                self._fetch_token()
        return {"Authorization": f"Bearer {self._access_token}"}

    def _is_valid_unsafe(self) -> bool:
        """Check validity WITHOUT acquiring lock — call only inside lock."""
        return bool(self._access_token) and time.time() < self._expires_at

    def is_valid(self) -> bool:
        """Thread-safe validity check for external callers."""
        with self._lock:
            return self._is_valid_unsafe()

    def refresh(self) -> None:
        with self._lock:
            self._fetch_token()


# ─────────────────────────────────────────────────────────────────────
# 6. FormBasedAuth — POST credentials to login URL, carry session cookie
# Example: Government portals, legacy enterprise intranets
#
# Layer-6 Fix: SESSION_TTL_SECONDS tracks session expiry.
# is_valid() now returns False when TTL expires, triggering
# automatic re-login on next get_cookies() call.
# Configurable per portal — default 30 minutes.
#
# Usage:
#   auth = FormBasedAuth(
#       login_url="https://portal.example.com/login",
#       username=os.environ["PORTAL_USER"],
#       password=os.environ["PORTAL_PASS"],
#       session_ttl=1800,   # optional: override per portal
#   )
# ─────────────────────────────────────────────────────────────────────
class FormBasedAuth(BaseAuthProvider):

    # Default session TTL: 30 minutes.
    # Override per-instance via session_ttl= constructor param.
    DEFAULT_SESSION_TTL = 1800

    def __init__(self, login_url: str, username: str, password: str,
                 username_field: str = "username",
                 password_field: str = "password",
                 session_ttl: int = DEFAULT_SESSION_TTL):
        self._login_url      = login_url
        self._username       = username
        self._password       = password
        self._username_field = username_field
        self._password_field = password_field
        self._session_ttl    = session_ttl
        self._session_cookies: Dict[str, str] = {}
        self._logged_in      = False
        # ✅ Layer-6: track session expiry timestamp
        self._session_expires_at: float = 0.0

    def _do_login(self) -> None:
        """POST credentials, store cookies and reset TTL timer."""
        resp = httpx.post(
            self._login_url,
            data={
                self._username_field: self._username,
                self._password_field: self._password,
            },
            timeout=20,
            follow_redirects=True,
        )
        resp.raise_for_status()
        self._session_cookies = dict(resp.cookies)
        self._logged_in       = True
        # ✅ Layer-6: record when this session expires
        self._session_expires_at = time.time() + self._session_ttl

    def get_headers(self) -> Dict[str, str]:
        # Form auth uses cookies — headers are empty.
        # Connectors call get_cookies() for cookie-based requests.
        return {}

    def get_cookies(self) -> Dict[str, str]:
        # ✅ Layer-6: re-login automatically if session has expired
        if not self.is_valid():
            self._do_login()
        return self._session_cookies

    def is_valid(self) -> bool:
        # ✅ Layer-6: checks both logged-in state AND session TTL
        return self._logged_in and time.time() < self._session_expires_at

    def refresh(self) -> None:
        """Force re-login — resets session and TTL."""
        self._logged_in          = False
        self._session_expires_at = 0.0
        self._do_login()
