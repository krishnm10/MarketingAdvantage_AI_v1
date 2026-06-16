// lib/authToken.ts — session hint for middleware + in-memory JWT for API Bearer auth

const AUTH_HINT_COOKIE = "mai_auth_hint";

/** In-memory only (not localStorage) — used when httpOnly cookie is cross-origin in dev. */
let sessionToken: string | null = null;

function setAuthHintCookie() {
  if (typeof document === "undefined") return;
  const maxAge = 30 * 60;
  document.cookie = `${AUTH_HINT_COOKIE}=1; path=/; max-age=${maxAge}; SameSite=Lax`;
}

function clearAuthHintCookie() {
  if (typeof document === "undefined") return;
  document.cookie = `${AUTH_HINT_COOKIE}=; path=/; max-age=0; SameSite=Lax`;
}

export const getAuthToken = (): string | null => sessionToken;

export const setAuthToken = (token?: string) => {
  if (typeof token === "string" && token) {
    sessionToken = token;
  }
  setAuthHintCookie();
};

export const clearAuthToken = () => {
  sessionToken = null;
  clearAuthHintCookie();
};

export const hasAuthHint = (): boolean => {
  if (typeof document === "undefined") return false;
  return document.cookie
    .split(";")
    .some((part) => part.trim().startsWith(`${AUTH_HINT_COOKIE}=`));
};
