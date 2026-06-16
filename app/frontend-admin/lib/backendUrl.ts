/** Server-side FastAPI base URL (no trailing slash). */
export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_API_URL || "http://localhost:8000";

export const TOKEN_MAX_AGE_SECONDS = 30 * 60;
