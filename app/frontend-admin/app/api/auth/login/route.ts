import { NextRequest, NextResponse } from "next/server";
import { BACKEND_URL, TOKEN_MAX_AGE_SECONDS } from "@/lib/backendUrl";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10_000);

  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v2/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeout);
    const detail =
      err instanceof Error && err.name === "AbortError"
        ? "Auth server timed out — is the backend still starting?"
        : "Cannot reach auth server";
    return NextResponse.json({ detail }, { status: 503 });
  } finally {
    clearTimeout(timeout);
  }

  let data: Record<string, unknown>;
  try {
    data = await upstream.json();
  } catch {
    return NextResponse.json(
      { detail: "Invalid response from auth server" },
      { status: 502 }
    );
  }

  if (!upstream.ok) {
    return NextResponse.json(data, { status: upstream.status });
  }

  const accessToken = data.access_token;
  if (typeof accessToken !== "string" || !accessToken) {
    return NextResponse.json(
      { detail: "Auth server did not return a token" },
      { status: 502 }
    );
  }

  const res = NextResponse.json(data);
  res.cookies.set("mai_auth_hint", "1", {
    path: "/",
    maxAge: TOKEN_MAX_AGE_SECONDS,
    sameSite: "lax",
    httpOnly: false,
  });
  res.cookies.set("mai_access_token", accessToken, {
    path: "/",
    maxAge: TOKEN_MAX_AGE_SECONDS,
    sameSite: "lax",
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
  });

  return res;
}
