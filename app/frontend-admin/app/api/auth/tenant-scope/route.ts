import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";
import { BACKEND_URL, TOKEN_MAX_AGE_SECONDS } from "@/lib/backendUrl";

export async function POST(req: NextRequest) {
  const token = cookies().get("mai_access_token")?.value;
  if (!token) {
    return NextResponse.json({ error: "unauthenticated" }, { status: 401 });
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body" }, { status: 400 });
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v2/auth/tenant-scope`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(15_000),
    });
  } catch {
    return NextResponse.json({ detail: "Cannot reach auth server" }, { status: 503 });
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
