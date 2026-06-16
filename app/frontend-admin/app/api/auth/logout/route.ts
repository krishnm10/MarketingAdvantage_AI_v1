import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import { BACKEND_URL } from "@/lib/backendUrl";

export async function POST() {
  const token = cookies().get("mai_access_token")?.value;

  if (token) {
    try {
      await fetch(`${BACKEND_URL}/api/v2/auth/logout`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
    } catch {
      // Best-effort upstream logout.
    }
  }

  const res = NextResponse.json({ ok: true });
  res.cookies.delete("mai_access_token");
  res.cookies.delete("mai_auth_hint");
  return res;
}
