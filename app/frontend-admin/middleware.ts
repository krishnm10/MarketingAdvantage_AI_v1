import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const PROTECTED_PREFIXES = [
  "/dashboard",
  "/settings",
  "/ingestion",
  "/pipeline",
  "/secrets",
  "/health",
  "/audit",
  "/integrity",
];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  const isProtected = PROTECTED_PREFIXES.some(
    (prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`)
  );
  if (!isProtected) {
    return NextResponse.next();
  }

  const token = request.cookies.get("mai_auth_hint")?.value;
  if (!token) {
    const loginUrl = new URL("/auth/login", request.url);
    loginUrl.searchParams.set("next", pathname);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    "/dashboard/:path*",
    "/settings/:path*",
    "/ingestion/:path*",
    "/pipeline/:path*",
    "/secrets/:path*",
    "/health/:path*",
    "/audit/:path*",
    "/integrity/:path*",
  ],
};
