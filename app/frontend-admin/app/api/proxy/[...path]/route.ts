import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";
import { BACKEND_URL } from "@/lib/backendUrl";

export const runtime = "nodejs";
export const maxDuration = 600;

const METHODS_WITH_BODY = new Set(["POST", "PUT", "PATCH", "DELETE"]);

function isAllowedProxyPath(path: string): boolean {
  return path.startsWith("api/v2/");
}

function upstreamTimeoutMs(path: string): number {
  if (path.includes("retrieve/chat")) {
    return 330_000;
  }
  // Large PDFs: parse + visual OCR + hundreds of Ollama embed batches can exceed 2 min.
  if (
    path.includes("ingestion/upload") ||
    path.includes("ingestion/media/upload")
  ) {
    return 600_000;
  }
  return 120_000;
}

async function proxyRequest(
  req: NextRequest,
  params: { path: string[] }
): Promise<NextResponse> {
  const requestedPath = params.path.join("/");

  if (!isAllowedProxyPath(requestedPath)) {
    return NextResponse.json({ error: "forbidden" }, { status: 403 });
  }

  const token = cookies().get("mai_access_token")?.value;
  if (!token) {
    return NextResponse.json({ error: "unauthenticated" }, { status: 401 });
  }

  const search = req.nextUrl.search;
  const backendUrl = `${BACKEND_URL}/${requestedPath}${search}`;
  const method = req.method.toUpperCase();

  const forwardedHeaders = new Headers();
  const contentType = req.headers.get("content-type");
  if (contentType) {
    forwardedHeaders.set("content-type", contentType);
  }
  forwardedHeaders.set("authorization", `Bearer ${token}`);
  forwardedHeaders.set("accept", req.headers.get("accept") || "application/json");

  const init: RequestInit & { duplex?: "half" } = {
    method,
    headers: forwardedHeaders,
    signal: AbortSignal.timeout(upstreamTimeoutMs(requestedPath)),
  };

  if (METHODS_WITH_BODY.has(method)) {
    init.body = req.body;
    init.duplex = "half";
  }

  let upstream: Response;
  try {
    upstream = await fetch(backendUrl, init);
  } catch (err) {
    const detail =
      err instanceof Error && err.name === "TimeoutError"
        ? "Upstream request timed out"
        : "Cannot reach backend";
    return NextResponse.json({ detail }, { status: 504 });
  }

  const responseHeaders = new Headers();
  const upstreamContentType = upstream.headers.get("content-type");
  if (upstreamContentType) {
    responseHeaders.set("content-type", upstreamContentType);
  }
  const disposition = upstream.headers.get("content-disposition");
  if (disposition) {
    responseHeaders.set("content-disposition", disposition);
  }

  return new NextResponse(upstream.body, {
    status: upstream.status,
    headers: responseHeaders,
  });
}

type RouteContext = { params: { path: string[] } };

export async function GET(req: NextRequest, ctx: RouteContext) {
  return proxyRequest(req, ctx.params);
}

export async function POST(req: NextRequest, ctx: RouteContext) {
  return proxyRequest(req, ctx.params);
}

export async function PUT(req: NextRequest, ctx: RouteContext) {
  return proxyRequest(req, ctx.params);
}

export async function PATCH(req: NextRequest, ctx: RouteContext) {
  return proxyRequest(req, ctx.params);
}

export async function DELETE(req: NextRequest, ctx: RouteContext) {
  return proxyRequest(req, ctx.params);
}

export async function HEAD(req: NextRequest, ctx: RouteContext) {
  return proxyRequest(req, ctx.params);
}
