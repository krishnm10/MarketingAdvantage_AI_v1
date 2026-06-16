export interface BreadcrumbItem {
  label: string;
  href: string;
}

const ROUTE_BREADCRUMBS: Record<string, BreadcrumbItem[]> = {
  "/dashboard": [{ label: "Dashboard", href: "/dashboard" }],

  // Pipeline Builder
  "/pipeline/ai-models": [
    { label: "Dashboard", href: "/dashboard" },
    { label: "Pipeline Builder", href: "/pipeline/ai-models" },
    { label: "AI Models", href: "/pipeline/ai-models" },
  ],
  "/pipeline/vector-db": [
    { label: "Dashboard", href: "/dashboard" },
    { label: "Pipeline Builder", href: "/pipeline/vector-db" },
    { label: "Vector Database", href: "/pipeline/vector-db" },
  ],
  "/pipeline/prompt-engineering": [
    { label: "Dashboard", href: "/dashboard" },
    { label: "Pipeline Builder", href: "/pipeline/prompt-engineering" },
    { label: "Prompt Engineering", href: "/pipeline/prompt-engineering" },
  ],

  // Ingestion & Sync
  "/ingestion/connectors": [
    { label: "Dashboard", href: "/dashboard" },
    { label: "Ingestion & Sync", href: "/ingestion/connectors" },
    { label: "Connectors", href: "/ingestion/connectors" },
  ],
  "/ingestion/chunking-tokenization": [
    { label: "Dashboard", href: "/dashboard" },
    { label: "Ingestion & Sync", href: "/ingestion/chunking-tokenization" },
    { label: "Chunking & Tokenization", href: "/ingestion/chunking-tokenization" },
  ],
  "/ingestion/deduplication": [
    { label: "Dashboard", href: "/dashboard" },
    { label: "Ingestion & Sync", href: "/ingestion/deduplication" },
    { label: "Deduplication Engine", href: "/ingestion/deduplication" },
  ],
  "/ingestion/files": [
    { label: "Dashboard", href: "/dashboard" },
    { label: "Ingestion & Sync", href: "/ingestion/files" },
    { label: "Ingested Files", href: "/ingestion/files" },
  ],

  // Secrets & Security
  "/secrets": [
    { label: "Dashboard", href: "/dashboard" },
    { label: "Secrets & Security", href: "/secrets" },
  ],

  // Health & Integrity
  "/health/system": [
    { label: "Dashboard", href: "/dashboard" },
    { label: "Health & Integrity", href: "/health/system" },
    { label: "System Health", href: "/health/system" },
  ],
  "/health/scheduler": [
    { label: "Dashboard", href: "/dashboard" },
    { label: "Health & Integrity", href: "/health/scheduler" },
    { label: "Validation Scheduler", href: "/health/scheduler" },
  ],
  "/health/phantom-profiler": [
    { label: "Dashboard", href: "/dashboard" },
    { label: "Health & Integrity", href: "/health/phantom-profiler" },
    { label: "Phantom Profiler", href: "/health/phantom-profiler" },
  ],
};

function humanizeSegment(segment: string): string {
  if (!segment) return "";
  return segment
    .split("-")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function resolveBreadcrumbs(pathname: string): BreadcrumbItem[] {
  const path = pathname.split("?")[0];

  let bestMatch: string | null = null;
  for (const prefix of Object.keys(ROUTE_BREADCRUMBS)) {
    if (path === prefix || path.startsWith(prefix + "/")) {
      if (!bestMatch || prefix.length > bestMatch.length) {
        bestMatch = prefix;
      }
    }
  }

  if (bestMatch) {
    return ROUTE_BREADCRUMBS[bestMatch];
  }

  const parts = path.split("/").filter(Boolean);
  return parts.map((part, index) => ({
    label: humanizeSegment(part),
    href: "/" + parts.slice(0, index + 1).join("/"),
  }));
}

