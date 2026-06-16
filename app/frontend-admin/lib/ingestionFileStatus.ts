/** Backend ingestion statuses: uploaded | pending | processing | processed | failed */

export type IngestionFileBucket = "completed" | "failed" | "in_progress";

export function ingestionFileBucket(status: string): IngestionFileBucket {
  const s = (status || "").toLowerCase();
  if (s === "processed" || s === "completed" || s === "success") return "completed";
  if (s === "failed" || s === "error") return "failed";
  return "in_progress";
}

export function canReingestFile(status: string): boolean {
  const s = (status || "").toLowerCase();
  return s === "failed" || s === "error" || s === "uploaded" || s === "pending" || s === "processing";
}

export function ingestionStatusBadgeClass(status: string): string {
  const bucket = ingestionFileBucket(status);
  if (bucket === "completed") return "badge-success";
  if (bucket === "failed") return "badge-danger";
  if ((status || "").toLowerCase() === "processing") return "badge-warning";
  return "badge-warning";
}
