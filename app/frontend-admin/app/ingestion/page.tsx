import { redirect } from "next/navigation";

/** Legacy route — bookmarks to /ingestion forward to Ingested Files. */
export default function IngestionIndexRedirect() {
  redirect("/ingestion/files");
}
