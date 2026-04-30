"use client";

import ActionButton from "./ActionButton";
import { confirmAction } from "../lib/confirm";

export default function IntegrityActions() {
  async function fixDbToChroma() {
    if (!confirmAction("Re-embed DB content into Chroma? (Safe)")) return;

    await fetch(
      "http://localhost:8000/api/v2/ingestion-admin/integrity/fix/db-to-chroma",
      { method: "POST" }
    );

    alert("DB → Chroma fix completed");
  }

  async function fixChromaToDb() {
    if (
      !confirmAction(
        "WARNING: This deletes vectors without DB rows. Proceed?"
      )
    )
      return;

    await fetch(
      "http://localhost:8000/api/v2/ingestion-admin/sync/fix/chroma-to-db",
      { method: "POST" }
    );

    alert("Chroma → DB cleanup completed");
  }

  return (
    <div className="flex gap-4">
      <ActionButton
        label="Fix DB → Chroma (Re-embed)"
        onClick={fixDbToChroma}
      />

      <ActionButton
        label="Fix Chroma → DB (Delete Orphans)"
        onClick={fixChromaToDb}
        variant="danger"
      />
    </div>
  );
}
