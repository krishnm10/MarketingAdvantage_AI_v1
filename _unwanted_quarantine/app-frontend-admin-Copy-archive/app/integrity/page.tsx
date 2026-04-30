"use client";

import RequireRole from "../../components/RequireRole";
import { confirmAction } from "../../lib/confirm";
import apiClient from "@/lib/apiClient";

async function fixDbToChroma() {
  if (!confirmAction("Re-embed DB content into Chroma? (Safe)")) return;

  try {
    await apiClient.post("/ingestion-admin/integrity/fix/db-to-chroma");
    alert("✅ DB → Chroma fix completed successfully");
  } catch (err) {
    console.error("Fix DB → Chroma failed:", err);
    alert("❌ DB → Chroma fix failed. Check backend logs.");
  }
}

async function fixChromaToDb() {
  if (
    !confirmAction(
      "⚠️ WARNING: This mutates DB content. Proceed ONLY if you know what you're doing."
    )
  )
    return;

  try {
    await apiClient.post("/ingestion-admin/integrity/fix/chroma-to-db");
    alert("✅ Chroma → DB cleanup completed successfully");
  } catch (err) {
    console.error("Fix Chroma → DB failed:", err);
    alert("❌ Chroma → DB cleanup failed. Check backend logs.");
  }
}

export default function IntegrityHome() {
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Integrity Controls</h1>

      {/* ADMIN ONLY */}
      <RequireRole allow={["admin"]}>
        <div className="space-x-4">
          <button
            onClick={fixDbToChroma}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Fix DB → Chroma
          </button>

          <button
            onClick={fixChromaToDb}
            className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
          >
            Fix Chroma → DB
          </button>
        </div>
      </RequireRole>

      <ul className="list-disc pl-6 text-sm text-gray-600">
        <li>DB → Chroma is safe and idempotent</li>
        <li>Chroma → DB deletes vectors without DB rows</li>
        <li>All actions are manual and auditable</li>
      </ul>
    </div>
  );
}
