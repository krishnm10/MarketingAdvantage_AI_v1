"use client";

import { useState } from "react";
import DiffViewer from "./DiffViewer";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";

type ChunkEditorProps = {
  chunkId: string;
  initialText: string;
};

export default function ChunkEditor({
  chunkId,
  initialText,
}: ChunkEditorProps) {
  const [text, setText] = useState(initialText);
  const [saving, setSaving] = useState(false);
  const [showDiff, setShowDiff] = useState(false);

  const hasChanges = text.trim() !== initialText.trim();

  async function save() {
    if (!hasChanges) {
      alert("No changes to save");
      return;
    }

    if (!window.confirm("Save changes to this chunk?")) return;

    setSaving(true);

    try {
      await apiClient.put(API.INGESTION_ADMIN.CHUNK_UPDATE(chunkId), {
        cleaned_text: text,
      });
      setShowDiff(false);
      alert("Chunk updated and re-embedded");
    } catch (err) {
      console.error("Chunk save failed", err);
      alert("Failed to save chunk");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-3">
      <textarea
        className="w-full rounded border p-2 text-sm"
        rows={6}
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      {hasChanges && (
        <div className="flex gap-3">
          <button
            onClick={() => setShowDiff(!showDiff)}
            className="rounded border px-3 py-1 text-xs"
          >
            {showDiff ? "Hide Diff" : "Show Diff"}
          </button>

          <button
            onClick={save}
            disabled={saving}
            className="rounded bg-blue-600 px-3 py-1 text-xs text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {saving ? "Saving..." : "Save"}
          </button>
        </div>
      )}

      {showDiff && (
        <DiffViewer before={initialText} after={text} />
      )}
    </div>
  );
}
