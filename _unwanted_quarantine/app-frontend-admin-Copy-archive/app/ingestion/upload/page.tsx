"use client";

import { useState } from "react";
import { useSession } from "next-auth/react";
import apiClient from "@/lib/apiClient";

export default function UploadPage() {
  const { data: session } = useSession();
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  const role = session?.user?.role;
  const canUpload = role === "admin" || role === "editor";

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setStatus("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await apiClient.post("/api/v2/ingestion/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setStatus(
        res.status === 200
          ? "✅ File uploaded successfully!"
          : "❌ Unexpected response from server."
      );
    } catch (err: any) {
      console.error("Upload error:", err);
      setStatus("❌ Upload failed: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  if (!canUpload)
    return (
      <div className="p-6 text-gray-500">
        🔒 You have view-only access. Uploads are disabled.
      </div>
    );

  return (
    <div className="p-6">
      <h1 className="text-xl font-bold mb-4">Upload New File</h1>
      <div className="bg-white p-4 shadow-md rounded-md w-96">
        <input
          type="file"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="mb-3 w-full border p-2 rounded"
        />
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? "Uploading..." : "Upload"}
        </button>
        {status && (
          <p className="mt-3 text-sm text-gray-700 text-center">{status}</p>
        )}
      </div>
    </div>
  );
}
