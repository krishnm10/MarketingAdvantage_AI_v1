"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Card from "@/components/ui/Card";
import Table from "@/components/ui/Table";
import Button from "@/components/ui/Button";
import apiClient from "@/lib/apiClient";

interface Chunk {
  id: string;
  cleaned_text: string;
  tokens: number;
  confidence: number;
  is_duplicate: boolean;
}
interface FileDetails {
  id: string;
  file_name: string;
  file_type: string;
  total_chunks: number;
  unique_chunks: number;
  duplicate_chunks: number;
  status: string;
}

export default function FileDetailPage() {
  const { fileId } = useParams<{ fileId: string }>();
  const [file, setFile] = useState<FileDetails | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!fileId) return;
    (async () => {
      try {
        const [fRes, cRes] = await Promise.all([
          apiClient.get(`/ingestion-admin/files/${fileId}`),
          apiClient.get(`/ingestion-admin/files/${fileId}/chunks`),
        ]);
        setFile(fRes.data);
        setChunks(cRes.data);
      } catch (err) {
        console.error("Failed to fetch file details:", err);
      } finally {
        setLoading(false);
      }
    })();
  }, [fileId]);

  const rows = chunks.map((c) => [
    c.cleaned_text.slice(0, 120) + (c.cleaned_text.length > 120 ? "…" : ""),
    c.tokens,
    c.confidence.toFixed(2),
    c.is_duplicate ? "Yes" : "No",
  ]);

  if (loading) return <p>Loading file details…</p>;
  if (!file) return <p>File not found.</p>;

  return (
    <div className="space-y-6">
      <Card title={`File: ${file.file_name}`}>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <p><strong>Type:</strong> {file.file_type}</p>
          <p><strong>Status:</strong> {file.status}</p>
          <p><strong>Total Chunks:</strong> {file.total_chunks}</p>
          <p><strong>Unique Chunks:</strong> {file.unique_chunks}</p>
          <p><strong>Duplicates:</strong> {file.duplicate_chunks}</p>
        </div>
        <Button variant="outline" className="mt-4" onClick={() => window.history.back()}>
          Back
        </Button>
      </Card>

      <Card title="Chunks">
        <Table
          headers={["Text Snippet", "Tokens", "Confidence", "Duplicate?"]}
          rows={rows}
        />
      </Card>
    </div>
  );
}
