"use client";
import { useEffect, useState } from "react";
import Table from "@/components/ui/Table";
import Button from "@/components/ui/Button";
import apiClient from "@/lib/apiClient";
import Link from "next/link";

interface IngestedFile {
  id: string;
  file_name: string;
  file_type: string;
  status: string;
  total_chunks: number;
  created_at: string;
}

export default function IngestionListPage() {
  const [files, setFiles] = useState<IngestedFile[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const res = await apiClient.get("/ingestion-admin/files");
        setFiles(res.data);
      } catch (err) {
        console.error("Failed to fetch ingestion files:", err);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const rows = files.map((f) => [
    <Link key={f.id} href={`/ingestion/${f.id}`} className="text-primary hover:underline">
      {f.file_name}
    </Link>,
    f.file_type,
    f.status,
    f.total_chunks,
    new Date(f.created_at).toLocaleString(),
  ]);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-semibold">Ingested Files</h1>
        <Link href="/ingestion/upload">
          <Button>Upload New</Button>
        </Link>
      </div>

      {loading ? (
        <p className="text-gray-500">Loading files…</p>
      ) : files.length === 0 ? (
        <p className="text-gray-500">No files ingested yet.</p>
      ) : (
        <Table headers={["File Name", "Type", "Status", "Chunks", "Created At"]} rows={rows} />
      )}
    </div>
  );
}
