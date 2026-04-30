"use client";

import { useSession } from "next-auth/react";
import { useState, useEffect } from "react";
import axios from "axios";
import clsx from "clsx";
import RequireRole from "@/app/components/RequireRole"; // ✅ Added role protection
import IngestionFeed from "@/app/components/ingestion-feed"; // ✅ Ensure this path exists

const tabs = ["Ingestion Files", "Upload", "Sync", "Health", "Live Feed"];

export default function UnifiedDashboard() {
  const { data: session } = useSession();
  const [activeTab, setActiveTab] = useState("Ingestion Files");
  const [files, setFiles] = useState<any[]>([]);
  const [health, setHealth] = useState<any>(null);
  const [syncResult, setSyncResult] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const role = session?.user?.role;
  const token = session?.user?.accessToken;
  const canEdit = role === "admin" || role === "editor";
  const canSync = role === "admin";

  // ------------------------
  // Fetch Ingestion Files
  // ------------------------
  const fetchFiles = async () => {
    try {
      const res = await axios.get(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v2/ingestion-admin/files`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setFiles(res.data);
    } catch (e) {
      console.error("Error fetching ingestion files:", e);
    }
  };

  // ------------------------
  // Fetch Health Info
  // ------------------------
  const fetchHealth = async () => {
    try {
      const res = await axios.get(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v2/ingestion/health`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setHealth(res.data);
    } catch {
      setHealth({ status: "offline" });
    }
  };

  // ------------------------
  // Handle Upload
  // ------------------------
  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setMessage("");
    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await axios.post(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v2/ingestion/upload`,
        formData,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "multipart/form-data",
          },
        }
      );
      setMessage(res.status === 200 ? "✅ Upload Successful" : "❌ Upload Failed");
      fetchFiles();
    } catch (err: any) {
      setMessage("❌ " + err.message);
    } finally {
      setLoading(false);
    }
  };

  // ------------------------
  // Handle Sync Operations
  // ------------------------
  const handleSync = async (endpoint: string, label: string) => {
    try {
      const res = await axios.post(
        `${process.env.NEXT_PUBLIC_BACKEND_URL}/api/v2/ingestion-admin/sync/${endpoint}`,
        {},
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setSyncResult(`✅ ${label}: ${res.data?.message || "done"}`);
    } catch (err: any) {
      setSyncResult(`❌ ${label} failed: ${err.message}`);
    }
  };

  // ------------------------
  // Periodic Refresh
  // ------------------------
  useEffect(() => {
    fetchFiles();
    fetchHealth();
    const interval = setInterval(() => {
      fetchFiles();
      fetchHealth();
    }, 15000);
    return () => clearInterval(interval);
  }, [token]);

  // ------------------------
  // Page Render
  // ------------------------
  return (
    <RequireRole allowedRoles={["admin", "editor", "viewer"]}>
      <div className="p-6">
        <h1 className="text-2xl font-bold mb-6">
          Ingestion Dashboard ({role?.toUpperCase()})
        </h1>

        {/* Tabs */}
        <div className="flex space-x-4 border-b mb-6">
          {tabs.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={clsx(
                "pb-2 text-lg font-medium transition-colors duration-200",
                activeTab === tab
                  ? "border-b-2 border-blue-600 text-blue-600"
                  : "text-gray-600 hover:text-blue-600"
              )}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Tab: Ingestion Files */}
        {activeTab === "Ingestion Files" && (
          <section>
            <h2 className="text-xl mb-3 font-semibold">📂 Ingested Files</h2>
            {files.length === 0 ? (
              <p className="text-gray-500">No files available.</p>
            ) : (
              <table className="min-w-full text-sm text-left border border-gray-200 rounded">
                <thead className="bg-gray-100 border-b">
                  <tr>
                    <th className="p-2">File Name</th>
                    <th className="p-2">Status</th>
                    <th className="p-2">Uploaded</th>
                    {canEdit && <th className="p-2">Actions</th>}
                  </tr>
                </thead>
                <tbody>
                  {files.map((f) => (
                    <tr key={f.id} className="border-b hover:bg-gray-50">
                      <td className="p-2">{f.file_name}</td>
                      <td className="p-2">{f.status}</td>
                      <td className="p-2">
                        {new Date(f.created_at).toLocaleString()}
                      </td>
                      {canEdit && (
                        <td className="p-2">
                          <button
                            onClick={() => alert(`Editing ${f.file_name}`)}
                            className="bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600"
                          >
                            Edit
                          </button>
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        )}

        {/* Tab: Upload */}
        {activeTab === "Upload" && (
          <section>
            {!canEdit ? (
              <p className="text-gray-500">🔒 You can only view data.</p>
            ) : (
              <div>
                <h2 className="text-xl mb-3 font-semibold">📤 Upload File</h2>
                <input
                  type="file"
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                  className="border p-2 rounded w-64 mb-3"
                />
                <button
                  onClick={handleUpload}
                  disabled={!file || loading}
                  className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 disabled:opacity-50"
                >
                  {loading ? "Uploading..." : "Upload"}
                </button>
                {message && (
                  <p className="mt-3 text-sm text-gray-700">{message}</p>
                )}
              </div>
            )}
          </section>
        )}

        {/* Tab: Sync */}
        {activeTab === "Sync" && (
          <section>
            {!canSync ? (
              <p className="text-gray-500">🔒 Only admins can sync data.</p>
            ) : (
              <div>
                <h2 className="text-xl mb-3 font-semibold">🔄 Sync Tools</h2>
                <div className="space-y-3">
                  <button
                    onClick={() => handleSync("orphans", "Delete Orphans")}
                    className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
                  >
                    Delete Orphans
                  </button>
                  <button
                    onClick={() =>
                      handleSync("fix/chroma-to-db", "Fix Chroma → DB")
                    }
                    className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
                  >
                    Fix Chroma → DB
                  </button>
                  <button
                    onClick={() =>
                      handleSync("fix/db-to-chroma", "Fix DB → Chroma")
                    }
                    className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
                  >
                    Fix DB → Chroma
                  </button>
                </div>
                {syncResult && (
                  <p className="mt-4 text-sm text-gray-700">{syncResult}</p>
                )}
              </div>
            )}
          </section>
        )}

        {/* Tab: Health */}
        {activeTab === "Health" && (
          <section>
            <h2 className="text-xl mb-3 font-semibold">🩺 System Health</h2>
            {health ? (
              <div className="bg-white shadow rounded p-4 text-sm text-gray-700">
                <p>
                  <strong>Status:</strong>{" "}
                  {health.status ||
                    (health.db_connected ? "✅ Online" : "❌ Offline")}
                </p>
                <p>
                  <strong>DB:</strong>{" "}
                  {health.db_connected ? "✅ Connected" : "❌ Not Connected"}
                </p>
                <p>
                  <strong>Chroma:</strong>{" "}
                  {health.chroma_connected ? "✅ Connected" : "❌ Not Connected"}
                </p>
                <p>
                  <strong>LLM Ready:</strong>{" "}
                  {health.llm_ready ? "✅ Yes" : "❌ No"}
                </p>
              </div>
            ) : (
              <p>Loading health data...</p>
            )}
          </section>
        )}

        {/* Tab: Live Feed */}
        {activeTab === "Live Feed" && (
          <section>
            <h2 className="text-xl mb-3 font-semibold">
              ⚡ Real-Time Ingestion Feed
            </h2>
            <IngestionFeed />
          </section>
        )}
      </div>
    </RequireRole>
  );
}
