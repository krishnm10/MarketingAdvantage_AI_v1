import RequireAuth from "../../components/RequireAuth";
import { api } from "../api";

export default async function AuditPage() {
  let logs: any[] = [];

  try {
    const res = await api<any>("/admin-audit");
    logs = Array.isArray(res) ? res : [];
  } catch (e) {
    logs = [];
  }

  return (
    <RequireAuth>
      <h1 className="text-xl font-semibold">Admin Audit Log</h1>

      {logs.length === 0 && <p>No audit logs</p>}

      {logs.map((log) => (
        <div key={log.id} className="border p-3">
          <b>{log.action}</b>
          <pre>{JSON.stringify(log.meta_data, null, 2)}</pre>
        </div>
      ))}
    </RequireAuth>
  );
}
