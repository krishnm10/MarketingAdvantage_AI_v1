"use client";
import { useAuth } from "@/lib/useAuth";
import { Pencil, Lock } from "lucide-react";

export default function ChunksTable({ chunks }: { chunks: any[] }) {
  const { role } = useAuth();
  const canEdit = role === "admin" || role === "editor";

  return (
    <div className="overflow-x-auto rounded-xl border border-slate-700/50 bg-slate-800/50 backdrop-blur-sm">
      <table className="min-w-full text-sm text-left">
        <thead className="bg-slate-800/80 text-slate-300">
          <tr>
            <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider">Chunk</th>
            <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider">Cleaned Text</th>
            <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-700/30">
          {chunks.map((chunk) => (
            <tr key={chunk.id} className="hover:bg-slate-700/30 transition-colors">
              <td className="px-4 py-3 text-slate-300 font-mono">{chunk.chunk_index}</td>
              <td className="px-4 py-3 text-slate-300 max-w-md truncate">{chunk.cleaned_text}</td>
              <td className="px-4 py-3">
                {canEdit ? (
                  <button className="flex items-center gap-1.5 bg-primary-600/80 text-white px-3 py-1.5 rounded-lg hover:bg-primary-500 transition-colors text-xs font-medium">
                    <Pencil className="w-3 h-3" /> Edit
                  </button>
                ) : (
                  <span className="flex items-center gap-1 text-slate-500 text-xs">
                    <Lock className="w-3 h-3" /> Read Only
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
