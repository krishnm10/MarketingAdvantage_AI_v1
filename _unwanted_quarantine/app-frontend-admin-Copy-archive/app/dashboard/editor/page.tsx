"use client";
import { useSession } from "next-auth/react";

export default function ChunksTable({ chunks }) {
  const { data: session } = useSession();
  const role = session?.user?.role;

  const canEdit = role === "admin" || role === "editor";

  return (
    <div className="overflow-x-auto bg-white shadow-md rounded-xl p-4">
      <table className="min-w-full text-sm text-left">
        <thead className="bg-gray-100 text-gray-700">
          <tr>
            <th className="p-2">Chunk</th>
            <th className="p-2">Cleaned Text</th>
            <th className="p-2">Actions</th>
          </tr>
        </thead>
        <tbody>
          {chunks.map((chunk) => (
            <tr key={chunk.id} className="border-b">
              <td className="p-2">{chunk.chunk_index}</td>
              <td className="p-2">{chunk.cleaned_text}</td>
              <td className="p-2">
                {canEdit ? (
                  <button className="bg-blue-500 text-white px-3 py-1 rounded-md hover:bg-blue-600">
                    Edit
                  </button>
                ) : (
                  <span className="text-gray-400 text-xs">Read Only</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
