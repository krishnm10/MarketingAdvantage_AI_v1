"use client";

export interface JsonPathBadgeProps {
  path: string;
}

export function JsonPathBadge({ path }: JsonPathBadgeProps) {
  if (!path) return null;

  return (
    <span className="inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-mono text-slate-500 border border-slate-200">
      {path}
    </span>
  );
}

