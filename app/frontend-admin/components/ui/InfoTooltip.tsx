"use client";

import { Info } from "lucide-react";

/**
 * A small 'i' icon that shows a tooltip on hover.
 * Pure CSS — no JS state, no external library.
 */
export default function InfoTooltip({ text }: { text: string }) {
  if (!text) return null;

  return (
    <span className="relative inline-flex items-center group/tip">
      <span className="flex items-center justify-center w-4 h-4 rounded-full text-slate-400 hover:text-primary-500 hover:bg-primary-50 cursor-help transition-colors">
        <Info className="w-3 h-3" />
      </span>
      <span
        role="tooltip"
        className="
          pointer-events-none absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2
          w-64 px-3 py-2 rounded-lg
          bg-slate-800 text-white text-[11px] leading-relaxed font-normal shadow-lg
          opacity-0 scale-95 transition-all duration-150
          group-hover/tip:opacity-100 group-hover/tip:scale-100
        "
      >
        {text}
        {/* arrow */}
        <span className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-800" />
      </span>
    </span>
  );
}
