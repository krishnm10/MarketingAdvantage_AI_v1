"use client";
import { InputHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
}

export default function Input({ label, error, className, ...props }: InputProps) {
  return (
    <div className="flex flex-col space-y-1.5">
      {label && <label className="text-sm font-medium text-slate-300">{label}</label>}
      <input
        className={cn(
          "border border-slate-600 bg-slate-800/50 rounded-lg px-3 py-2.5 text-sm text-slate-200 placeholder:text-slate-500 focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500 focus:outline-none transition-all",
          error && "border-red-500/50 focus:ring-red-500/50 focus:border-red-500",
          className
        )}
        {...props}
      />
      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  );
}

