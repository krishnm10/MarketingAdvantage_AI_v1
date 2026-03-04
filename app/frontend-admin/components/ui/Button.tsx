"use client";
import { cn } from "@/lib/utils";
import { ButtonHTMLAttributes } from "react";
import { Loader2 } from "lucide-react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "outline" | "danger";
  loading?: boolean;
}

export default function Button({
  className,
  children,
  variant = "primary",
  loading = false,
  ...props
}: ButtonProps) {
  const base =
    "inline-flex items-center justify-center px-4 py-2 rounded-md text-sm font-medium focus:outline-none transition";

  const variants = {
    primary: "bg-gradient-to-r from-primary-600 to-primary-500 text-white shadow-lg shadow-primary-600/25 hover:shadow-primary-600/40",
    secondary: "bg-slate-700 text-white hover:bg-slate-800",
    outline: "border border-slate-200 text-slate-600 hover:bg-slate-50 hover:border-slate-300",
    danger: "bg-gradient-to-r from-red-600 to-red-500 text-white shadow-lg shadow-red-600/25 hover:shadow-red-600/40",
  };

  return (
    <button
      className={cn(base, variants[variant], className, loading && "opacity-70")}
      disabled={loading}
      {...props}
    >
      {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
      {children}
    </button>
  );
}
