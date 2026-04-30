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
    primary: "bg-primary text-white hover:bg-teal-700",
    secondary: "bg-gray-700 text-white hover:bg-gray-800",
    outline: "border border-gray-300 hover:bg-gray-100",
    danger: "bg-red-600 text-white hover:bg-red-700",
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
