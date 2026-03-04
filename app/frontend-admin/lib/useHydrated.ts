"use client";

import { useEffect, useState } from "react";

/**
 * Returns `true` only after the component has mounted on the client.
 * Use this to guard any locale-dependent rendering (dates, numbers)
 * that would cause a hydration mismatch between server and client.
 *
 * Usage:
 *   const hydrated = useHydrated();
 *   <span>{hydrated ? new Date(ts).toLocaleString() : "—"}</span>
 */
export function useHydrated(): boolean {
  const [hydrated, setHydrated] = useState(false);
  useEffect(() => setHydrated(true), []);
  return hydrated;
}

/**
 * Format a date string for display — returns placeholder during SSR.
 * Safe to call directly in JSX without hydration mismatch.
 */
export function useFormatDate() {
  const hydrated = useHydrated();

  return {
    /** e.g. "3/4/2026, 10:09:06 AM" */
    formatDateTime: (value: string | Date | undefined | null, fb = "—") =>
      !value ? fb : hydrated ? new Date(value).toLocaleString() : fb,

    /** e.g. "10:09:06 AM" */
    formatTime: (value: string | Date | undefined | null, fb = "—") =>
      !value ? fb : hydrated ? new Date(value).toLocaleTimeString() : fb,

    /** e.g. "3/4/2026" */
    formatDate: (value: string | Date | undefined | null, fb = "—") =>
      !value ? fb : hydrated ? new Date(value).toLocaleDateString() : fb,

    /** Current time string (for "last refreshed" labels) */
    nowTimeStr: () => (hydrated ? new Date().toLocaleTimeString() : "—"),

    hydrated,
  };
}

const fallback = "—";
