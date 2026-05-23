"use client";

import { useEffect, useState } from "react";

/**
 * Returns a debounced copy of `value` updated after `delayMs` of stability.
 */
export function useDebounce<T>(value: T, delayMs: number = 500): T {
  const [debounced, setDebounced] = useState<T>(value);

  useEffect(() => {
    const handle = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(handle);
  }, [value, delayMs]);

  return debounced;
}
