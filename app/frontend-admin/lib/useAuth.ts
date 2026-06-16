// lib/useAuth.ts — thin reader over AuthProvider context
"use client";

import { useAuthContext } from "@/contexts/AuthContext";

export function useAuth() {
  const ctx = useAuthContext();
  if (ctx) {
    return ctx;
  }

  return {
    user: null,
    role: "viewer",
    isAuthenticated: false,
    loading: false,
    refresh: async () => {},
  };
}
