"use client";
import { useSession, signIn, signOut } from "next-auth/react";

export function useAuth() {
  const { data: session, status } = useSession();
  const loading = status === "loading";

  const login = async (email: string, password: string) => {
    await signIn("credentials", { email, password, redirect: true });
  };

  const logout = async () => {
    await signOut({ redirect: true });
  };

  return {
    user: session?.user,
    token: session?.user?.accessToken,
    role: session?.user?.role || "viewer",
    loading,
    login,
    logout,
  };
}
