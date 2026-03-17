"use client";
import { useSession, signIn, signOut } from "next-auth/react";

type SessionUser = {
  accessToken?: string;
  role?: string;
};

export function useAuth() {
  const { data: session, status } = useSession();
  const loading = status === "loading";
  const user = session?.user as SessionUser | undefined;

  const login = async (email: string, password: string) => {
    await signIn("credentials", { email, password, redirect: true });
  };

  const logout = async () => {
    await signOut({ redirect: true });
  };

  return {
    user: session?.user,
    token: user?.accessToken,
    role: user?.role || "viewer",
    loading,
    login,
    logout,
  };
}
