"use client";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

type SessionUser = {
  role?: string;
};

export default function AuthGuard({
  children,
  role,
}: {
  children: React.ReactNode;
  role?: string;
}) {
  const { data: session, status } = useSession();
  const router = useRouter();
  const userRole = (session?.user as SessionUser | undefined)?.role;

  useEffect(() => {
    if (status === "unauthenticated") router.push("/auth/login");
    if (role && userRole !== role) router.push("/dashboard");
  }, [status, router, role, userRole]);

  if (status === "loading") return <p className="text-gray-500">Loading...</p>;
  return <>{children}</>;
}
