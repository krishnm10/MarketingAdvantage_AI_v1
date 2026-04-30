"use client";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function AuthGuard({
  children,
  role,
}: {
  children: React.ReactNode;
  role?: string;
}) {
  const { data: session, status } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (status === "unauthenticated") router.push("/auth/login");
    if (role && session?.user?.role !== role) router.push("/dashboard");
  }, [status, router, role, session]);

  if (status === "loading") return <p className="text-gray-500">Loading...</p>;
  return <>{children}</>;
}
