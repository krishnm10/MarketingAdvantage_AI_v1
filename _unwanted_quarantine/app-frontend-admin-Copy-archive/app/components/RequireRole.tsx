"use client";

import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { ReactNode, useEffect } from "react";

interface RequireRoleProps {
  allowedRoles: string[];
  children: ReactNode;
}

export default function RequireRole({ allowedRoles, children }: RequireRoleProps) {
  const { data: session, status } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (status === "loading") return;

    // Redirect if no session or role mismatch
    const userRole = session?.user?.role;
    if (!session || !allowedRoles.includes(userRole || "")) {
      router.replace("/auth/login");
    }
  }, [session, status, router, allowedRoles]);

  if (status === "loading") {
    return (
      <div className="flex justify-center items-center h-screen">
        <p className="text-gray-600 text-sm">Checking access...</p>
      </div>
    );
  }

  if (!session || !allowedRoles.includes(session.user?.role || "")) {
    return (
      <div className="flex justify-center items-center h-screen">
        <p className="text-red-500 text-sm">
          Unauthorized — your role does not have access to this page.
        </p>
      </div>
    );
  }

  return <>{children}</>;
}
