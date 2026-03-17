"use client";

import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { ReactNode, useEffect } from "react";

type SessionUser = {
  role?: string;
};

interface RequireRoleProps {
  allowedRoles: string[];
  children: ReactNode;
}

export default function RequireRole({ allowedRoles, children }: RequireRoleProps) {
  const { data: session, status } = useSession();
  const router = useRouter();
  const userRole = (session?.user as SessionUser | undefined)?.role;

  useEffect(() => {
    if (status === "loading") return;

    if (!session || !allowedRoles.includes(userRole || "")) {
      router.replace("/auth/login");
    }
  }, [session, status, router, allowedRoles, userRole]);

  if (status === "loading") {
    return (
      <div className="flex justify-center items-center h-screen">
        <p className="text-gray-600 text-sm">Checking access...</p>
      </div>
    );
  }

  if (!session || !allowedRoles.includes(userRole || "")) {
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
