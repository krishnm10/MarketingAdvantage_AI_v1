"use client";

import { ReactNode } from "react";
import { getUserRole, Role } from "../lib/auth";

export default function RequireRole({
  allow,
  children,
}: {
  allow: Role[];
  children: ReactNode;
}) {
  const role = getUserRole();

  if (!role) return null;
  if (!allow.includes(role)) return null;

  return <>{children}</>;
}
