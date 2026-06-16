"use client";

import { useAuth } from "@/lib/useAuth";

export default function RoleBadge() {
  const { role } = useAuth();
  if (!role) return null;

  return (
    <div style={{ fontSize: 12, opacity: 0.6 }}>
      Role: <b>{role}</b>
    </div>
  );
}
