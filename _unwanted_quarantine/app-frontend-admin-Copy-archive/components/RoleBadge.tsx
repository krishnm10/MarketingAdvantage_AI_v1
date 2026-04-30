"use client";

import { getUserRole } from "../lib/auth";

export default function RoleBadge() {
  const role = getUserRole();
  if (!role) return null;

  return (
    <div style={{ fontSize: 12, opacity: 0.6 }}>
      Role: <b>{role}</b>
    </div>
  );
}
