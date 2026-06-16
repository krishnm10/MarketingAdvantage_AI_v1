"use client";

import AppShell from "@/components/layout/AppShell";

export default function HealthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <AppShell>{children}</AppShell>;
}

