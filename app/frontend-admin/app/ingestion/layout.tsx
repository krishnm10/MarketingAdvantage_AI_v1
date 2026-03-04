"use client";

import AppShell from "@/components/layout/AppShell";

export default function IngestionLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <AppShell>{children}</AppShell>;
}
