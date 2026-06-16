"use client";

import AppShell from "@/components/layout/AppShell";

export default function PipelineLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <AppShell>{children}</AppShell>;
}

