"use client";

import { Suspense } from "react";
import Sidebar from "./Sidebar";
import Navbar from "./Navbar";
import Footer from "./Footer";
import { AuthProvider } from "@/contexts/AuthContext";
import { TenantProvider } from "@/contexts/TenantContext";
import { useTenant } from "@/contexts/TenantContext";
import { ConfigProvider, ConfigBootGate } from "@/contexts/ConfigContext";

function AppShellContent({ children }: { children: React.ReactNode }) {
  return (
    <TenantScopedShell>{children}</TenantScopedShell>
  );
}

function TenantScopedShell({ children }: { children: React.ReactNode }) {
  const { clientId, tenantVersion } = useTenant();

  return (
    <div
      key={`${clientId}-${tenantVersion}`}
      className="flex h-screen overflow-hidden bg-slate-50"
    >
      <Sidebar />

      <div className="flex flex-1 flex-col ml-64 transition-all duration-300">
        <Navbar />
        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-7xl px-6 py-6 animate-fade-in">
            {children}
          </div>
        </main>
        <Footer />
      </div>
    </div>
  );
}

export default function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <Suspense fallback={<div className="flex h-screen items-center justify-center">Loading...</div>}>
      <AuthProvider>
        <TenantProvider>
          <ConfigProvider>
            <ConfigBootGate>
              <AppShellContent>{children}</AppShellContent>
            </ConfigBootGate>
          </ConfigProvider>
        </TenantProvider>
      </AuthProvider>
    </Suspense>
  );
}
