"use client";

import { useState, useEffect, useRef } from "react";
import { usePathname, useRouter } from "next/navigation";
import {
  LogOut,
  Bell,
  Search,
  User,
  ChevronRight,
  Building2,
  ChevronDown,
  Check,
  Loader2,
  AlertCircle,
  Plus,
} from "lucide-react";
import { clearAuthToken, hasAuthHint } from "@/lib/authToken";
import { cn } from "@/lib/utils";
import { useTenantOptional } from "@/contexts/TenantContext";
import { resolveBreadcrumbs } from "@/lib/breadcrumbs";
import CreateTenantDialog from "@/components/tenant/CreateTenantDialog";

function TenantSelector() {
  const tenant = useTenantOptional();
  const [open, setOpen] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  const [filter, setFilter] = useState("");
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setOpen(false);
        setFilter("");
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  if (!tenant) {
    return null;
  }

  const { clientId, setClientId, tenants, loadingTenants, tenantError } = tenant;

  const filteredTenants = tenants.filter((t) =>
    t.label.toLowerCase().includes(filter.toLowerCase())
  );

  const isInList = tenants.some((t) => t.id === clientId);

  return (
    <>
    <div className="relative" ref={dropdownRef}>
      {/* Trigger button */}
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          "flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm transition-colors",
          open
            ? "border-primary-300 bg-primary-50 text-primary-700"
            : "border-slate-200 bg-slate-50 text-slate-600 hover:border-slate-300 hover:bg-white"
        )}
      >
        <Building2 className="h-3.5 w-3.5" />
        {loadingTenants ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : (
          <span className="max-w-[120px] truncate font-medium">{clientId}</span>
        )}
        {tenantError && (
          <span title={tenantError}>
            <AlertCircle className="h-3.5 w-3.5 text-amber-500" />
          </span>
        )}
        {!isInList && !loadingTenants && (
          <span className="text-[10px] text-amber-600 bg-amber-100 px-1 rounded">
            custom
          </span>
        )}
        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 transition-transform",
            open && "rotate-180"
          )}
        />
      </button>

      {/* Dropdown */}
      {open && (
        <div className="absolute right-0 top-full mt-1 w-64 rounded-lg border border-slate-200 bg-white shadow-lg z-50 overflow-hidden">
          {/* Search */}
          <div className="p-2 border-b border-slate-100">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-slate-400" />
              <input
                type="text"
                placeholder="Search tenants..."
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="w-full pl-8 pr-3 py-1.5 text-sm border border-slate-200 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                autoFocus
              />
            </div>
          </div>

          {/* Tenant list */}
          <div className="max-h-64 overflow-y-auto p-1">
            {loadingTenants ? (
              <div className="flex items-center justify-center py-4 text-sm text-slate-400">
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
                Loading tenants...
              </div>
            ) : filteredTenants.length === 0 ? (
              <div className="py-4 text-center text-sm text-slate-400">
                No tenants found
              </div>
            ) : (
              filteredTenants.map((t) => (
                <button
                  key={t.id}
                  onClick={() => {
                    setClientId(t.id);
                    setOpen(false);
                    setFilter("");
                  }}
                  className={cn(
                    "w-full flex items-center gap-2 px-3 py-2 text-sm rounded-md transition-colors",
                    t.id === clientId
                      ? "bg-primary-50 text-primary-700"
                      : "text-slate-600 hover:bg-slate-50"
                  )}
                >
                  <Building2 className="h-3.5 w-3.5 flex-shrink-0" />
                  <span className="flex-1 text-left truncate">{t.label}</span>
                  {t.id === clientId && (
                    <Check className="h-3.5 w-3.5 text-primary-600" />
                  )}
                </button>
              ))
            )}
          </div>

          {/* Footer with count + create */}
          <div className="border-t border-slate-100">
            <button
              type="button"
              onClick={() => {
                setOpen(false);
                setFilter("");
                setCreateOpen(true);
              }}
              className="w-full flex items-center gap-2 px-3 py-2.5 text-sm font-medium text-primary-700 hover:bg-primary-50 transition-colors"
            >
              <Plus className="h-3.5 w-3.5" />
              New tenant…
            </button>
            <div className="px-3 py-2 bg-slate-50 text-[11px] text-slate-400">
              {tenants.length} tenant{tenants.length !== 1 ? "s" : ""} available
            </div>
          </div>
        </div>
      )}
    </div>
    <CreateTenantDialog open={createOpen} onClose={() => setCreateOpen(false)} />
    </>
  );
}

export default function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const breadcrumbs = resolveBreadcrumbs(pathname).map((crumb, index, all) => ({
    ...crumb,
    isLast: index === all.length - 1,
  }));
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    setIsLoggedIn(hasAuthHint());
  }, []);

  const handleLogout = async () => {
    try {
      await fetch("/api/auth/logout", { method: "POST", credentials: "include" });
    } catch {
      // Best-effort — clear local hint regardless.
    }
    clearAuthToken();
    router.push("/auth/login");
  };

  return (
    <header className="sticky top-0 z-30 h-16 bg-white/80 backdrop-blur-md border-b border-slate-200/60 flex items-center justify-between px-6">
      {/* Breadcrumbs */}
      <div className="flex items-center gap-1.5 text-sm">
        {breadcrumbs.map((crumb, i) => (
          <span key={crumb.href} className="flex items-center gap-1.5">
            {i > 0 && <ChevronRight className="h-3.5 w-3.5 text-slate-300" />}
            <span
              className={cn(
                "font-medium",
                crumb.isLast ? "text-slate-900" : "text-slate-400"
              )}
            >
              {crumb.label}
            </span>
          </span>
        ))}
      </div>

      {/* Right side */}
      <div className="flex items-center gap-3">
        {/* Tenant Selector */}
        <TenantSelector />

        {/* Search */}
        <button className="flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-sm text-slate-400 hover:border-slate-300 hover:bg-white transition-colors">
          <Search className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">Search...</span>
          <kbd className="hidden sm:inline-flex h-5 items-center rounded border border-slate-200 bg-white px-1.5 text-[10px] font-medium text-slate-400">
            ⌘K
          </kbd>
        </button>

        {/* Notifications */}
        <button className="relative rounded-lg p-2 text-slate-400 hover:bg-slate-100 hover:text-slate-600 transition-colors">
          <Bell className="h-4 w-4" />
          <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-primary-500 ring-2 ring-white" />
        </button>

        {/* Separator */}
        <div className="h-6 w-px bg-slate-200" />

        {/* User */}
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-primary-500 to-primary-700 shadow-sm">
            <User className="h-4 w-4 text-white" />
          </div>
          <div className="hidden sm:block">
            <p className="text-sm font-semibold text-slate-700 leading-none">Admin</p>
            <p className="text-[11px] text-slate-400">Administrator</p>
          </div>
        </div>

        {/* Logout */}
        {isLoggedIn && (
          <button
            onClick={handleLogout}
            className="rounded-lg p-2 text-slate-400 hover:bg-red-50 hover:text-red-500 transition-colors"
            title="Logout"
          >
            <LogOut className="h-4 w-4" />
          </button>
        )}
      </div>
    </header>
  );
}
