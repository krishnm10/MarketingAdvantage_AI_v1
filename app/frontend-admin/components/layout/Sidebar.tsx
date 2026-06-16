"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  BarChart3,
  Building2,
  ChevronLeft,
  ChevronRight,
  Database,
  FileStack,
  LayoutDashboard,
  RefreshCw,
  ScrollText,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Upload,
  Wand2,
  Zap,
} from "lucide-react";
import { useState, type ComponentType } from "react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/lib/useAuth";

type NavItem = {
  href: string;
  label: string;
  icon: ComponentType<{ className?: string }>;
  adminOnly?: boolean;
  legacy?: boolean;
};

type NavSection = {
  label: string;
  adminOnly?: boolean;
  items: NavItem[];
};

const NAV_SECTIONS: NavSection[] = [
  {
    label: "Main",
    items: [
      { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
    ],
  },
  {
    label: "Pipeline Builder",
    items: [
      { href: "/pipeline/ai-models", label: "AI Models", icon: Wand2 },
      { href: "/pipeline/vector-db", label: "Vector Database", icon: Database },
      { href: "/pipeline/prompt-engineering", label: "Prompt Engineering", icon: Sparkles },
    ],
  },
  {
    label: "Ingestion & Sync",
    items: [
      { href: "/ingestion/connectors", label: "Connectors", icon: RefreshCw },
      { href: "/ingestion/chunking-tokenization", label: "Chunking & Tokenization", icon: FileStack },
      { href: "/ingestion/deduplication", label: "Deduplication Engine", icon: BarChart3 },
      { href: "/ingestion/files", label: "Ingested Files", icon: Upload },
      { href: "/dashboard/sync", label: "Sync Monitor", icon: Activity },
    ],
  },
  {
    label: "Secrets & Security",
    adminOnly: true,
    items: [
      { href: "/secrets", label: "Secrets & Security (BYOK)", icon: ShieldCheck, adminOnly: true },
    ],
  },
  {
    label: "Health & Integrity",
    items: [
      { href: "/health/system", label: "System Health", icon: Activity },
      { href: "/health/scheduler", label: "Validation Scheduler", icon: Zap },
      { href: "/health/phantom-profiler", label: "Phantom Profiler", icon: BarChart3 },
      { href: "/integrity", label: "Integrity", icon: ShieldCheck, adminOnly: true },
      { href: "/audit", label: "Audit Log", icon: ScrollText },
      { href: "/audit/ingestion", label: "Ingestion Audit", icon: FileStack },
    ],
  },
  {
    label: "Advanced",
    items: [
      { href: "/dashboard/multi-customer-rag", label: "Customers & RAG", icon: Building2 },
      { href: "/dashboard/retrieve/chat", label: "RAG Chat", icon: Search },
      { href: "/dashboard/retrieve", label: "Retrieve (Advanced)", icon: SlidersHorizontal },
      {
        href: "/settings",
        label: "Legacy Configuration",
        icon: ScrollText,
        adminOnly: true,
        legacy: true,
      },
      { href: "/settings/reranking", label: "Reranking & Rules", icon: SlidersHorizontal },
      { href: "/settings/prompt-builder", label: "Prompt Library", icon: Sparkles },
      { href: "/settings/databases", label: "Databases", icon: Database },
    ],
  },
];

function NavSkeleton({ collapsed }: { collapsed: boolean }) {
  return (
    <div className="space-y-6 px-3 py-4">
      {[1, 2, 3].map((section) => (
        <div key={section} className="space-y-2">
          {!collapsed && (
            <div className="mx-3 h-2 w-20 animate-pulse rounded bg-slate-700" />
          )}
          {[1, 2].map((item) => (
            <div
              key={item}
              className="mx-1 h-9 animate-pulse rounded-lg bg-slate-800"
            />
          ))}
        </div>
      ))}
    </div>
  );
}

export default function Sidebar() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const { role, loading } = useAuth();
  const isAdmin = role === "admin";

  const visibleSections = NAV_SECTIONS.map((section) => ({
    ...section,
    items: section.items.filter((item) => !item.adminOnly || isAdmin),
  })).filter((section) => {
    if (section.adminOnly && !isAdmin) return false;
    return section.items.length > 0;
  });

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 z-40 flex h-screen flex-col bg-slate-900 text-slate-300 transition-all duration-300 ease-in-out shadow-sidebar",
        collapsed ? "w-[68px]" : "w-64"
      )}
    >
      <div className="flex h-16 items-center justify-between border-b border-slate-700/50 px-4">
        {!collapsed && (
          <div className="flex items-center gap-2.5 animate-fade-in">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary-500 to-primary-700 shadow-lg shadow-primary-600/20">
              <Zap className="h-4 w-4 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold text-white leading-none">MAI Admin</h1>
              <span className="text-[10px] text-slate-500 font-medium">v2.0 — Enterprise</span>
            </div>
          </div>
        )}
        {collapsed && (
          <div className="mx-auto flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary-500 to-primary-700">
            <Zap className="h-4 w-4 text-white" />
          </div>
        )}
      </div>

      <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-6">
        {loading ? (
          <NavSkeleton collapsed={collapsed} />
        ) : (
          visibleSections.map((section) => (
            <div key={section.label}>
              {!collapsed && (
                <p className="mb-2 px-3 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                  {section.label}
                </p>
              )}
              <ul className="space-y-0.5">
                {section.items.map((item) => {
                  const isActive =
                    pathname === item.href || pathname.startsWith(item.href + "/");
                  return (
                    <li key={item.href}>
                      <Link
                        href={item.href}
                        className={cn(
                          "group flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200",
                          isActive
                            ? "bg-primary-600/10 text-primary-400 shadow-sm"
                            : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                        )}
                        title={collapsed ? item.label : undefined}
                      >
                        <item.icon
                          className={cn(
                            "h-[18px] w-[18px] flex-shrink-0 transition-colors",
                            isActive
                              ? "text-primary-400"
                              : "text-slate-500 group-hover:text-slate-300"
                          )}
                        />
                        {!collapsed && (
                          <span className="flex items-center gap-1.5">
                            {item.label}
                            {item.legacy && (
                              <span className="rounded bg-amber-500/20 px-1 py-0.5 text-[9px] font-bold uppercase text-amber-400">
                                legacy
                              </span>
                            )}
                          </span>
                        )}
                        {isActive && !collapsed && (
                          <span className="ml-auto h-1.5 w-1.5 rounded-full bg-primary-400 animate-pulse-dot" />
                        )}
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))
        )}
      </nav>

      <div className="border-t border-slate-700/50 p-3">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="flex w-full items-center justify-center rounded-lg py-2 text-slate-500 hover:bg-slate-800 hover:text-slate-300 transition-colors"
        >
          {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </button>
      </div>
    </aside>
  );
}
