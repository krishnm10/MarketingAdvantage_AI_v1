"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Upload,
  FileStack,
  ShieldCheck,
  RefreshCw,
  ScrollText,
  Settings,
  Activity,
  Database,
  Zap,
  Search,
  ChevronLeft,
  ChevronRight,
  Eye,
  Wand2,
  SlidersHorizontal,
  Sparkles,
  BarChart3,
  Building2,
} from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

const NAV_SECTIONS = [
    {
    label: "Main",
    items: [
      { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
      { href: "/dashboard/multi-customer-rag", label: "Customers & RAG", icon: Building2 },
      { href: "/dashboard/health", label: "System Health", icon: Activity },
    ],
  },
  {
    label: "Ingestion",
    items: [
      { href: "/ingestion", label: "Files", icon: FileStack },
      { href: "/ingestion/upload", label: "Upload", icon: Upload },
      { href: "/dashboard/sync", label: "Sync", icon: RefreshCw },
    ],
  },
  {
    label: "Retrieval",
    items: [
      { href: "/dashboard/retrieve/chat", label: "RAG Chat", icon: Search },
      { href: "/dashboard/retrieve", label: "Retrieve (Advanced)", icon: SlidersHorizontal },
    ],
  },
  {
    label: "Administration",
    items: [
      { href: "/integrity",        label: "Integrity",          icon: ShieldCheck },
      { href: "/audit",            label: "Audit Log",          icon: ScrollText },
      { href: "/audit/ingestion",  label: "Ingestion Audit",    icon: BarChart3 },
      { href: "/settings",         label: "Configuration",      icon: Settings },
    ],
  },
  {
    label: "Infrastructure",
    items: [
      { href: "/settings/pipeline",       label: "Pipeline Builder",    icon: Wand2 },
      { href: "/settings/reranking",      label: "Reranking & Rules",   icon: SlidersHorizontal },
      { href: "/settings/prompt-builder", label: "Prompt Library",      icon: Sparkles },
      { href: "/settings/databases",      label: "Databases",           icon: Database },
      { href: "/settings/vision-encoder", label: "Vision Encoder",      icon: Eye },
    ],
  },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 z-40 flex h-screen flex-col bg-slate-900 text-slate-300 transition-all duration-300 ease-in-out shadow-sidebar",
        collapsed ? "w-[68px]" : "w-64"
      )}
    >
      {/* Logo / Brand */}
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

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-6">
        {NAV_SECTIONS.map((section) => (
          <div key={section.label}>
            {!collapsed && (
              <p className="mb-2 px-3 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                {section.label}
              </p>
            )}
            <ul className="space-y-0.5">
              {section.items.map((item) => {
                const isActive = pathname === item.href || pathname.startsWith(item.href + "/");
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
                          isActive ? "text-primary-400" : "text-slate-500 group-hover:text-slate-300"
                        )}
                      />
                      {!collapsed && <span>{item.label}</span>}
                      {isActive && !collapsed && (
                        <span className="ml-auto h-1.5 w-1.5 rounded-full bg-primary-400 animate-pulse-dot" />
                      )}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>

      {/* Collapse toggle */}
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
