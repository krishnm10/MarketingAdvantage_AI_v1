import { getServerSession } from "next-auth";
import { authOptions } from "../api/auth/[...nextauth]/route";
import { redirect } from "next/navigation";
import Link from "next/link";

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // ✅ Session Handling (SSR-safe)
  const session = await getServerSession(authOptions);

  if (!session) {
    redirect("/auth/login");
  }

  const role = session.user?.role || "viewer";

  // ✅ Define Navigation Tabs Based on Role
  const navItems = [
    { href: "/dashboard", label: "Ingestion Files", roles: ["admin", "editor", "viewer"] },
    { href: "/dashboard/upload", label: "Upload", roles: ["admin", "editor"] },
    { href: "/dashboard/sync", label: "Sync", roles: ["admin"] },
    { href: "/dashboard/health", label: "Health", roles: ["admin", "editor", "viewer"] },
    { href: "/dashboard/live", label: "Live Feed", roles: ["admin", "editor", "viewer"] },
  ];

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* ✅ Header */}
      <header className="bg-white shadow p-4 flex justify-between items-center border-b">
        <h1 className="text-xl font-bold text-gray-700">
          MarketingAdvantage.AI — Dashboard ({role.toUpperCase()})
        </h1>
        <div className="text-sm text-gray-600">
          {session.user?.email ? (
            <>
              Signed in as <strong>{session.user.email}</strong>
            </>
          ) : (
            "Not signed in"
          )}
        </div>
      </header>

      {/* ✅ Navigation Bar */}
      <nav className="bg-gray-100 px-6 py-3 flex flex-wrap gap-4 border-b border-gray-200">
        {navItems
          .filter((item) => item.roles.includes(role))
          .map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="text-gray-700 hover:text-blue-600 text-sm font-medium transition-colors duration-200"
            >
              {item.label}
            </Link>
          ))}
      </nav>

      {/* ✅ Main Content Area */}
      <main className="flex-1 p-6 overflow-y-auto">{children}</main>

      {/* ✅ Footer */}
      <footer className="p-4 text-xs text-gray-500 text-center border-t bg-white">
        © {new Date().getFullYear()} MarketingAdvantage.AI — Real-time Ingestion Monitor
      </footer>
    </div>
  );
}
