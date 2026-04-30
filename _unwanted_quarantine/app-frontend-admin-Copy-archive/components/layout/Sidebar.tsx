import Link from "next/link";

export default function Sidebar() {
  const menu = [
    { href: "/dashboard", label: "Dashboard" },
    { href: "/ingestion", label: "Ingestion" },
    { href: "/admin/integrity", label: "Integrity" },
    { href: "/admin/sync", label: "Sync" },
    { href: "/admin/audit", label: "Audit" },
    { href: "/settings", label: "Settings" },
  ];

  return (
    <aside className="w-64 bg-secondary text-white min-h-screen p-5 space-y-4">
      <h2 className="text-xl font-bold mb-4">Ingestion Admin</h2>
      <nav className="flex flex-col space-y-2">
        {menu.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className="px-3 py-2 rounded hover:bg-primary transition"
          >
            {item.label}
          </Link>
        ))}
      </nav>
    </aside>
  );
}
