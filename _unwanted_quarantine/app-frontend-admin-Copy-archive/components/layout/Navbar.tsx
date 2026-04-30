"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { LogOut } from "lucide-react";
import Button from "@/components/ui/Button";

export default function Navbar() {
  const pathname = usePathname();
  const user = { name: "Admin", role: "admin" }; // 🔐 will be replaced with NextAuth session

  return (
    <header className="h-14 bg-white border-b border-gray-200 flex items-center justify-between px-6 shadow-sm">
      <h1 className="text-lg font-semibold text-gray-700 capitalize">
        {pathname === "/" ? "Dashboard" : pathname.replace("/", "").replace("-", " ")}
      </h1>
      <div className="flex items-center space-x-4">
        <span className="text-sm text-gray-600">
          {user.name} ({user.role})
        </span>
        <Button variant="outline" onClick={() => console.log("logout")}>
          <LogOut className="h-4 w-4 mr-1" /> Logout
        </Button>
      </div>
    </header>
  );
}
