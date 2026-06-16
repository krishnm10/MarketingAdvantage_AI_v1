import { redirect } from "next/navigation";

/** Legacy route — bookmarks to /dashboard/health forward to the Health hub. */
export default function DashboardHealthRedirect() {
  redirect("/health/system");
}
