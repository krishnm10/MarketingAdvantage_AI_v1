export { default } from "next-auth/middleware";

export const config = {
  matcher: [
    "/dashboard/:path*",       // Protect dashboard and its subroutes
    "/api/v2/ingestion-admin/:path*", // Protect admin APIs if called directly
  ],
};
