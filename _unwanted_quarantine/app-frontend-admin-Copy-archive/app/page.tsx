import Link from "next/link";

export default function HomePage() {
  return (
    <div className="flex flex-col items-center justify-center text-center py-20">
      <h1 className="text-4xl font-bold mb-4">Marketing Advantage AI — Admin Console</h1>
      <p className="text-gray-500 mb-8">
        Manage ingestion, integrity, and audit systems with full visibility.
      </p>
      <Link href="/dashboard" className="bg-primary text-white px-6 py-3 rounded-lg hover:bg-teal-700">
        Go to Dashboard
      </Link>
    </div>
  );
}
