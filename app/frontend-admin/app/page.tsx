import Link from "next/link";

export default function HomePage() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 relative overflow-hidden">
      {/* Background grid effect */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:3rem_3rem] opacity-30" />

      {/* Glow effects */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-primary-500/10 rounded-full blur-[120px]" />
      <div className="absolute bottom-0 right-0 w-[400px] h-[400px] bg-blue-500/5 rounded-full blur-[100px]" />

      <div className="relative z-10 flex flex-col items-center text-center px-6 max-w-3xl">
        {/* Logo */}
        <div className="mb-8 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary-500 to-primary-700 shadow-2xl shadow-primary-600/30">
          <svg className="h-8 w-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>

        {/* Title */}
        <h1 className="text-5xl font-extrabold tracking-tight text-white mb-3">
          Marketing Advantage{" "}
          <span className="bg-gradient-to-r from-primary-400 to-primary-300 bg-clip-text text-transparent">
            AI
          </span>
        </h1>

        {/* Subtitle */}
        <p className="text-lg text-slate-400 mb-4 font-medium">
          Enterprise Admin Console
        </p>
        <p className="text-sm text-slate-500 max-w-lg mb-10 leading-relaxed">
          Manage your AI-powered content ingestion pipeline, monitor system health,
          configure vector databases, and audit platform operations — all from a single, unified dashboard.
        </p>

        {/* CTA Buttons */}
        <div className="flex items-center gap-4">
          <Link
            href="/dashboard"
            className="group relative inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-primary-600 to-primary-500 px-8 py-3.5 text-sm font-semibold text-white shadow-lg shadow-primary-600/25 hover:shadow-primary-600/40 transition-all duration-300 hover:-translate-y-0.5"
          >
            Go to Dashboard
            <svg className="h-4 w-4 transition-transform group-hover:translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </Link>
          <Link
            href="/auth/login"
            className="inline-flex items-center gap-2 rounded-xl border border-slate-600 bg-slate-800/50 px-8 py-3.5 text-sm font-semibold text-slate-300 hover:bg-slate-700/50 hover:text-white transition-all duration-300 hover:-translate-y-0.5"
          >
            Sign In
          </Link>
        </div>

        {/* Feature badges */}
        <div className="mt-16 flex flex-wrap items-center justify-center gap-3">
          {[
            "Pluggable Pipeline",
            "ChromaDB + Qdrant",
            "Real-time Ingestion",
            "AI-Powered RAG",
            "Multi-LLM Support",
          ].map((feature) => (
            <span
              key={feature}
              className="rounded-full border border-slate-700 bg-slate-800/60 px-4 py-1.5 text-xs font-medium text-slate-400"
            >
              {feature}
            </span>
          ))}
        </div>
      </div>

      {/* Bottom credit */}
      <div className="absolute bottom-6 text-xs text-slate-600">
        © {new Date().getFullYear()} Marketing Advantage AI — v2.0 Enterprise
      </div>
    </div>
  );
}
