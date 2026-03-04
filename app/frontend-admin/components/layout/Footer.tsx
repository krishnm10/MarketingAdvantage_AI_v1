export default function Footer() {
  return (
    <footer className="mt-auto border-t border-slate-200/60 bg-white/50 backdrop-blur-sm">
      <div className="flex items-center justify-between px-6 py-3">
        <p className="text-xs text-slate-400">
          © {new Date().getFullYear()} Marketing Advantage AI — Enterprise Edition
        </p>
        <div className="flex items-center gap-4 text-xs text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse-dot" />
            System Online
          </span>
          <span>v2.0.0</span>
        </div>
      </div>
    </footer>
  );
}
