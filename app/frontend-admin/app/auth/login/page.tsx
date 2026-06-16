"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Zap, Loader2, AlertCircle, Eye, EyeOff } from "lucide-react";
import { setAuthToken } from "@/lib/authToken";

export default function LoginPage() {
  const router = useRouter();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ username, password }),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        throw new Error(
          typeof data?.detail === "string"
            ? data.detail
            : "Invalid username or password"
        );
      }

      setAuthToken(data.access_token);
      router.push("/dashboard");
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Invalid username or password";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 relative overflow-hidden">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:3rem_3rem] opacity-20" />
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-primary-500/10 rounded-full blur-[120px]" />

      <div className="relative z-10 w-full max-w-md px-6">
        <div className="text-center mb-8">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-primary-500 to-primary-700 shadow-2xl shadow-primary-600/30">
            <Zap className="h-7 w-7 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white">Welcome Back</h1>
          <p className="mt-1 text-sm text-slate-400">Sign in to Marketing Advantage AI Admin</p>
        </div>

        <form
          onSubmit={handleSubmit}
          className="rounded-2xl border border-slate-700/50 bg-slate-900/80 backdrop-blur-xl p-8 shadow-2xl"
        >
          {error && (
            <div className="mb-5 flex items-center gap-2 rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-3">
              <AlertCircle className="h-4 w-4 text-red-400 flex-shrink-0" />
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}

          <div className="mb-4">
            <label className="block text-xs font-medium text-slate-400 mb-1.5">Username</label>
            <input
              type="text"
              placeholder="Enter your username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full rounded-lg border border-slate-600/50 bg-slate-800/50 px-4 py-2.5 text-sm text-white placeholder-slate-500 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 focus:outline-none transition-all"
              required
            />
          </div>

          <div className="mb-6">
            <label className="block text-xs font-medium text-slate-400 mb-1.5">Password</label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-lg border border-slate-600/50 bg-slate-800/50 px-4 py-2.5 pr-10 text-sm text-white placeholder-slate-500 focus:border-primary-500 focus:ring-2 focus:ring-primary-500/20 focus:outline-none transition-all"
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
              >
                {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-lg bg-gradient-to-r from-primary-600 to-primary-500 py-2.5 text-sm font-semibold text-white shadow-lg shadow-primary-600/25 hover:shadow-primary-600/40 disabled:opacity-50 transition-all duration-300 flex items-center justify-center gap-2"
          >
            {loading ? (
              <><Loader2 className="h-4 w-4 animate-spin" /> Signing in...</>
            ) : (
              "Sign In"
            )}
          </button>
        </form>

        <p className="mt-6 text-center text-xs text-slate-600">
          Marketing Advantage AI — Enterprise Admin Console
        </p>
      </div>
    </div>
  );
}
