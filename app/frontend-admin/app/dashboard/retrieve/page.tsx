"use client";

import Link from "next/link";
import { useState, useRef, useEffect } from "react";
import apiClient from "@/lib/apiClient";
import {
  Search,
  Send,
  Loader2,
  AlertCircle,
  FileText,
  Shield,
  Clock,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Target,
  Layers,
  Hash,
  BarChart3,
  Zap,
  Copy,
  Check,
  Trash2,
  Info,
  Bot,
  MessageSquare,
  Bug,
  Database,
  Brain,
  Server,
  XCircle,
  Settings,
  Building2,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";
import InfoTooltip from "@/components/ui/InfoTooltip";
import { useTenant } from "@/contexts/TenantContext";

/* ────────────────────────────────────────────────────────────
   Types
   ──────────────────────────────────────────────────────────── */

interface SignalDetail {
  semantic_score?: number | null;
  tap_trust_score?: number | null;
  agentic_validation_score?: number | null;
  reasoning_quality_score?: number | null;
  conflict_modifier?: number | null;
  temporal_decay?: number | null;
}

interface ResultItem {
  rank: number;
  chunk_id: string;
  text: string;
  score: number;
  trust_decision?: string | null;
  explanation: Record<string, any>;
  signals?: SignalDetail | null;
}

interface RetrieveResponse {
  query: string;
  intent: string;
  search_mode?: string;
  total_results: number;
  total_dropped: number;
  latency_ms: number;
  results: ResultItem[];
  answer?: string | null;
  answer_model?: string | null;
  answer_latency_ms?: number | null;
  answer_error?: string | null;
  debug_info?: Record<string, any> | null;
}

interface HistoryEntry {
  query: string;
  intent: string;
  response: RetrieveResponse;
  timestamp: number;
}

const INTENTS = [
  { value: "answer", label: "Answer", description: "Strict, high-trust retrieval", icon: Target },
  { value: "explore", label: "Explore", description: "Broader recall for exploration", icon: Sparkles },
  { value: "audit", label: "Audit", description: "No filtering — full transparency", icon: Layers },
] as const;

/* ────────────────────────────────────────────────────────────
   Helpers
   ──────────────────────────────────────────────────────────── */

function trustColor(decision?: string | null) {
  if (!decision) return "text-slate-400";
  const d = decision.toLowerCase();
  if (d === "trusted") return "text-emerald-600";
  if (d === "provisional") return "text-amber-600";
  return "text-red-500";
}

function trustBg(decision?: string | null) {
  if (!decision) return "bg-slate-100 text-slate-600";
  const d = decision.toLowerCase();
  if (d === "trusted") return "bg-emerald-50 text-emerald-700 border-emerald-200";
  if (d === "provisional") return "bg-amber-50 text-amber-700 border-amber-200";
  return "bg-red-50 text-red-700 border-red-200";
}

function scoreColor(score: number) {
  if (score >= 0.7) return "text-emerald-600";
  if (score >= 0.4) return "text-amber-600";
  return "text-red-500";
}

function ScoreBar({ value, max = 1, color }: { value: number; max?: number; color: string }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="flex items-center gap-2 min-w-0">
      <div className="flex-1 h-1.5 rounded-full bg-slate-100 overflow-hidden">
        <div className={cn("h-full rounded-full transition-all", color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-slate-500 w-10 text-right">{value.toFixed(3)}</span>
    </div>
  );
}

/* ────────────────────────────────────────────────────────────
   Debug Panel Component
   ──────────────────────────────────────────────────────────── */

function DebugPanel({ entry }: { entry: HistoryEntry }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const d = entry.response.debug_info;
  const ae = entry.response.answer_error;

  const copyTrace = () => {
    const trace = {
      query: entry.query,
      intent: entry.intent,
      timestamp: new Date(entry.timestamp).toISOString(),
      response_meta: {
        total_results: entry.response.total_results,
        total_dropped: entry.response.total_dropped,
        latency_ms: entry.response.latency_ms,
        answer_model: entry.response.answer_model,
        answer_latency_ms: entry.response.answer_latency_ms,
        answer_error: entry.response.answer_error,
      },
      debug_info: d,
      results_summary: entry.response.results.map((r) => ({
        rank: r.rank,
        chunk_id: r.chunk_id,
        score: r.score,
        trust_decision: r.trust_decision,
      })),
    };
    navigator.clipboard.writeText(JSON.stringify(trace, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="ml-11 mt-2">
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          "flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-medium border transition-colors",
          open
            ? "bg-slate-100 text-slate-700 border-slate-200"
            : "text-slate-400 border-transparent hover:bg-slate-50 hover:text-slate-600 hover:border-slate-200"
        )}
      >
        <Bug className="w-3 h-3" />
        Debug Panel
        {open ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
      </button>

      {open && (
        <div className="mt-2 rounded-xl border border-slate-200 bg-slate-50/80 overflow-hidden shadow-sm">
          {/* Header row */}
          <div className="flex items-center justify-between px-4 py-2.5 bg-slate-100 border-b border-slate-200">
            <span className="text-[11px] font-semibold text-slate-600 uppercase tracking-wider flex items-center gap-1.5">
              <Settings className="w-3 h-3" /> Pipeline Debug
            </span>
            <button
              onClick={copyTrace}
              className="flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-700 transition-colors"
            >
              {copied ? <Check className="w-3 h-3 text-emerald-500" /> : <Copy className="w-3 h-3" />}
              {copied ? "Copied!" : "Copy trace JSON"}
            </button>
          </div>

          {/* Answer error banner */}
          {ae && (
            <div className="flex items-start gap-2 px-4 py-3 bg-red-50 border-b border-red-100">
              <XCircle className="w-3.5 h-3.5 text-red-500 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-[11px] font-semibold text-red-700">LLM Generation Error</p>
                <p className="text-[11px] text-red-600 mt-0.5 leading-relaxed">{ae}</p>
              </div>
            </div>
          )}

          {/* Pipeline config chips */}
          {d && (
            <div className="px-4 py-3 border-b border-slate-200">
              <p className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold mb-2">Pipeline Config</p>
              <div className="flex flex-wrap gap-1.5">
                {[
                  { label: "Embedder",    value: d.embedder,          icon: Brain },
                  { label: "LLM",         value: d.llm_provider,      icon: Bot },
                  { label: "VectorDB",    value: d.vectordb,           icon: Database },
                  { label: "Chunking",    value: d.chunking_strategy,  icon: Layers },
                  { label: "Search",      value: d.search_mode,        icon: Search },
                  { label: "Collection",  value: d.collection,         icon: Server },
                  { label: "Reranker",    value: d.reranker_used,      icon: Layers },
                ].map(({ label, value, icon: Icon }) => value && (
                  <span key={label} className="inline-flex items-center gap-1 text-[11px] bg-white border border-slate-200 text-slate-700 px-2 py-1 rounded-md">
                    <Icon className="w-3 h-3 text-slate-400" />
                    <span className="text-slate-400">{label}:</span>
                    <span className="font-semibold">{String(value)}</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Score summary */}
          {d && (
            <div className="px-4 py-3 border-b border-slate-200">
              <p className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold mb-2">Retrieval Signals</p>
              <div className="flex flex-wrap gap-3 text-[11px]">
                <span className="flex items-center gap-1 text-slate-600">
                  <Target className="w-3 h-3 text-slate-400" />
                  Top K: <strong className="ml-0.5">{d.top_k_returned ?? "—"}</strong>
                  {d.top_k_requested && <span className="text-slate-400"> (req: {d.top_k_requested})</span>}
                </span>
                <span className="flex items-center gap-1 text-slate-600">
                  <BarChart3 className="w-3 h-3 text-slate-400" />
                  Dropped: <strong className="ml-0.5">{d.total_dropped ?? "—"}</strong>
                </span>
                {d.max_score != null && (
                  <span className={cn("flex items-center gap-1", d.max_score >= (d.score_gate_threshold ?? 0.25) ? "text-emerald-600" : "text-red-500")}>
                    <Zap className="w-3 h-3" />
                    Max score: <strong className="ml-0.5">{d.max_score.toFixed(4)}</strong>
                    <span className="text-slate-400">(gate: {(d.score_gate_threshold ?? 0.25).toFixed(2)})</span>
                  </span>
                )}
                {d.hyde_enabled && (
                  <span className="flex items-center gap-1 text-violet-600">
                    <Sparkles className="w-3 h-3" />
                    HyDE active
                  </span>
                )}
              </div>
            </div>
          )}

          {/* LLM answer status */}
          <div className="px-4 py-3">
            <p className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold mb-2">Answer Generation</p>
            <div className="flex flex-wrap gap-3 text-[11px]">
              <span className="flex items-center gap-1 text-slate-600">
                <Bot className="w-3 h-3 text-slate-400" />
                Requested: <strong className="ml-0.5">{d?.generate_answer ? "yes" : "no"}</strong>
              </span>
              <span className="flex items-center gap-1 text-slate-600">
                <Check className="w-3 h-3 text-slate-400" />
                Generated: <strong className={cn("ml-0.5", d?.answer_generated ? "text-emerald-600" : "text-slate-500")}>{d?.answer_generated ? "yes" : "no"}</strong>
              </span>
              {entry.response.answer_model && (
                <span className="flex items-center gap-1 text-slate-600">
                  <Brain className="w-3 h-3 text-slate-400" />
                  Model: <strong className="ml-0.5 font-mono">{entry.response.answer_model}</strong>
                </span>
              )}
              {entry.response.answer_latency_ms != null && (
                <span className="flex items-center gap-1 text-slate-600">
                  <Clock className="w-3 h-3 text-slate-400" />
                  Latency: <strong className="ml-0.5">{entry.response.answer_latency_ms.toFixed(0)}ms</strong>
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ────────────────────────────────────────────────────────────
   Result Card Component
   ──────────────────────────────────────────────────────────── */

function ResultCard({ result }: { result: ResultItem }) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const signals = result.signals;

  const copyText = () => {
    navigator.clipboard.writeText(result.text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="rounded-xl border border-slate-200/60 bg-white shadow-card overflow-hidden transition-shadow hover:shadow-md">
      {/* Header */}
      <div className="flex items-center gap-3 px-5 py-3.5 border-b border-slate-100 bg-slate-50/50">
        <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-primary-50 text-primary-600 text-xs font-bold">
          #{result.rank}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className={cn("text-sm font-semibold", scoreColor(result.score))}>
              {(result.score * 100).toFixed(1)}%
            </span>
            <span className="text-slate-300">·</span>
            <span className={cn("text-xs font-medium px-2 py-0.5 rounded-full border", trustBg(result.trust_decision))}>
              <Shield className="w-3 h-3 inline mr-1 -mt-0.5" />
              {result.trust_decision || "unknown"}
            </span>
          </div>
          <p className="text-[10px] text-slate-400 font-mono mt-0.5 truncate">{result.chunk_id}</p>
        </div>
        <div className="flex items-center gap-1">
          <button onClick={copyText} className="p-1.5 rounded-md hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors" title="Copy text">
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
          </button>
          <button onClick={() => setExpanded(!expanded)} className="p-1.5 rounded-md hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors" title="Toggle details">
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Text content */}
      <div className="px-5 py-4">
        <p className={cn("text-sm text-slate-700 leading-relaxed whitespace-pre-wrap", !expanded && "line-clamp-4")}>
          {result.text}
        </p>
        {!expanded && result.text.length > 300 && (
          <button onClick={() => setExpanded(true)} className="text-xs text-primary-600 hover:text-primary-700 font-medium mt-2">
            Show full text →
          </button>
        )}
      </div>

      {/* Expanded details: signals */}
      {expanded && signals && (
        <div className="px-5 pb-4 border-t border-slate-100 pt-4">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
            <BarChart3 className="w-3.5 h-3.5 inline mr-1 -mt-0.5" />
            Signal Breakdown
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-2">
            {signals.semantic_score != null && (
              <div>
                <p className="text-[10px] text-slate-400 mb-0.5">Semantic Score</p>
                <ScoreBar value={signals.semantic_score} color="bg-blue-500" />
              </div>
            )}
            {signals.tap_trust_score != null && (
              <div>
                <p className="text-[10px] text-slate-400 mb-0.5">TAP Trust Score</p>
                <ScoreBar value={signals.tap_trust_score} color="bg-emerald-500" />
              </div>
            )}
            {signals.agentic_validation_score != null && (
              <div>
                <p className="text-[10px] text-slate-400 mb-0.5">Agentic Validation</p>
                <ScoreBar value={signals.agentic_validation_score} color="bg-violet-500" />
              </div>
            )}
            {signals.reasoning_quality_score != null && (
              <div>
                <p className="text-[10px] text-slate-400 mb-0.5">Reasoning Quality</p>
                <ScoreBar value={signals.reasoning_quality_score} color="bg-amber-500" />
              </div>
            )}
            {signals.conflict_modifier != null && (
              <div>
                <p className="text-[10px] text-slate-400 mb-0.5">Conflict Modifier</p>
                <ScoreBar value={signals.conflict_modifier} color="bg-rose-500" />
              </div>
            )}
            {signals.temporal_decay != null && (
              <div>
                <p className="text-[10px] text-slate-400 mb-0.5">Temporal Decay</p>
                <ScoreBar value={signals.temporal_decay} color="bg-orange-500" />
              </div>
            )}
          </div>

          {/* Raw explanation */}
          {Object.keys(result.explanation).length > 0 && (
            <details className="mt-4">
              <summary className="text-[10px] font-semibold text-slate-400 cursor-pointer hover:text-slate-600 uppercase tracking-wider">
                Raw Explanation JSON
              </summary>
              <pre className="mt-2 p-3 rounded-lg bg-slate-50 border border-slate-100 text-[11px] text-slate-600 font-mono overflow-auto max-h-48">
                {JSON.stringify(result.explanation, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}

/* ────────────────────────────────────────────────────────────
   Main Page
   ──────────────────────────────────────────────────────────── */

export default function RetrievePage() {
  const { clientId } = useTenant();
  const [query, setQuery] = useState("");
  const [intent, setIntent] = useState<string>("answer");
  const [topK, setTopK] = useState<string>("");
  const [searchMode, setSearchMode] = useState<string>("semantic");
  const [enableHyde, setEnableHyde] = useState(false);
  const [hybridAlpha, setHybridAlpha] = useState<number>(0.7);
  const [similarityThreshold, setSimilarityThreshold] = useState<string>("");
  const [generateAnswer, setGenerateAnswer] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [showSettings, setShowSettings] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  // Focus input on mount
  useEffect(() => { inputRef.current?.focus(); }, []);

  const handleSubmit = async () => {
    const q = query.trim();
    if (!q || loading) return;

    setLoading(true);
    setError("");

    try {
      const payload: Record<string, any> = {
        query: q,
        intent,
        client_id: clientId,
      };
      if (topK && parseInt(topK) > 0) payload.top_k = parseInt(topK);
      if (searchMode !== "semantic") payload.search_mode = searchMode;
      if (enableHyde) payload.enable_hyde = true;
      if (searchMode === "hybrid") payload.hybrid_alpha = hybridAlpha;
      if (similarityThreshold && parseFloat(similarityThreshold) > 0)
        payload.similarity_threshold = parseFloat(similarityThreshold);
      if (generateAnswer) payload.generate_answer = true;

      const res = await apiClient.post<RetrieveResponse>("/api/v2/retrieve/query", payload);
      const entry: HistoryEntry = {
        query: q,
        intent,
        response: res.data,
        timestamp: Date.now(),
      };
      setHistory((prev) => [entry, ...prev]);
      setQuery("");

      // Auto-scroll to results
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Retrieval failed. Check the backend.");
      console.error("Retrieve error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const clearHistory = () => setHistory([]);

  const selectedIntent = INTENTS.find((i) => i.value === intent)!;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
          <Search className="w-6 h-6 text-primary-600" />
          Enterprise Retrieval
        </h1>
        <p className="text-slate-500 text-sm mt-1 flex items-center gap-2">
          Semantic search with governance scoring — mirrors the Retrieve CLI
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 text-xs">
            <Building2 className="w-3 h-3" />
            {clientId}
          </span>
        </p>
        <Link
          href="/dashboard/multi-customer-rag"
          className="inline-flex items-center gap-1.5 mt-3 text-xs font-medium text-primary-600 hover:text-primary-800"
        >
          <Building2 className="w-3.5 h-3.5" />
          Customers & RAG dashboard
          <ChevronRight className="w-3.5 h-3.5 opacity-70" />
        </Link>
      </div>

      {/* Query input card */}
      <div className="rounded-xl border border-slate-200/60 bg-white shadow-card overflow-hidden">
        <div className="p-5">
          {/* Intent selector */}
          <div className="flex items-center gap-2 mb-4">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider mr-1">Intent:</span>
            {INTENTS.map((i) => {
              const Icon = i.icon;
              const active = intent === i.value;
              return (
                <button
                  key={i.value}
                  onClick={() => setIntent(i.value)}
                  className={cn(
                    "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border",
                    active
                      ? "bg-primary-50 text-primary-700 border-primary-200 shadow-sm"
                      : "text-slate-500 border-transparent hover:bg-slate-50 hover:text-slate-700"
                  )}
                  title={i.description}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {i.label}
                </button>
              );
            })}

            {/* Settings toggle */}
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={cn(
                "ml-auto flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs transition-colors border",
                showSettings
                  ? "bg-slate-100 text-slate-700 border-slate-200"
                  : "text-slate-400 border-transparent hover:bg-slate-50 hover:text-slate-600"
              )}
            >
              <Zap className="w-3 h-3" /> Advanced
            </button>
          </div>

          {/* Advanced settings */}
          {showSettings && (
            <div className="space-y-3 mb-4 p-4 rounded-lg bg-slate-50 border border-slate-100">
              <p className="text-[10px] text-slate-500">
                Session-only — changes here apply to this console only and do not alter tenant defaults in Settings → Reranking & Rules.
              </p>
              {/* Row 1: Top K + Search Mode */}
              <div className="flex flex-wrap items-center gap-4">
                <div className="flex items-center gap-2">
                  <label className="text-xs text-slate-500 font-medium">Top K:</label>
                  <InfoTooltip text="Number of top matching chunks to retrieve. Higher values return more context but may include less relevant results. This setting affects only the current request/session. To change tenant defaults, use Settings → Reranking & Rules." />
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={topK}
                    onChange={(e) => setTopK(e.target.value)}
                    placeholder="default"
                    className="w-20 rounded-md border border-slate-200 px-2.5 py-1 text-xs text-slate-700 focus:border-primary-300 focus:ring-1 focus:ring-primary-200 outline-none"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-slate-500 font-medium">Search Mode:</label>
                  <InfoTooltip text="How to search the vector store. Semantic = meaning-based (best for questions); Hybrid = combines semantic + keyword BM25 (best overall); Keyword = exact text match only. This setting affects only the current request/session. To change tenant defaults, use Settings → Reranking & Rules." />
                  <select
                    value={searchMode}
                    onChange={(e) => setSearchMode(e.target.value)}
                    className="rounded-md border border-slate-200 px-2.5 py-1 text-xs text-slate-700 focus:border-primary-300 focus:ring-1 focus:ring-primary-200 outline-none bg-white"
                  >
                    <option value="semantic">Semantic</option>
                    <option value="hybrid">Hybrid (Semantic + BM25)</option>
                    <option value="keyword">Keyword (BM25)</option>
                  </select>
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-slate-500 font-medium">Min Score:</label>
                  <InfoTooltip text="Minimum composite governance score (0.0–1.0). This is NOT raw cosine similarity — it combines semantic relevance, trust, and temporal signals. Typical useful values: 0.0 (no filter) to 0.5. Values above 0.6 will drop most results. This setting affects only the current request/session. To change tenant defaults, use Settings → Reranking & Rules." />
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={similarityThreshold}
                    onChange={(e) => setSimilarityThreshold(e.target.value)}
                    placeholder="0.0"
                    className="w-20 rounded-md border border-slate-200 px-2.5 py-1 text-xs text-slate-700 focus:border-primary-300 focus:ring-1 focus:ring-primary-200 outline-none"
                  />
                </div>
                {similarityThreshold && parseFloat(similarityThreshold) > 0.6 && (
                  <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-amber-50 border border-amber-200 text-amber-700 text-xs">
                    <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
                    Min Score above 0.6 will likely return 0 results — the governance composite score rarely exceeds this. Try 0.0–0.5.
                  </div>
                )}
              </div>

              {/* Row 2: Hybrid Alpha slider (visible when hybrid mode) */}
              {searchMode === "hybrid" && (
                <div className="flex items-center gap-3">
                  <label className="text-xs text-slate-500 font-medium whitespace-nowrap">
                    Hybrid Alpha: <span className="font-mono text-slate-700">{hybridAlpha.toFixed(2)}</span>
                  </label>
                  <InfoTooltip text="Balance between semantic and keyword search. 0.0 = pure keyword (BM25); 1.0 = pure semantic (embeddings). 0.5 is a balanced default." />
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={hybridAlpha}
                    onChange={(e) => setHybridAlpha(parseFloat(e.target.value))}
                    className="flex-1 h-1.5 accent-primary-600"
                  />
                  <div className="flex justify-between text-[10px] text-slate-400 w-36">
                    <span>Keyword</span>
                    <span>Semantic</span>
                  </div>
                </div>
              )}

              {/* Row 3: HyDE toggle */}
              <div className="flex items-center gap-3">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableHyde}
                    onChange={(e) => setEnableHyde(e.target.checked)}
                    className="rounded border-slate-300 text-primary-600 focus:ring-primary-200"
                  />
                  <span className="text-xs text-slate-500 font-medium">HyDE</span>
                </label>
                <InfoTooltip text="Hypothetical Document Embeddings — uses an LLM to generate a hypothetical answer first, then searches using that answer's embedding. Improves recall for complex questions but adds latency. This setting affects only the current request/session. To change tenant defaults, use Settings → Reranking & Rules." />
                <span className="text-[10px] text-slate-400">
                  Generate a hypothetical answer to improve embedding quality (requires LLM)
                </span>
              </div>

              {/* Row 4: Generate Answer toggle */}
              <div className="flex items-center gap-3 pt-1 border-t border-slate-200/60 mt-1">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={generateAnswer}
                    onChange={(e) => setGenerateAnswer(e.target.checked)}
                    className="rounded border-slate-300 text-primary-600 focus:ring-primary-200"
                  />
                  <span className="text-xs text-slate-700 font-semibold flex items-center gap-1">
                    <Bot className="w-3.5 h-3.5 text-primary-600" />
                    Generate LLM Answer
                  </span>
                </label>
                <InfoTooltip text="After retrieval, send the top chunks as context to the LLM resolved from merged Client JSON (single_llm_provider / routing) to generate a grounded, cited answer. Requires a valid LLM API key or credential. This setting affects only the current request/session. To change tenant defaults, use Settings → Reranking & Rules." />
                <span className="text-[10px] text-slate-400">
                  Full RAG: retrieve → ground → LLM generates a cited answer from your documents
                </span>
              </div>
            </div>
          )}

          {/* Query textarea */}
          <div className="relative">
            <textarea
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question… (Enter to send, Shift+Enter for new line)"
              rows={3}
              disabled={loading}
              className="w-full resize-none rounded-lg border border-slate-200 bg-slate-50/50 px-4 py-3 pr-14 text-sm text-slate-800 placeholder:text-slate-400 focus:border-primary-300 focus:ring-2 focus:ring-primary-100 outline-none transition-all disabled:opacity-60"
            />
            <button
              onClick={handleSubmit}
              disabled={loading || !query.trim()}
              className={cn(
                "absolute right-3 bottom-3 flex items-center justify-center w-9 h-9 rounded-lg transition-all",
                loading || !query.trim()
                  ? "bg-slate-100 text-slate-300 cursor-not-allowed"
                  : "bg-primary-600 text-white hover:bg-primary-700 shadow-sm"
              )}
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Error banner */}
        {error && (
          <div className="flex items-center gap-2 px-5 py-3 bg-red-50 border-t border-red-100 text-red-700 text-sm">
            <AlertCircle className="w-4 h-4 flex-shrink-0" /> {error}
          </div>
        )}
      </div>

      {/* Loading overlay */}
      {loading && (
        <div className="flex items-center justify-center gap-3 py-8 text-slate-500">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span className="text-sm">
            {generateAnswer ? "Retrieving context & generating answer…" : "Embedding query & running retrieval…"}
          </span>
        </div>
      )}

      {/* Results */}
      <div ref={resultsRef}>
        {history.map((entry, hi) => (
          <div key={entry.timestamp} className={cn("space-y-4", hi > 0 && "mt-8 pt-8 border-t border-slate-200/60")}>
            {/* Query echo */}
            <div className="flex items-start gap-3">
              <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-slate-800 text-white flex-shrink-0">
                <Search className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-slate-900">{entry.query}</p>
                <div className="flex items-center gap-3 mt-1 text-[11px] text-slate-400">
                  <span className="flex items-center gap-1">
                    <Target className="w-3 h-3" />
                    {entry.intent}
                  </span>
                  {entry.response.search_mode && entry.response.search_mode !== "semantic" && (
                    <span className="flex items-center gap-1 text-primary-500">
                      <Zap className="w-3 h-3" />
                      {entry.response.search_mode}
                    </span>
                  )}
                  <span className="flex items-center gap-1">
                    <FileText className="w-3 h-3" />
                    {entry.response.total_results} result{entry.response.total_results !== 1 ? "s" : ""}
                  </span>
                  <span className="flex items-center gap-1">
                    <Hash className="w-3 h-3" />
                    {entry.response.total_dropped} dropped
                  </span>
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {entry.response.latency_ms.toFixed(0)}ms
                  </span>
                </div>
              </div>
            </div>

            {/* LLM Generated Answer */}
            {entry.response.answer && (
              <div className="ml-11 rounded-xl border border-primary-200 bg-primary-50/40 overflow-hidden shadow-sm">
                <div className="flex items-center gap-2 px-4 py-2.5 bg-primary-50 border-b border-primary-200/70">
                  <Bot className="w-4 h-4 text-primary-600 flex-shrink-0" />
                  <span className="text-xs font-semibold text-primary-700">AI Answer</span>
                  {entry.response.answer_model && (
                    <span className="ml-1 text-[10px] text-primary-400 font-mono">{entry.response.answer_model}</span>
                  )}
                  {entry.response.answer_latency_ms != null && (
                    <span className="ml-auto text-[10px] text-primary-400 flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {entry.response.answer_latency_ms.toFixed(0)}ms
                    </span>
                  )}
                </div>
                <div className="px-5 py-4">
                  <p className="text-sm text-slate-800 leading-relaxed whitespace-pre-wrap">{entry.response.answer}</p>
                  <p className="text-[10px] text-primary-400 mt-3 flex items-center gap-1">
                    <MessageSquare className="w-3 h-3" />
                    Answer grounded on {Math.min(entry.response.total_results, 5)} retrieved source{entry.response.total_results !== 1 ? "s" : ""}. Verify citations against source text below.
                  </p>
                </div>
              </div>
            )}

            {/* LLM Generation Error Banner (shown when generate_answer=true but generation failed) */}
            {!entry.response.answer && entry.response.answer_error && (
              <div className="ml-11 flex items-start gap-3 p-3.5 rounded-xl bg-red-50 border border-red-200 text-red-700 shadow-sm">
                <XCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-xs font-semibold">Answer Generation Failed</p>
                  <p className="text-xs text-red-600 mt-0.5 leading-relaxed">{entry.response.answer_error}</p>
                </div>
              </div>
            )}

            {/* Debug Panel */}
            <DebugPanel entry={entry} />

            {/* No results */}
            {entry.response.results.length === 0 && (
              <div className="flex items-center gap-3 p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-800">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-semibold">No trusted results found</p>
                  <p className="text-xs text-amber-600 mt-0.5">
                    Try a different query, use the &ldquo;Explore&rdquo; intent for broader recall, or check that documents have been ingested.
                  </p>
                </div>
              </div>
            )}

            {/* Result cards */}
            <div className="space-y-3 ml-11">
              {entry.response.results.map((r) => (
                <ResultCard key={`${entry.timestamp}-${r.chunk_id}`} result={r} />
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Empty state */}
      {history.length === 0 && !loading && (
        <div className="text-center py-16">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-slate-100 mb-4">
            <Search className="w-7 h-7 text-slate-400" />
          </div>
          <p className="text-sm font-semibold text-slate-600">No queries yet</p>
          <p className="text-xs text-slate-400 mt-1 max-w-xs mx-auto">
            Type a question above to search your ingested knowledge base using enterprise retrieval with governance scoring.
          </p>
        </div>
      )}

      {/* Clear history */}
      {history.length > 0 && (
        <div className="flex justify-center pt-4">
          <button onClick={clearHistory} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-slate-400 hover:text-red-500 hover:bg-red-50 transition-colors border border-transparent hover:border-red-200">
            <Trash2 className="w-3 h-3" /> Clear history
          </button>
        </div>
      )}
    </div>
  );
}
