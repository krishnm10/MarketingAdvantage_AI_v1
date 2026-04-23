"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import apiClient from "@/lib/apiClient";
import {
  Send,
  Loader2,
  AlertCircle,
  Bot,
  User,
  Settings,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  Brain,
  Database,
  Layers,
  Search,
  Server,
  Shield,
  Zap,
  Bug,
  Trash2,
  Info,
  FileText,
} from "lucide-react";
import { cn } from "@/lib/utils";

/* ─── Types ─────────────────────────────────────────────────── */

interface LLMProvider {
  provider: string;
  display_name: string;
  default_model: string;
  api_key_set: boolean;
  api_key_env: string;
  recommended: boolean;
}

interface RerankerOption {
  name: string;
  description: string;
  requires_gpu: boolean;
  requires_api_key: boolean;
}

interface ChatResult {
  rank: number;
  chunk_id: string;
  text: string;
  score: number;
  trust_decision?: string | null;
  trust_state?: string | null;
}

interface ChatResponse {
  session_id: string;
  query: string;
  rewritten_query?: string | null;
  intent: string;
  search_mode: string;
  total_results: number;
  total_dropped: number;
  latency_ms: number;
  results: ChatResult[];
  answer?: string | null;
  answer_model?: string | null;
  answer_latency_ms?: number | null;
  answer_error?: string | null;
  debug_info?: Record<string, any> | null;
}

interface Turn {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  response?: ChatResponse | null;
  error?: string | null;
}

/* ─── Mini Components ───────────────────────────────────────── */

function TrustBadge({ state }: { state?: string | null }) {
  if (!state || state === "validated")
    return <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200">Trusted</span>;
  return <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-50 text-amber-700 border border-amber-200">Unvalidated</span>;
}

function DebugChip({ icon: Icon, label, value }: { icon: any; label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1 text-[11px] bg-white border border-slate-200 text-slate-700 px-2 py-1 rounded-md">
      <Icon className="w-3 h-3 text-slate-400" />
      <span className="text-slate-400">{label}:</span>
      <span className="font-semibold">{value}</span>
    </span>
  );
}

/* ─── Main Page ─────────────────────────────────────────────── */

export default function ChatRetrievePage() {
  const [sessionId] = useState(() => crypto.randomUUID());
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // Config state
  const [llmProviders, setLlmProviders] = useState<LLMProvider[]>([]);
  const [rerankers, setRerankers] = useState<RerankerOption[]>([]);
  const [selectedLLM, setSelectedLLM] = useState("");
  const [selectedReranker, setSelectedReranker] = useState("none");
  const [topK, setTopK] = useState(5);
  const [searchMode, setSearchMode] = useState("semantic");
  const [enableHyde, setEnableHyde] = useState(false);
  const [generateAnswer, setGenerateAnswer] = useState(true);
  const [showConfig, setShowConfig] = useState(true);
  const [expandedDebug, setExpandedDebug] = useState<Set<string>>(new Set());

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Fetch available models on mount
  useEffect(() => {
    apiClient.get<LLMProvider[]>("/api/v2/models/llm").then((r) => {
      setLlmProviders(r.data);
      const def = r.data.find((p) => p.recommended && p.api_key_set) || r.data.find((p) => p.api_key_set);
      if (def) setSelectedLLM(def.provider);
    }).catch(() => {});

    apiClient.get<RerankerOption[]>("/api/v2/models/reranker").then((r) => {
      setRerankers(r.data);
    }).catch(() => {});
  }, []);

  // Auto-scroll on new turns
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  const toggleDebug = useCallback((id: string) => {
    setExpandedDebug((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;

    const userTurn: Turn = { id: crypto.randomUUID(), role: "user", content: text, timestamp: Date.now() };
    const asstId = crypto.randomUUID();
    const asstTurn: Turn = { id: asstId, role: "assistant", content: "", timestamp: Date.now() };

    setTurns((prev) => [...prev, userTurn, asstTurn]);
    setInput("");
    setLoading(true);

    // Build message history for multi-turn
    const allTurns = [...turns, userTurn];
    const messages = allTurns
      .filter((t) => t.role === "user" || (t.role === "assistant" && t.response?.answer))
      .map((t) => ({
        role: t.role,
        content: t.role === "user" ? t.content : (t.response?.answer || t.content),
      }));

    try {
      const res = await apiClient.post<ChatResponse>("/api/v2/retrieve/chat", {
        session_id: sessionId,
        messages,
        llm_provider: selectedLLM || undefined,
        reranker: selectedReranker === "none" ? undefined : selectedReranker,
        intent: "answer",
        top_k: topK,
        search_mode: searchMode,
        enable_hyde: enableHyde,
        generate_answer: generateAnswer,
      });

      setTurns((prev) =>
        prev.map((t) =>
          t.id === asstId
            ? { ...t, content: res.data.answer || "No answer generated.", response: res.data }
            : t
        )
      );
    } catch (err: any) {
      const errMsg = err?.response?.data?.detail || err?.message || "Request failed";
      setTurns((prev) =>
        prev.map((t) => (t.id === asstId ? { ...t, content: "", error: errMsg } : t))
      );
    } finally {
      setLoading(false);
    }
  }, [input, loading, turns, sessionId, selectedLLM, selectedReranker, topK, searchMode, enableHyde, generateAnswer]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setTurns([]);
  };

  return (
    <div className="flex h-[calc(100vh-4rem)] overflow-hidden">
      {/* ═══ Left: Config Panel ═══ */}
      <div
        className={cn(
          "border-r border-slate-200 bg-slate-50/80 transition-all duration-300 flex flex-col",
          showConfig ? "w-72" : "w-0 overflow-hidden"
        )}
      >
        <div className="p-4 border-b border-slate-200 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-700">Session Config</h3>
          <button onClick={() => setShowConfig(false)} className="text-slate-400 hover:text-slate-600">
            <ChevronDown className="w-4 h-4 rotate-90" />
          </button>
        </div>

        <div className="p-4 space-y-4 overflow-y-auto flex-1">
          {/* LLM Provider */}
          <div>
            <label className="text-xs font-medium text-slate-500 block mb-1">LLM Provider</label>
            <select
              value={selectedLLM}
              onChange={(e) => setSelectedLLM(e.target.value)}
              className="w-full rounded-md border border-slate-200 px-2.5 py-1.5 text-xs text-slate-700 bg-white focus:border-primary-300 outline-none"
            >
              {llmProviders.map((p) => (
                <option key={p.provider} value={p.provider} disabled={!p.api_key_set}>
                  {p.display_name} {p.api_key_set ? `(${p.default_model})` : "(no API key)"}{p.recommended ? " *" : ""}
                </option>
              ))}
            </select>
          </div>

          {/* Reranker */}
          <div>
            <label className="text-xs font-medium text-slate-500 block mb-1">Reranker</label>
            <select
              value={selectedReranker}
              onChange={(e) => setSelectedReranker(e.target.value)}
              className="w-full rounded-md border border-slate-200 px-2.5 py-1.5 text-xs text-slate-700 bg-white focus:border-primary-300 outline-none"
            >
              {rerankers.map((r) => (
                <option key={r.name} value={r.name}>
                  {r.name === "none" ? "None" : r.name}
                </option>
              ))}
            </select>
          </div>

          {/* Top K */}
          <div>
            <label className="text-xs font-medium text-slate-500 block mb-1">Top K: {topK}</label>
            <input
              type="range" min={1} max={20} value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full h-1.5 accent-primary-600"
            />
          </div>

          {/* Search Mode */}
          <div>
            <label className="text-xs font-medium text-slate-500 block mb-1">Search Mode</label>
            <select
              value={searchMode}
              onChange={(e) => setSearchMode(e.target.value)}
              className="w-full rounded-md border border-slate-200 px-2.5 py-1.5 text-xs text-slate-700 bg-white focus:border-primary-300 outline-none"
            >
              <option value="semantic">Semantic</option>
              <option value="hybrid">Hybrid (Semantic + BM25)</option>
              <option value="keyword">Keyword (BM25)</option>
            </select>
          </div>

          {/* Toggles */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox" checked={enableHyde}
                onChange={(e) => setEnableHyde(e.target.checked)}
                className="rounded border-slate-300 text-primary-600 focus:ring-primary-500 w-3.5 h-3.5"
              />
              <span className="text-xs text-slate-600">HyDE Expansion</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox" checked={generateAnswer}
                onChange={(e) => setGenerateAnswer(e.target.checked)}
                className="rounded border-slate-300 text-primary-600 focus:ring-primary-500 w-3.5 h-3.5"
              />
              <span className="text-xs text-slate-600">Generate LLM Answer</span>
            </label>
          </div>

          {/* Embedder info */}
          <div className="rounded-lg bg-blue-50 border border-blue-200 p-3">
            <div className="flex items-center gap-1.5 text-[11px] text-blue-700 font-medium">
              <Info className="w-3.5 h-3.5" />
              Embedder Fixed
            </div>
            <p className="text-[10px] text-blue-600 mt-1">
              The embedder is always the one used for ingestion to keep vector indices compatible.
            </p>
          </div>
        </div>

        <div className="p-4 border-t border-slate-200">
          <button
            onClick={clearChat}
            className="w-full flex items-center justify-center gap-1.5 text-xs text-slate-500 hover:text-red-600 transition-colors py-1.5"
          >
            <Trash2 className="w-3.5 h-3.5" /> Clear conversation
          </button>
        </div>
      </div>

      {/* ═══ Main: Chat Area ═══ */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="h-14 border-b border-slate-200 flex items-center px-4 gap-3 bg-white">
          {!showConfig && (
            <button
              onClick={() => setShowConfig(true)}
              className="text-slate-400 hover:text-slate-600 p-1 rounded hover:bg-slate-100"
            >
              <Settings className="w-4 h-4" />
            </button>
          )}
          <Bot className="w-5 h-5 text-primary-500" />
          <h2 className="text-sm font-semibold text-slate-700">RAG Chat Console</h2>
          <span className="text-[10px] text-slate-400 ml-2">Multi-turn retrieval-augmented generation</span>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
          {turns.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <Bot className="w-12 h-12 text-slate-300 mb-4" />
              <h3 className="text-lg font-semibold text-slate-600 mb-1">Ask anything about your documents</h3>
              <p className="text-sm text-slate-400 max-w-md">
                Your questions are embedded and matched against ingested documents. Answers are grounded in retrieved context with citations.
              </p>
            </div>
          )}

          {turns.map((turn) => (
            <div key={turn.id} className={cn("flex gap-3", turn.role === "user" ? "justify-end" : "justify-start")}>
              {turn.role === "assistant" && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center">
                  <Bot className="w-4 h-4 text-primary-600" />
                </div>
              )}

              <div className={cn("max-w-[75%] rounded-2xl px-4 py-3", turn.role === "user"
                ? "bg-primary-600 text-white"
                : "bg-white border border-slate-200 shadow-sm"
              )}>
                {/* User message */}
                {turn.role === "user" && <p className="text-sm whitespace-pre-wrap">{turn.content}</p>}

                {/* Assistant message */}
                {turn.role === "assistant" && (
                  <div className="space-y-2">
                    {turn.error && (
                      <div className="flex items-center gap-2 text-red-600 text-xs">
                        <AlertCircle className="w-3.5 h-3.5" /> {turn.error}
                      </div>
                    )}

                    {!turn.error && !turn.response && loading && turn.id === turns[turns.length - 1]?.id && (
                      <div className="flex items-center gap-2 text-slate-400 text-xs">
                        <Loader2 className="w-3.5 h-3.5 animate-spin" /> Thinking...
                      </div>
                    )}

                    {turn.response?.answer && (
                      <p className="text-sm text-slate-700 whitespace-pre-wrap leading-relaxed">{turn.response.answer}</p>
                    )}

                    {turn.response?.answer_error && !turn.response?.answer && (
                      <div className="flex items-start gap-2 text-amber-700 text-xs bg-amber-50 rounded-lg p-2 border border-amber-200">
                        <AlertCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                        <span>{turn.response.answer_error}</span>
                      </div>
                    )}

                    {turn.response?.rewritten_query && (
                      <p className="text-[10px] text-slate-400 italic">
                        Rewritten query: &quot;{turn.response.rewritten_query}&quot;
                      </p>
                    )}

                    {/* Retrieved chunks (collapsed) */}
                    {turn.response && turn.response.results.length > 0 && (
                      <div className="mt-2">
                        <button
                          onClick={() => toggleDebug(turn.id)}
                          className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-slate-600"
                        >
                          <FileText className="w-3 h-3" />
                          {turn.response.results.length} sources | {turn.response.latency_ms.toFixed(0)}ms
                          {expandedDebug.has(turn.id) ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                        </button>

                        {expandedDebug.has(turn.id) && (
                          <div className="mt-2 space-y-2">
                            {/* Debug chips */}
                            {turn.response.debug_info && (
                              <div className="flex flex-wrap gap-1 mb-2">
                                <DebugChip icon={Brain} label="Embedder" value={turn.response.debug_info.embedder} />
                                <DebugChip icon={Bot} label="LLM" value={`${turn.response.debug_info.llm_provider}/${turn.response.debug_info.llm_model || ""}`} />
                                <DebugChip icon={Layers} label="Reranker" value={turn.response.debug_info.reranker_used || "none"} />
                                <DebugChip icon={Search} label="Search" value={turn.response.debug_info.search_mode} />
                                <DebugChip icon={Database} label="VectorDB" value={turn.response.debug_info.vectordb} />
                                {turn.response.debug_info.query_rewritten && (
                                  <DebugChip icon={Zap} label="Rewrite" value="yes" />
                                )}
                              </div>
                            )}

                            {/* Sources */}
                            {turn.response.results.map((r) => (
                              <div
                                key={r.chunk_id}
                                className="rounded-lg border border-slate-200 bg-slate-50/80 p-2.5"
                              >
                                <div className="flex items-center gap-2 mb-1">
                                  <span className="text-[10px] font-semibold text-slate-500">#{r.rank}</span>
                                  <span className={cn("text-[10px] font-mono", r.score >= 0.5 ? "text-emerald-600" : "text-amber-600")}>
                                    {r.score.toFixed(4)}
                                  </span>
                                  <TrustBadge state={r.trust_state} />
                                </div>
                                <p className="text-[11px] text-slate-600 line-clamp-3 leading-relaxed">
                                  {r.text}
                                </p>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {turn.role === "user" && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center">
                  <User className="w-4 h-4 text-slate-600" />
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-slate-200 bg-white p-4">
          <div className="flex items-end gap-3 max-w-4xl mx-auto">
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question... (Enter to send, Shift+Enter for newline)"
                rows={1}
                className="w-full resize-none rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 placeholder:text-slate-400 focus:border-primary-300 focus:ring-2 focus:ring-primary-100 outline-none transition-all"
                style={{ minHeight: "44px", maxHeight: "120px" }}
                onInput={(e) => {
                  const el = e.target as HTMLTextAreaElement;
                  el.style.height = "auto";
                  el.style.height = Math.min(el.scrollHeight, 120) + "px";
                }}
              />
            </div>
            <button
              onClick={sendMessage}
              disabled={!input.trim() || loading}
              className={cn(
                "flex items-center justify-center rounded-xl w-11 h-11 transition-all",
                input.trim() && !loading
                  ? "bg-primary-600 text-white hover:bg-primary-700 shadow-sm"
                  : "bg-slate-100 text-slate-400 cursor-not-allowed"
              )}
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
