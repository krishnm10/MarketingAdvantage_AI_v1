"use client";

import { useEffect, useState, useCallback } from "react";
import {
  SlidersHorizontal, Zap, Filter, ShieldCheck, Brain, ChevronDown,
  ChevronUp, RefreshCw, CheckCircle2, XCircle, Info, Save, AlertTriangle,
  Loader2, BarChart3, Layers, Search, Hash, Cpu, Cloud,
} from "lucide-react";
import { cn } from "@/lib/utils";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";

/* ─── Types ─── */
interface RerankerEntry {
  model_id: string;
  provider: string;
  display_name: string;
  max_input_tokens_per_pair: number;
  score_space: string;
  supports_batch_scoring: boolean;
  lang_support: string[];
  tier: string;
  requires_gpu: boolean;
  default_top_k: number;
  notes_short: string;
  env_key: string | null;
}

interface QueryTransform {
  strategy: string;
  display_name: string;
  description: string;
  requires_llm: boolean;
  recall_impact: string;
}

interface RuleEntry {
  rule_name: string;
  technical_implementation: string;
  goal: string;
  pipeline_stage: string;
  config_keys: string[];
}

interface PipelineConfig {
  client_id: string;
  is_default?: boolean;
  message?: string;
  retrieval: {
    search_mode: string;
    top_k_retrieval: number;
    top_k_final: number;
    hybrid_alpha: number;
    enable_hyde: boolean;
    enable_multi_query: boolean;
    multi_query_count: number;
    enable_threshold_gate: boolean;
    threshold_min_score: number;
    threshold_min_results: number;
    enable_token_budget: boolean;
    token_budget_context_fraction: number;
    prompt_template_id: string | null;
  };
  reranker: {
    type: string | null;
    model: string | null;
    top_k: number;
    device: string;
  } | null;
}

/* ─── Infer reranker metadata from catalog model_id ─── */
function inferRerankerType(modelId: string | null): { type: string; provider?: string } {
  if (!modelId) return { type: "crossencoder" };
  if (modelId.startsWith("llm-judge/")) {
    if (modelId.includes("gemini")) return { type: "llm_judge", provider: "gemini" };
    return { type: "llm_judge", provider: "openai" };
  }
  if (modelId.startsWith("cohere/")) return { type: "cohere" };
  if (modelId.startsWith("BAAI/bge-reranker")) return { type: "bge_reranker" };
  if (modelId.startsWith("flashrank/")) return { type: "flashrank" };
  if (modelId.startsWith("colbert/")) return { type: "colbert" };
  return { type: "crossencoder" };
}

/* ─── Provider badge color ─── */
const PROVIDER_COLORS: Record<string, string> = {
  gemini:     "bg-blue-500/15 text-blue-300 border border-blue-500/30",
  openai_llm: "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20",
  huggingface:"bg-orange-500/10 text-orange-400 border border-orange-500/20",
  cohere:     "bg-violet-500/10 text-violet-400 border border-violet-500/20",
  flashrank:  "bg-cyan-500/10 text-cyan-400 border border-cyan-500/20",
  colbert:    "bg-rose-500/10 text-rose-400 border border-rose-500/20",
};

/* ─── Stage badges ─── */
const STAGE_COLORS: Record<string, string> = {
  "pre-retrieval":              "bg-blue-500/10 text-blue-400 border border-blue-500/20",
  "retrieval":                  "bg-violet-500/10 text-violet-400 border border-violet-500/20",
  "reranking":                  "bg-amber-500/10 text-amber-400 border border-amber-500/20",
  "post-rerank":                "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20",
  "post-generation":            "bg-rose-500/10 text-rose-400 border border-rose-500/20",
  "reranking or post-generation": "bg-amber-500/10 text-amber-400 border border-amber-500/20",
  "post-rerank context assembly": "bg-teal-500/10 text-teal-400 border border-teal-500/20",
};

const RECALL_COLORS: Record<string, string> = {
  baseline: "text-slate-400",
  medium:   "text-yellow-400",
  high:     "text-emerald-400",
};

export default function RerankingPage() {
  const [rerankers,       setRerankers]       = useState<RerankerEntry[]>([]);
  const [transforms,      setTransforms]      = useState<QueryTransform[]>([]);
  const [rules,           setRules]           = useState<RuleEntry[]>([]);
  const [pipelineConfig,  setPipelineConfig]  = useState<PipelineConfig | null>(null);
  const [clientId,        setClientId]        = useState("default");
  const [loading,         setLoading]         = useState(true);
  const [saving,          setSaving]          = useState(false);
  const [saveStatus,      setSaveStatus]      = useState<"idle"|"ok"|"err">("idle");
  const [expandedRule,    setExpandedRule]    = useState<string | null>(null);
  const [activeTab,       setActiveTab]       = useState<"config"|"rules"|"catalog">("config");

  /* ─── Local editable state ─── */
  const [enableHyde,         setEnableHyde]         = useState(false);
  const [enableMultiQuery,   setEnableMultiQuery]   = useState(false);
  const [multiQueryCount,    setMultiQueryCount]    = useState(3);
  const [enableThreshold,    setEnableThreshold]    = useState(false);
  const [thresholdScore,     setThresholdScore]     = useState(0.0);
  const [thresholdMinResults,setThresholdMinResults]= useState(1);
  const [enableTokenBudget,  setEnableTokenBudget]  = useState(true);
  const [budgetFraction,     setBudgetFraction]     = useState(0.6);
  const [searchMode,         setSearchMode]         = useState("semantic");
  const [hybridAlpha,        setHybridAlpha]        = useState(0.7);
  const [topKRetrieval,      setTopKRetrieval]      = useState(20);
  const [topKFinal,          setTopKFinal]          = useState(5);
  const [rerankerModel,      setRerankerModel]      = useState<string | null>(null);
  const [rerankerTopK,       setRerankerTopK]       = useState(5);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [r, t, ru] = await Promise.all([
        apiClient.get(API.RAG_CONFIG.RERANKERS()),
        apiClient.get(API.RAG_CONFIG.QUERY_TRANSFORMS()),
        apiClient.get(API.RAG_CONFIG.RERANKER_RULES()),
      ]);
      setRerankers(r.data);
      setTransforms(t.data);
      setRules(ru.data);

      const cfg = await apiClient.get(API.RAG_CONFIG.GET_PIPELINE(clientId));
      const d = cfg.data as PipelineConfig;
      setPipelineConfig(d);
      setEnableHyde(d.retrieval.enable_hyde);
      setEnableMultiQuery(d.retrieval.enable_multi_query);
      setMultiQueryCount(d.retrieval.multi_query_count);
      setEnableThreshold(d.retrieval.enable_threshold_gate);
      setThresholdScore(d.retrieval.threshold_min_score);
      setThresholdMinResults(d.retrieval.threshold_min_results);
      setEnableTokenBudget(d.retrieval.enable_token_budget);
      setBudgetFraction(d.retrieval.token_budget_context_fraction);
      setSearchMode(d.retrieval.search_mode);
      setHybridAlpha(d.retrieval.hybrid_alpha);
      setTopKRetrieval(d.retrieval.top_k_retrieval);
      setTopKFinal(d.retrieval.top_k_final);
      if (d.reranker) {
        setRerankerModel(d.reranker.model);
        setRerankerTopK(d.reranker.top_k);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [clientId]);

  useEffect(() => { load(); }, [load]);

  const handleSave = async () => {
    setSaving(true);
    setSaveStatus("idle");
    try {
      const inferred = inferRerankerType(rerankerModel);
      await apiClient.put(API.RAG_CONFIG.PUT_PIPELINE(clientId), {
        enable_hyde:           enableHyde,
        enable_multi_query:    enableMultiQuery,
        multi_query_count:     multiQueryCount,
        enable_threshold_gate: enableThreshold,
        threshold_min_score:   thresholdScore,
        threshold_min_results: thresholdMinResults,
        enable_token_budget:   enableTokenBudget,
        token_budget_fraction: budgetFraction,
        search_mode:           searchMode,
        hybrid_alpha:          hybridAlpha,
        reranker_model_id:     rerankerModel,
        reranker_top_k:        rerankerTopK,
        reranker_type:         rerankerModel ? inferred.type : null,
        ...(inferred.provider ? { judge_provider: inferred.provider } : {}),
      });
      setSaveStatus("ok");
      // Re-fetch so the UI reflects what is now persisted on disk
      await load();
    } catch {
      setSaveStatus("err");
    } finally {
      setSaving(false);
      setTimeout(() => setSaveStatus("idle"), 3000);
    }
  };

  if (loading) {
    return (
      <div className="flex h-80 items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary-400" />
        <span className="ml-3 text-slate-400">Loading reranking configuration…</span>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <SlidersHorizontal className="h-6 w-6 text-primary-400" />
            Reranking & Retrieval Rules
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Configure post-retrieval reranking, query transforms, and context assembly rules for your RAG pipeline.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <input
            value={clientId}
            onChange={e => setClientId(e.target.value)}
            placeholder="client_id"
            className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:border-primary-500 focus:outline-none"
          />
          <button
            onClick={load}
            className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-700 transition-colors"
          >
            <RefreshCw className="h-3.5 w-3.5" /> Reload
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className={cn(
              "flex items-center gap-1.5 rounded-lg px-4 py-1.5 text-sm font-medium transition-colors",
              saving
                ? "bg-slate-700 text-slate-400 cursor-not-allowed"
                : "bg-primary-600 text-white hover:bg-primary-700"
            )}
          >
            {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
            {saving ? "Saving…" : "Save"}
          </button>
          {saveStatus === "ok"  && <CheckCircle2 className="h-5 w-5 text-emerald-400" />}
          {saveStatus === "err" && <XCircle      className="h-5 w-5 text-rose-400" />}
        </div>
      </div>

      {/* Default-config notice */}
      {pipelineConfig?.is_default && (
        <div className="flex items-center gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-300">
          <AlertTriangle className="h-4 w-4 flex-shrink-0" />
          <span>
            No saved config found for <span className="font-mono text-amber-200">{clientId}</span> — showing defaults.
            Save once to persist your selections to disk.
          </span>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 rounded-xl bg-slate-800/50 p-1 w-fit">
        {(["config", "rules", "catalog"] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={cn(
              "rounded-lg px-4 py-1.5 text-sm font-medium capitalize transition-colors",
              activeTab === tab
                ? "bg-primary-600 text-white shadow"
                : "text-slate-400 hover:text-slate-200"
            )}
          >
            {tab === "config"  ? "Pipeline Config" : tab === "rules" ? "Rules Reference" : "Reranker Catalog"}
          </button>
        ))}
      </div>

      {/* ── TAB: Pipeline Config ── */}
      {activeTab === "config" && (
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">

          {/* Retrieval Settings */}
          <div className="rounded-xl border border-slate-700/50 bg-slate-900 p-5 space-y-4">
            <h2 className="text-sm font-semibold text-white flex items-center gap-2">
              <Search className="h-4 w-4 text-violet-400" /> Retrieval Settings
            </h2>
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-slate-400 mb-1">Search Mode</label>
                <select
                  value={searchMode}
                  onChange={e => setSearchMode(e.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-white focus:border-primary-500 focus:outline-none"
                >
                  <option value="semantic">Semantic (Vector Only)</option>
                  <option value="hybrid">Hybrid (Vector + BM25 + RRF)</option>
                  <option value="keyword">Keyword (BM25 Only)</option>
                </select>
              </div>
              {searchMode === "hybrid" && (
                <div>
                  <label className="block text-xs text-slate-400 mb-1">
                    Hybrid Alpha — Semantic Weight: <span className="text-white">{hybridAlpha}</span>
                  </label>
                  <input
                    type="range" min={0} max={1} step={0.05}
                    value={hybridAlpha}
                    onChange={e => setHybridAlpha(parseFloat(e.target.value))}
                    className="w-full accent-primary-500"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
                    <span>0.0 = BM25 only</span>
                    <span>1.0 = Vector only</span>
                  </div>
                </div>
              )}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Top-K Retrieval</label>
                  <input
                    type="number" min={5} max={100}
                    value={topKRetrieval}
                    onChange={e => setTopKRetrieval(parseInt(e.target.value))}
                    className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-white focus:border-primary-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Top-K Final</label>
                  <input
                    type="number" min={1} max={20}
                    value={topKFinal}
                    onChange={e => setTopKFinal(parseInt(e.target.value))}
                    className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-white focus:border-primary-500 focus:outline-none"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Reranker Selection */}
          <div className="rounded-xl border border-slate-700/50 bg-slate-900 p-5 space-y-4">
            <h2 className="text-sm font-semibold text-white flex items-center gap-2">
              <Layers className="h-4 w-4 text-amber-400" /> Reranker / LLM-as-Judge
            </h2>
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-slate-400 mb-1">Reranker Model</label>
                <select
                  value={rerankerModel ?? ""}
                  onChange={e => setRerankerModel(e.target.value || null)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-white focus:border-primary-500 focus:outline-none"
                >
                  <option value="">— None (skip reranking) —</option>
                  {(["local", "api"] as const).map(tier => {
                    const group = rerankers.filter(r => r.tier === tier);
                    if (!group.length) return null;
                    return (
                      <optgroup key={tier} label={tier === "local" ? "Local / GPU Models" : "API-based (Cloud)"}>
                        {group.map(r => (
                          <option key={r.model_id} value={r.model_id}>
                            {r.display_name} {r.requires_gpu ? "⚡GPU" : ""}
                            {r.provider === "gemini" ? " [Gemini]" :
                             r.provider === "openai_llm" ? " [OpenAI LLM]" : ""}
                          </option>
                        ))}
                      </optgroup>
                    );
                  })}
                </select>
              </div>
              {rerankerModel && (
                <>
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Reranker Top-K</label>
                    <input
                      type="number" min={1} max={20}
                      value={rerankerTopK}
                      onChange={e => setRerankerTopK(parseInt(e.target.value))}
                      className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-white focus:border-primary-500 focus:outline-none"
                    />
                  </div>
                  {(() => {
                    const r = rerankers.find(x => x.model_id === rerankerModel);
                    return r ? (
                      <div className="rounded-lg bg-slate-800 p-3 text-xs text-slate-400 space-y-1">
                        <p><span className="text-slate-300">Score space:</span> {r.score_space}</p>
                        <p><span className="text-slate-300">Max tokens/pair:</span> {r.max_input_tokens_per_pair}</p>
                        <p><span className="text-slate-300">Languages:</span> {r.lang_support.join(", ")}</p>
                        <p className="text-slate-500 italic">{r.notes_short}</p>
                      </div>
                    ) : null;
                  })()}
                </>
              )}
            </div>
          </div>

          {/* Query Transforms */}
          <div className="rounded-xl border border-slate-700/50 bg-slate-900 p-5 space-y-4">
            <h2 className="text-sm font-semibold text-white flex items-center gap-2">
              <Brain className="h-4 w-4 text-blue-400" /> Query Transforms (Pre-Retrieval)
            </h2>
            <div className="space-y-4">
              {/* HyDE */}
              <div className="flex items-start gap-3">
                <button
                  onClick={() => setEnableHyde(!enableHyde)}
                  className={cn(
                    "mt-0.5 h-5 w-9 flex-shrink-0 rounded-full transition-colors relative",
                    enableHyde ? "bg-primary-600" : "bg-slate-700"
                  )}
                >
                  <span className={cn(
                    "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform",
                    enableHyde ? "left-4" : "left-0.5"
                  )} />
                </button>
                <div>
                  <p className="text-sm font-medium text-white">HyDE — Hypothetical Document Embeddings</p>
                  <p className="text-xs text-slate-400 mt-0.5">
                    LLM generates a hypothetical answer; its embedding is used instead of the raw query.
                    Bridges vocabulary gap. Requires LLM.
                  </p>
                </div>
              </div>
              {/* Multi-Query */}
              <div className="flex items-start gap-3">
                <button
                  onClick={() => setEnableMultiQuery(!enableMultiQuery)}
                  className={cn(
                    "mt-0.5 h-5 w-9 flex-shrink-0 rounded-full transition-colors relative",
                    enableMultiQuery ? "bg-primary-600" : "bg-slate-700"
                  )}
                >
                  <span className={cn(
                    "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform",
                    enableMultiQuery ? "left-4" : "left-0.5"
                  )} />
                </button>
                <div className="flex-1">
                  <p className="text-sm font-medium text-white">Multi-Query Expansion + RRF</p>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Generate {multiQueryCount} diverse query variants; retrieve for each; merge via RRF.
                  </p>
                  {enableMultiQuery && (
                    <div className="mt-2">
                      <label className="text-xs text-slate-400">Variants: <span className="text-white">{multiQueryCount}</span></label>
                      <input
                        type="range" min={2} max={8}
                        value={multiQueryCount}
                        onChange={e => setMultiQueryCount(parseInt(e.target.value))}
                        className="w-full mt-1 accent-primary-500"
                      />
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Post-Processing Rules */}
          <div className="rounded-xl border border-slate-700/50 bg-slate-900 p-5 space-y-4">
            <h2 className="text-sm font-semibold text-white flex items-center gap-2">
              <Filter className="h-4 w-4 text-emerald-400" /> Post-Processing Rules
            </h2>
            <div className="space-y-4">
              {/* Threshold Gate */}
              <div>
                <div className="flex items-start gap-3">
                  <button
                    onClick={() => setEnableThreshold(!enableThreshold)}
                    className={cn(
                      "mt-0.5 h-5 w-9 flex-shrink-0 rounded-full transition-colors relative",
                      enableThreshold ? "bg-primary-600" : "bg-slate-700"
                    )}
                  >
                    <span className={cn(
                      "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform",
                      enableThreshold ? "left-4" : "left-0.5"
                    )} />
                  </button>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-white">Threshold Gate</p>
                    <p className="text-xs text-slate-400 mt-0.5">Filter candidates below calibrated rerank score.</p>
                  </div>
                </div>
                {enableThreshold && (
                  <div className="mt-3 ml-12 space-y-2">
                    <div>
                      <label className="text-xs text-slate-400">
                        Min Score: <span className="text-white">{thresholdScore.toFixed(2)}</span>
                        <span className="text-slate-500 ml-1">(calibrate from eval data)</span>
                      </label>
                      <input
                        type="range" min={0} max={1} step={0.01}
                        value={thresholdScore}
                        onChange={e => setThresholdScore(parseFloat(e.target.value))}
                        className="w-full mt-1 accent-primary-500"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-slate-400">Safety Floor (min results)</label>
                      <input
                        type="number" min={1} max={10}
                        value={thresholdMinResults}
                        onChange={e => setThresholdMinResults(parseInt(e.target.value))}
                        className="mt-1 w-24 rounded border border-slate-700 bg-slate-800 px-2 py-1 text-sm text-white focus:border-primary-500 focus:outline-none"
                      />
                    </div>
                  </div>
                )}
              </div>
              {/* Token Budget */}
              <div>
                <div className="flex items-start gap-3">
                  <button
                    onClick={() => setEnableTokenBudget(!enableTokenBudget)}
                    className={cn(
                      "mt-0.5 h-5 w-9 flex-shrink-0 rounded-full transition-colors relative",
                      enableTokenBudget ? "bg-primary-600" : "bg-slate-700"
                    )}
                  >
                    <span className={cn(
                      "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform",
                      enableTokenBudget ? "left-4" : "left-0.5"
                    )} />
                  </button>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-white">Adaptive Token Budget</p>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Trim context to fit within the generator's context window.
                    </p>
                  </div>
                </div>
                {enableTokenBudget && (
                  <div className="mt-3 ml-12">
                    <label className="text-xs text-slate-400">
                      Context Fraction: <span className="text-white">{Math.round(budgetFraction * 100)}%</span>
                      <span className="text-slate-500 ml-1">of context window</span>
                    </label>
                    <input
                      type="range" min={0.1} max={0.95} step={0.05}
                      value={budgetFraction}
                      onChange={e => setBudgetFraction(parseFloat(e.target.value))}
                      className="w-full mt-1 accent-primary-500"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
                      <span>10% context</span>
                      <span>95% context</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB: Rules Reference ── */}
      {activeTab === "rules" && (
        <div className="overflow-hidden rounded-xl border border-slate-700/50">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-700 bg-slate-800/60">
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">Rule</th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">Implementation</th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">Goal</th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">Stage</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/50">
              {rules.map((rule) => (
                <tr key={rule.rule_name} className="bg-slate-900 hover:bg-slate-800/50 transition-colors">
                  <td className="px-4 py-3 font-medium text-white whitespace-nowrap text-sm">{rule.rule_name}</td>
                  <td className="px-4 py-3 text-xs text-slate-400 max-w-xs">{rule.technical_implementation}</td>
                  <td className="px-4 py-3 text-xs text-slate-300 max-w-xs">{rule.goal}</td>
                  <td className="px-4 py-3">
                    <span className={cn(
                      "rounded-full px-2 py-0.5 text-[10px] font-medium whitespace-nowrap",
                      STAGE_COLORS[rule.pipeline_stage] ?? "bg-slate-700 text-slate-300"
                    )}>
                      {rule.pipeline_stage}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── TAB: Reranker Catalog ── */}
      {activeTab === "catalog" && (
        <div className="space-y-6">
          {(["local", "api"] as const).map(tier => {
            const group = rerankers.filter(r => r.tier === tier);
            if (!group.length) return null;
            return (
              <div key={tier}>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3 flex items-center gap-1.5">
                  {tier === "local" ? <Cpu className="h-3.5 w-3.5" /> : <Cloud className="h-3.5 w-3.5" />}
                  {tier === "local" ? "Local / On-Premise" : "Cloud API"}
                </h3>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                  {group.map((r) => (
                    <div key={r.model_id}
                      className={cn(
                        "rounded-xl border bg-slate-900 p-4 space-y-2 transition-colors hover:border-primary-500/40",
                        rerankerModel === r.model_id
                          ? "border-primary-500/60 ring-1 ring-primary-500/30"
                          : "border-slate-700/50"
                      )}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <p className="text-sm font-semibold text-white">{r.display_name}</p>
                          <p className="text-[10px] text-slate-500 font-mono mt-0.5">{r.model_id}</p>
                        </div>
                        <div className="flex flex-col gap-1 items-end">
                          {/* Provider badge */}
                          {PROVIDER_COLORS[r.provider] ? (
                            <span className={cn("rounded-full px-2 py-0.5 text-[10px] font-medium", PROVIDER_COLORS[r.provider])}>
                              {r.provider}
                            </span>
                          ) : (
                            <span className={cn(
                              "rounded-full px-2 py-0.5 text-[10px] font-medium",
                              r.tier === "api"   ? "bg-blue-500/10 text-blue-400"   :
                              r.tier === "local" ? "bg-emerald-500/10 text-emerald-400" :
                                                  "bg-slate-700 text-slate-300"
                            )}>
                              {r.tier === "api" ? <Cloud className="inline h-2.5 w-2.5 mr-0.5" /> : <Cpu className="inline h-2.5 w-2.5 mr-0.5" />}
                              {r.tier}
                            </span>
                          )}
                          {r.requires_gpu && (
                            <span className="rounded-full bg-amber-500/10 text-amber-400 px-2 py-0.5 text-[10px] font-medium">GPU</span>
                          )}
                          {rerankerModel === r.model_id && (
                            <span className="rounded-full bg-primary-500/20 text-primary-300 px-2 py-0.5 text-[10px] font-medium flex items-center gap-1">
                              <CheckCircle2 className="h-2.5 w-2.5" /> Active
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-x-4 text-xs text-slate-400">
                        <span>Score: <span className="text-white">{r.score_space}</span></span>
                        <span>Tokens/pair: <span className="text-white">{r.max_input_tokens_per_pair}</span></span>
                        <span>Batch: <span className={r.supports_batch_scoring ? "text-emerald-400" : "text-slate-500"}>{r.supports_batch_scoring ? "Yes" : "No"}</span></span>
                        <span>Top-K: <span className="text-white">{r.default_top_k}</span></span>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {r.lang_support.map(l => (
                          <span key={l} className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-400">{l}</span>
                        ))}
                      </div>
                      <p className="text-[11px] text-slate-500 italic">{r.notes_short}</p>
                      {r.env_key && (
                        <p className="text-[10px] text-slate-600 font-mono">
                          Env: <span className="text-slate-500">{r.env_key}</span>
                        </p>
                      )}
                      <button
                        onClick={() => { setRerankerModel(r.model_id); setRerankerTopK(r.default_top_k); setActiveTab("config"); }}
                        className={cn(
                          "w-full rounded-lg py-1.5 text-xs font-medium transition-colors",
                          rerankerModel === r.model_id
                            ? "bg-primary-600/20 text-primary-300 hover:bg-primary-600/30"
                            : "bg-slate-800 text-primary-400 hover:bg-slate-700 hover:text-primary-300"
                        )}
                      >
                        {rerankerModel === r.model_id ? "Selected ✓" : "Use this reranker →"}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
