"use client";

import Link from "next/link";
import { useState, useRef, useEffect, useCallback, useMemo, type ComponentType, type ReactNode, type HTMLAttributes } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize from "rehype-sanitize";
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
  Building2,
  Sparkles,
  ImageIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { pipelineSettingsHref } from "@/lib/forbiddenEnvConfigKeys";
import { useTenant } from "@/contexts/TenantContext";
import { API } from "@/lib/apiRoutes";
import { useAuth } from "@/lib/useAuth";
import type { EffectiveTenantRuntime } from "@/lib/effectiveTenantRuntime";
import { parseChatDebugInfo } from "@/lib/chatDebugInfo";
import { EffectiveThisTurnStrip } from "@/components/runtime/EffectiveThisTurnStrip";

/** RAG chat can chain rewrite + HyDE + hybrid retrieve + answer; default apiClient 30s trips first on slow local LLMs. */
const CHAT_RETRIEVE_TIMEOUT_MS = 5 * 60 * 1000;

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
  debug_info?: Record<string, unknown> | null;
}

interface Turn {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  response?: ChatResponse | null;
  error?: string | null;
}

interface PromptTemplateListItem {
  template_id: string;
  name: string;
  description?: string;
  has_examples: boolean;
}

interface TenantPromptConfig {
  client_id: string;
  prompt_template_id: string | null;
  rewrite_enabled: boolean;
}

interface PromptTemplatePreview {
  template_id: string;
  name: string;
  system_instructions: string;
  has_examples: boolean;
  examples_note?: string;
}

/* ─── Mini Components ───────────────────────────────────────── */

function TrustBadge({ state }: { state?: string | null }) {
  if (!state || state === "validated")
    return <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200">Trusted</span>;
  return <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-50 text-amber-700 border border-amber-200">Unvalidated</span>;
}

function DebugChip({
  icon: Icon,
  label,
  value,
}: {
  icon: ComponentType<{ className?: string }>;
  label: string;
  value: string;
}) {
  return (
    <span className="inline-flex items-center gap-1 text-[11px] bg-white border border-slate-200 text-slate-700 px-2 py-1 rounded-md">
      <Icon className="w-3 h-3 text-slate-400" />
      <span className="text-slate-400">{label}:</span>
      <span className="font-semibold">{value}</span>
    </span>
  );
}

/** Map API provider id to dropdown option value (global catalog uses gemini not google). */
function normalizeLlmProvider(provider: string): string {
  const p = (provider || "").trim().toLowerCase();
  if (p === "google") return "gemini";
  return p || "none";
}

function applyRuntimeToSession(
  rt: EffectiveTenantRuntime,
  setters: {
    setTenantRuntime: (v: EffectiveTenantRuntime) => void;
    setRuntimeWarnings: (v: string[]) => void;
    setSelectedLLM: (v: string) => void;
    setSelectedReranker: (v: string) => void;
    setTopK: (v: number) => void;
    setSearchMode: (v: string) => void;
    setEnableHyde: (v: boolean) => void;
  }
) {
  setters.setTenantRuntime(rt);
  setters.setRuntimeWarnings(rt.warnings ?? []);
  setters.setSelectedLLM(normalizeLlmProvider(rt.llm.effective_provider));
  setters.setSelectedReranker(rt.reranker.effective_plugin || "none");
  setters.setTopK(rt.retrieval.top_k_final);
  setters.setSearchMode(rt.retrieval.search_mode);
  setters.setEnableHyde(rt.retrieval.enable_hyde);
}

/** Wrap [Source N] citations in subtle chips (paragraphs/lists only, not tables). */
function decorateSourceCitations(node: ReactNode): ReactNode {
  if (typeof node === "string") {
    const parts = node.split(/(\[Source \d+\])/gi);
    if (parts.length === 1) return node;
    return parts.map((part, i) =>
      /^\[Source \d+\]$/i.test(part) ? (
        <span
          key={i}
          className="inline-flex items-center mx-0.5 px-1.5 py-0.5 rounded-md bg-primary-50 text-primary-700 text-[10px] font-semibold ring-1 ring-primary-200/60 align-middle"
        >
          {part}
        </span>
      ) : (
        part
      )
    );
  }
  if (Array.isArray(node)) {
    return node.map((child, i) => (
      <span key={i}>{decorateSourceCitations(child)}</span>
    ));
  }
  return node;
}

function JsonImageConceptCard({
  payload,
}: {
  payload: { engine?: string; action?: string; prompt: string };
}) {
  const [expanded, setExpanded] = useState(false);
  const engineLabel = payload.engine || "Image engine";
  const actionLabel = payload.action || "text2im";
  const prompt = payload.prompt || "";

  return (
    <div className="my-4 rounded-xl border border-slate-200/80 bg-gradient-to-b from-white to-slate-50/80 shadow-card ring-1 ring-primary-100/80 overflow-hidden">
      <div className="flex items-center justify-between gap-2 px-4 py-3 border-b border-slate-100 bg-white/90">
        <div className="flex items-center gap-2 min-w-0">
          <span className="flex-shrink-0 w-7 h-7 rounded-lg bg-primary-50 border border-primary-100 flex items-center justify-center">
            <Sparkles className="w-3.5 h-3.5 text-primary-600" />
          </span>
          <div className="min-w-0">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-700">
              AI Generated Visual Concept
            </div>
            <div className="text-[10px] text-slate-400 truncate">
              {engineLabel} • {actionLabel}
            </div>
          </div>
        </div>
      </div>
      {prompt && (
        <div className="px-4 pt-3">
          <p
            className={cn(
              "text-xs text-slate-600 leading-relaxed",
              !expanded && "line-clamp-3"
            )}
          >
            {prompt}
          </p>
          {prompt.length > 120 && (
            <button
              type="button"
              onClick={() => setExpanded((v) => !v)}
              className="mt-1 text-[10px] font-medium text-primary-600 hover:text-primary-700"
            >
              {expanded ? "Show less" : "Show full brief"}
            </button>
          )}
        </div>
      )}
      <div className="mx-4 my-3 h-36 rounded-lg border border-dashed border-slate-200 bg-[linear-gradient(to_right,#f1f5f9_1px,transparent_1px),linear-gradient(to_bottom,#f1f5f9_1px,transparent_1px)] bg-[size:16px_16px] flex flex-col items-center justify-center gap-2">
        <ImageIcon className="w-5 h-5 text-slate-300" />
        <span className="text-[11px] text-slate-400 text-center px-4">
          Visual concept preview — image engine not connected
        </span>
      </div>
    </div>
  );
}

function AssistantMarkdown({ content }: { content: string }) {
  return (
    <div className="rounded-xl bg-gradient-to-b from-slate-50/80 to-white px-2 py-2 sm:px-3 animate-fade-in">
      <div className="text-sm text-slate-700 leading-relaxed max-w-none space-y-0.5">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeSanitize]}
          components={{
            h1: ({ children, ...props }) => (
              <h1
                className="text-lg font-bold text-slate-900 mt-4 mb-2 first:mt-0 border-b border-slate-200 pb-2"
                {...props}
              >
                {children}
              </h1>
            ),
            h2: ({ children, ...props }) => (
              <h2
                className="text-base font-semibold text-slate-900 mt-5 mb-2 first:mt-0 pl-3 border-l-4 border-primary-500"
                {...props}
              >
                {children}
              </h2>
            ),
            h3: ({ children, ...props }) => (
              <h3
                className="text-sm font-semibold text-slate-800 mt-4 mb-1.5 first:mt-0"
                {...props}
              >
                {children}
              </h3>
            ),
            h4: ({ children, ...props }) => (
              <h4
                className="text-xs font-semibold uppercase tracking-wide text-slate-500 mt-3 mb-1 first:mt-0"
                {...props}
              >
                {children}
              </h4>
            ),
            p: ({ children, ...props }) => (
              <p className="text-sm leading-relaxed text-slate-700 my-2" {...props}>
                {decorateSourceCitations(children)}
              </p>
            ),
            strong: ({ children, ...props }) => (
              <strong className="font-semibold text-slate-900" {...props}>
                {children}
              </strong>
            ),
            em: ({ children, ...props }) => (
              <em className="italic text-slate-600" {...props}>
                {children}
              </em>
            ),
            ul: ({ children, ...props }) => (
              <ul className="my-2 space-y-1.5 pl-1 list-none [&_ul]:mt-1.5 [&_ul]:ml-3 [&_ul_li]:before:w-1 [&_ul_li]:before:h-1 [&_ul_li]:before:bg-primary-400" {...props}>
                {children}
              </ul>
            ),
            ol: ({ children, ...props }) => (
              <ol
                className="my-2 space-y-1.5 pl-5 list-decimal marker:text-primary-600 marker:font-semibold [&>li]:pl-0 [&>li]:before:content-none"
                {...props}
              >
                {children}
              </ol>
            ),
            li: ({ children, ...props }) => (
              <li
                className="relative pl-5 text-sm leading-relaxed text-slate-700 before:content-[''] before:absolute before:left-0 before:top-[0.6em] before:w-1.5 before:h-1.5 before:rounded-full before:bg-primary-500 before:ring-2 before:ring-primary-100 [&>ol]:list-decimal [&>ol]:pl-5 [&>ol]:mt-1.5 [&>ol>li]:pl-0 [&>ol>li]:before:content-none"
                {...props}
              >
                {decorateSourceCitations(children)}
              </li>
            ),
            hr: (props) => (
              <hr className="my-5 border-0 border-t border-slate-200/90" {...props} />
            ),
            blockquote: ({ children, ...props }) => (
              <blockquote
                className="my-3 border-l-4 border-amber-300 bg-amber-50/60 rounded-r-lg px-3 py-2 text-sm text-amber-900/90"
                {...props}
              >
                {children}
              </blockquote>
            ),
            a: ({ children, href, ...props }) => (
              <a
                href={href}
                className="text-primary-700 underline underline-offset-2 hover:text-primary-800"
                target="_blank"
                rel="noopener noreferrer"
                {...props}
              >
                {children}
              </a>
            ),
            pre: ({ children, ...props }) => (
              <pre
                className="my-3 overflow-x-auto rounded-lg bg-slate-100 border border-slate-200 px-3 py-2 text-xs text-slate-800"
                {...props}
              >
                {children}
              </pre>
            ),
            table: ({ children, ...props }) => (
              <div className="my-3 overflow-x-auto rounded-xl border border-slate-200 bg-slate-50">
                <table className="min-w-full text-sm text-slate-800" {...props}>
                  {children}
                </table>
              </div>
            ),
            th: ({ children, ...props }) => (
              <th
                className="px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wide bg-slate-100 border-b border-slate-200 whitespace-nowrap"
                {...props}
              >
                {children}
              </th>
            ),
            td: ({ children, ...props }) => (
              <td
                className="px-3 py-2 align-top text-xs text-slate-700 border-b border-slate-100"
                {...props}
              >
                {children}
              </td>
            ),
            code: (codeProps) => {
              const { inline, className, children, ...props } = codeProps as {
                inline?: boolean;
                className?: string;
                children?: ReactNode;
              } & HTMLAttributes<HTMLElement>;

              const lang = (className || "").replace("language-", "");
              const raw = String(children || "").trim();

              if (!inline && lang === "json-image") {
                try {
                  const payload = JSON.parse(raw) as {
                    engine?: string;
                    action?: string;
                    prompt: string;
                  };
                  return <JsonImageConceptCard payload={payload} />;
                } catch {
                  // Fallback to plain code rendering below
                }
              }

              if (inline) {
                return (
                  <code
                    className="font-mono text-xs bg-slate-100 text-slate-800 px-1.5 py-0.5 rounded border border-slate-200/80"
                    {...props}
                  >
                    {children}
                  </code>
                );
              }

              return (
                <code className={cn("font-mono text-xs", className)} {...props}>
                  {children}
                </code>
              );
            },
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    </div>
  );
}

/* ─── Main Page ─────────────────────────────────────────────── */

export default function ChatRetrievePage() {
  const { clientId } = useTenant();
  const { role } = useAuth();
  const isAdmin = role === "admin";
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
  const [runtimeWarnings, setRuntimeWarnings] = useState<string[]>([]);
  const [tenantRuntime, setTenantRuntime] = useState<EffectiveTenantRuntime | null>(null);
  const [sessionConfigReady, setSessionConfigReady] = useState(false);
  const [sessionConfigLoading, setSessionConfigLoading] = useState(false);

  const [showPromptPanel, setShowPromptPanel] = useState(false);
  const [templateList, setTemplateList] = useState<PromptTemplateListItem[]>([]);
  const [activeTemplateId, setActiveTemplateId] = useState<string | null>(null);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>("");
  const [previewInstructions, setPreviewInstructions] = useState("");
  const [promptPanelLoading, setPromptPanelLoading] = useState(false);
  const [sessionEditEnabled, setSessionEditEnabled] = useState(false);
  const [sessionPromptOverride, setSessionPromptOverride] = useState<string | null>(null);
  const [autoRewrite, setAutoRewrite] = useState(false);
  const [promptSaveToast, setPromptSaveToast] = useState<{
    type: "success" | "error";
    msg: string;
  } | null>(null);
  const [renderMode, setRenderMode] = useState<"plain" | "rich">("plain");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const runtimeFetchGenRef = useRef(0);

  const [catalogError, setCatalogError] = useState<string | null>(null);

  // Global catalogs — options only; never set selectedLLM from this list.
  useEffect(() => {
    if (!clientId) return;
    let cancelled = false;
    (async () => {
      setCatalogError(null);
      try {
        const [llmRes, rrRes] = await Promise.allSettled([
          apiClient.get<LLMProvider[]>(API.MODELS.LLM(clientId)),
          apiClient.get<RerankerOption[]>(API.MODELS.RERANKER()),
        ]);
        if (cancelled) return;
        if (llmRes.status === "fulfilled") {
          setLlmProviders(Array.isArray(llmRes.value.data) ? llmRes.value.data : []);
        } else {
          setLlmProviders([]);
          setCatalogError("Could not load LLM providers.");
        }
        if (rrRes.status === "fulfilled") {
          setRerankers(Array.isArray(rrRes.value.data) ? rrRes.value.data : []);
        } else {
          setRerankers([]);
          setCatalogError((prev) => prev ?? "Could not load reranker plugins.");
        }
      } catch {
        if (!cancelled) {
          setCatalogError("Could not load model catalogs.");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [clientId]);

  const llmOptions = useMemo(() => {
    const options = [...llmProviders];
    const effective = selectedLLM || tenantRuntime?.llm.effective_provider;
    if (effective && !options.some((p) => p.provider === effective)) {
      options.unshift({
        provider: effective,
        display_name: effective,
        default_model: tenantRuntime?.llm.effective_model || "",
        api_key_set: true,
        api_key_env: "(tenant default)",
        recommended: false,
      });
    }
    return options;
  }, [llmProviders, selectedLLM, tenantRuntime]);

  const rerankerOptions = useMemo(() => {
    const options = [...rerankers];
    const effective = selectedReranker || tenantRuntime?.reranker.effective_plugin;
    if (effective && !options.some((r) => r.name === effective)) {
      options.unshift({
        name: effective,
        description: "Tenant effective reranker",
        requires_gpu: false,
        requires_api_key: false,
      });
    }
    return options;
  }, [rerankers, selectedReranker, tenantRuntime]);

  // Hydrate session controls from tenant SSOT (debounced clientId from TenantContext).
  useEffect(() => {
    if (!clientId) return;

    const fetchGen = ++runtimeFetchGenRef.current;
    const controller = new AbortController();

    setSessionConfigReady(false);
    setSessionConfigLoading(true);

    apiClient
      .get<EffectiveTenantRuntime>(API.MODELS.RUNTIME(clientId), {
        signal: controller.signal,
      })
      .then((r) => {
        if (fetchGen !== runtimeFetchGenRef.current) return;
        applyRuntimeToSession(r.data, {
          setTenantRuntime,
          setRuntimeWarnings,
          setSelectedLLM,
          setSelectedReranker,
          setTopK,
          setSearchMode,
          setEnableHyde,
        });
        setSessionConfigReady(true);
      })
      .catch((err: unknown) => {
        if (fetchGen !== runtimeFetchGenRef.current) return;
        const canceled =
          (err as { code?: string; name?: string })?.code === "ERR_CANCELED" ||
          (err as { name?: string })?.name === "CanceledError";
        if (canceled) return;
        setTenantRuntime(null);
        setRuntimeWarnings([]);
        setSessionConfigReady(false);
      })
      .finally(() => {
        if (fetchGen === runtimeFetchGenRef.current) {
          setSessionConfigLoading(false);
        }
      });

    return () => {
      controller.abort();
    };
  }, [clientId]);

  const loadTemplatePreview = useCallback(async (templateId: string) => {
    if (!templateId) {
      setPreviewInstructions("");
      return;
    }
    try {
      const res = await apiClient.get<PromptTemplatePreview>(
        API.PROMPT_TEMPLATES.GET(templateId)
      );
      setPreviewInstructions(res.data.system_instructions || "");
    } catch {
      setPreviewInstructions("");
    }
  }, []);

  useEffect(() => {
    if (!clientId) return;
    let cancelled = false;
    setPromptPanelLoading(true);

    (async () => {
      try {
        const [cfgRes, listRes] = await Promise.all([
          apiClient.get<TenantPromptConfig>(API.TENANT_PROMPT_CONFIG.GET(clientId)),
          apiClient.get<PromptTemplateListItem[]>(API.PROMPT_TEMPLATES.LIST()),
        ]);
        if (cancelled) return;
        setTemplateList(listRes.data);
        const tid = cfgRes.data.prompt_template_id || "";
        setActiveTemplateId(cfgRes.data.prompt_template_id);
        setSelectedTemplateId(tid);
        if (tid) {
          await loadTemplatePreview(tid);
        } else {
          setPreviewInstructions("");
        }
      } catch {
        if (!cancelled) {
          setTemplateList([]);
          setPreviewInstructions("");
        }
      } finally {
        if (!cancelled) setPromptPanelLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [clientId, loadTemplatePreview]);

  useEffect(() => {
    if (!selectedTemplateId) return;
    void loadTemplatePreview(selectedTemplateId);
  }, [selectedTemplateId, loadTemplatePreview]);

  useEffect(() => {
    if (!promptSaveToast) return;
    const t = setTimeout(() => setPromptSaveToast(null), 5000);
    return () => clearTimeout(t);
  }, [promptSaveToast]);

  const handleSaveDefaultTemplate = useCallback(async () => {
    if (!clientId || !selectedTemplateId) return;
    try {
      await apiClient.patch(API.TENANT_PROMPT_CONFIG.PATCH(clientId), {
        prompt_template_id: selectedTemplateId,
      });
      setActiveTemplateId(selectedTemplateId);
      setPromptSaveToast({ type: "success", msg: "Default prompt template saved." });
    } catch (err: unknown) {
      const ax = err as { response?: { data?: { detail?: string } } };
      const msg =
        ax?.response?.data?.detail || "Failed to save default prompt template.";
      setPromptSaveToast({ type: "error", msg: String(msg) });
    }
  }, [clientId, selectedTemplateId]);

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
      const body: Record<string, unknown> = {
        session_id: sessionId,
        messages,
        client_id: clientId,
        llm_provider: selectedLLM || undefined,
        reranker: selectedReranker,
        intent: "answer",
        top_k: topK,
        search_mode: searchMode,
        rewrite_enabled: autoRewrite,
        generate_answer: generateAnswer,
      };
      if (sessionPromptOverride) {
        body.system_prompt_override = sessionPromptOverride;
      }
      body.enable_hyde = enableHyde;

      const res = await apiClient.post<ChatResponse>(
        "/api/v2/retrieve/chat",
        body,
        { timeout: CHAT_RETRIEVE_TIMEOUT_MS }
      );

      setTurns((prev) =>
        prev.map((t) =>
          t.id === asstId
            ? { ...t, content: res.data.answer || "No answer generated.", response: res.data }
            : t
        )
      );
    } catch (err: unknown) {
      const ax = err as {
        response?: { data?: { detail?: string } };
        message?: string;
      };
      const errMsg =
        ax?.response?.data?.detail || ax?.message || "Request failed";
      setTurns((prev) =>
        prev.map((t) => (t.id === asstId ? { ...t, content: "", error: errMsg } : t))
      );
    } finally {
      setLoading(false);
    }
  }, [
    input,
    loading,
    turns,
    sessionId,
    selectedLLM,
    selectedReranker,
    topK,
    searchMode,
    generateAnswer,
    autoRewrite,
    enableHyde,
    sessionPromptOverride,
    clientId,
  ]);

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
          {catalogError && (
            <div className="rounded-md border border-red-200 bg-red-50 px-2.5 py-2 text-[10px] text-red-800">
              {catalogError}
            </div>
          )}
          {runtimeWarnings.length > 0 && (
            <div className="rounded-md border border-amber-200 bg-amber-50 px-2.5 py-2 text-[10px] text-amber-900 space-y-1">
              {runtimeWarnings.map((w, i) => (
                <p key={i}>{w}</p>
              ))}
            </div>
          )}
          {tenantRuntime?.reranker.coercion_applied && (
            <p className="text-[10px] text-slate-500">
              Reranker effective:{" "}
              <span className="font-semibold">{tenantRuntime.reranker.effective_plugin}</span>
              {tenantRuntime.reranker.configured_type
                ? ` (configured: ${tenantRuntime.reranker.configured_type})`
                : null}
            </p>
          )}
          {sessionConfigLoading && (
            <p className="text-[10px] text-slate-500 flex items-center gap-1">
              <Loader2 className="h-3 w-3 animate-spin" />
              Loading tenant session config…
            </p>
          )}
          {sessionConfigReady && tenantRuntime && (
            <p className="text-[10px] text-slate-500">
              SSOT: {tenantRuntime.llm.effective_provider} / {tenantRuntime.reranker.effective_plugin}
            </p>
          )}
          <p className="text-[10px] text-slate-500">
            Session-only — changes here apply to this console only and do not alter tenant defaults in Settings → Reranking & Rules.
          </p>
          {/* LLM Provider */}
          <div>
            <label className="text-xs font-medium text-slate-500 block mb-1">LLM Provider</label>
            <select
              value={selectedLLM}
              onChange={(e) => setSelectedLLM(e.target.value)}
              disabled={!sessionConfigReady || sessionConfigLoading}
              className="w-full rounded-md border border-slate-200 px-2.5 py-1.5 text-xs text-slate-700 bg-white focus:border-primary-300 outline-none disabled:opacity-60"
            >
              {!sessionConfigReady && (
                <option value="">Loading tenant defaults…</option>
              )}
              {llmOptions.map((p) => (
                <option key={p.provider} value={p.provider} disabled={!p.api_key_set}>
                  {p.display_name} {p.api_key_set ? `(${p.default_model})` : "(no API key)"}{p.recommended ? " (catalog)" : ""}
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
              disabled={!sessionConfigReady || sessionConfigLoading}
              className="w-full rounded-md border border-slate-200 px-2.5 py-1.5 text-xs text-slate-700 bg-white focus:border-primary-300 outline-none disabled:opacity-60"
            >
              {rerankerOptions.map((r) => (
                <option key={r.name} value={r.name}>
                  {r.name === "none" ? "None" : r.name}
                </option>
              ))}
            </select>
          </div>

          {/* Top K */}
          <div>
            <label
              className="text-xs font-medium text-slate-500 block mb-1"
              title="This setting affects only the current request/session. To change tenant defaults, use Settings → Reranking & Rules."
            >
              Top K: {topK}
            </label>
            <input
              type="range" min={1} max={20} value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full h-1.5 accent-primary-600"
            />
          </div>

          {/* Search Mode */}
          <div>
            <label
              className="text-xs font-medium text-slate-500 block mb-1"
              title="This setting affects only the current request/session. To change tenant defaults, use Settings → Reranking & Rules."
            >
              Search Mode
            </label>
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

          {/* Answer display mode */}
          <div>
            <label className="text-xs font-medium text-slate-500 block mb-1">
              Answer Display
            </label>
            <div
              className="inline-flex rounded-full bg-slate-100 p-0.5"
              role="group"
              aria-label="Answer display mode"
            >
              <button
                type="button"
                onClick={() => setRenderMode("plain")}
                aria-pressed={renderMode === "plain"}
                className={cn(
                  "px-2.5 py-1 text-[11px] rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-1 focus-visible:ring-offset-slate-100 transition-colors",
                  renderMode === "plain"
                    ? "bg-white text-slate-800 shadow-sm"
                    : "text-slate-500"
                )}
              >
                Plain text
              </button>
              <button
                type="button"
                onClick={() => setRenderMode("rich")}
                aria-pressed={renderMode === "rich"}
                className={cn(
                  "ml-1 px-2.5 py-1 text-[11px] rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-1 focus-visible:ring-offset-slate-100 transition-colors",
                  renderMode === "rich"
                    ? "bg-white text-slate-800 shadow-sm"
                    : "text-slate-500"
                )}
              >
                Rich markdown & visuals
              </button>
            </div>
            <p className="mt-1 text-[10px] text-slate-500">
              Controls only how answers are displayed in this console.
            </p>
          </div>

          {/* Toggles */}
          <div className="space-y-2">
            <label
              className="flex items-center gap-2 cursor-pointer"
              title="This setting affects only the current request/session. To change tenant defaults, use Settings → Reranking & Rules."
            >
              <input
                type="checkbox" checked={enableHyde}
                onChange={(e) => setEnableHyde(e.target.checked)}
                className="rounded border-slate-300 text-primary-600 focus:ring-primary-500 w-3.5 h-3.5"
              />
              <span className="text-xs text-slate-600">HyDE Expansion</span>
            </label>
            <label
              className="flex items-center gap-2 cursor-pointer"
              title="This setting affects only the current request/session. To change tenant defaults, use Settings → Reranking & Rules."
            >
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
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 text-[10px] ml-2">
            <Building2 className="w-3 h-3" />
            {clientId}
          </span>
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

              <div
                className={cn(
                  "rounded-2xl px-4 py-3",
                  turn.role === "user"
                    ? "max-w-[75%] bg-primary-600 text-white"
                    : cn(
                        "bg-white border border-slate-200 shadow-sm",
                        renderMode === "rich" && turn.response?.answer
                          ? "max-w-[92%]"
                          : "max-w-[75%]"
                      )
                )}
              >
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
                      renderMode === "plain" ? (
                        <p className="text-sm text-slate-700 whitespace-pre-wrap leading-relaxed">
                          {turn.response.answer}
                        </p>
                      ) : (
                        <AssistantMarkdown content={turn.response.answer} />
                      )
                    )}

                    {turn.response?.debug_info &&
                      !turn.error &&
                      (() => {
                        const dbg = parseChatDebugInfo(turn.response.debug_info);
                        if (!dbg) return null;
                        const genAttempted =
                          turn.response.answer_latency_ms != null ||
                          Boolean(turn.response.answer) ||
                          Boolean(turn.response.answer_error);
                        return (
                          <EffectiveThisTurnStrip
                            debug={dbg}
                            observability={dbg.observability}
                            generateAnswer={genAttempted}
                          />
                        );
                      })()}

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
                          {turn.response.results.length} sources
                          {expandedDebug.has(turn.id) ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                        </button>

                        {expandedDebug.has(turn.id) && (
                          <div className="mt-2 space-y-2">
                            {/* Debug chips */}
                            {(() => {
                              const dbg = parseChatDebugInfo(turn.response.debug_info);
                              if (!dbg) return null;
                              return (
                              <div className="flex flex-wrap gap-1 mb-2">
                                {dbg.embedder && (
                                  <DebugChip icon={Brain} label="Embedder" value={dbg.embedder} />
                                )}
                                {dbg.llm_provider && (
                                  <DebugChip icon={Bot} label="LLM" value={`${dbg.llm_provider}/${dbg.llm_model || ""}`} />
                                )}
                                <DebugChip icon={Layers} label="Reranker" value={dbg.reranker_used || "none"} />
                                {dbg.search_mode && (
                                  <DebugChip icon={Search} label="Search" value={dbg.search_mode} />
                                )}
                                {dbg.vectordb && (
                                  <DebugChip icon={Database} label="VectorDB" value={dbg.vectordb} />
                                )}
                                {dbg.query_rewritten && (
                                  <DebugChip icon={Zap} label="Rewrite" value="yes" />
                                )}
                              </div>
                              );
                            })()}

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
        <div className="border-t border-slate-200 bg-white p-4 space-y-3">
          {promptSaveToast && (
            <div
              className={cn(
                "max-w-4xl mx-auto flex items-center gap-2 rounded-lg border px-3 py-2 text-xs",
                promptSaveToast.type === "success"
                  ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                  : "border-red-200 bg-red-50 text-red-800"
              )}
            >
              {promptSaveToast.type === "success" ? (
                <Check className="w-3.5 h-3.5" />
              ) : (
                <AlertCircle className="w-3.5 h-3.5" />
              )}
              <span>{promptSaveToast.msg}</span>
            </div>
          )}

          {/* Prompt panel — collapsed by default */}
          <div className="max-w-4xl mx-auto rounded-xl border border-slate-200 bg-slate-50/80 overflow-hidden">
            <button
              type="button"
              onClick={() => setShowPromptPanel((v) => !v)}
              className="w-full flex items-center justify-between px-4 py-2.5 text-left hover:bg-slate-100/80 transition-colors"
            >
              <span className="flex items-center gap-2 text-sm font-medium text-slate-700">
                Prompt
                {sessionPromptOverride !== null && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 border border-amber-200">
                    Modified
                  </span>
                )}
              </span>
              {showPromptPanel ? (
                <ChevronUp className="w-4 h-4 text-slate-400" />
              ) : (
                <ChevronDown className="w-4 h-4 text-slate-400" />
              )}
            </button>

            {showPromptPanel && (
              <div className="px-4 pb-4 space-y-3 border-t border-slate-200 bg-white">
                {promptPanelLoading ? (
                  <p className="text-xs text-slate-500 flex items-center gap-1 pt-3">
                    <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading prompt config…
                  </p>
                ) : (
                  <>
                    <div className="pt-3">
                      <label className="text-xs font-medium text-slate-500 block mb-1">
                        Template
                      </label>
                      <select
                        value={selectedTemplateId}
                        onChange={(e) => {
                          setSelectedTemplateId(e.target.value);
                          if (sessionEditEnabled) {
                            setSessionPromptOverride(null);
                          }
                        }}
                        className="w-full rounded-md border border-slate-200 px-2.5 py-1.5 text-xs text-slate-700 bg-white focus:border-primary-300 outline-none"
                      >
                        <option value="">System default</option>
                        {templateList.map((t) => (
                          <option key={t.template_id} value={t.template_id}>
                            {t.name}
                            {t.has_examples ? " (contains examples)" : ""}
                          </option>
                        ))}
                      </select>
                      {templateList
                        .filter((t) => t.template_id === selectedTemplateId && t.has_examples)
                        .map((t) => (
                          <span
                            key={t.template_id}
                            className="inline-flex mt-1.5 text-[10px] px-1.5 py-0.5 rounded bg-amber-50 text-amber-800 border border-amber-200"
                          >
                            Contains examples
                          </span>
                        ))}
                    </div>

                    <div>
                      <label className="text-xs font-medium text-slate-500 block mb-1">
                        Preview as seen by LLM (examples removed)
                      </label>
                      <textarea
                        readOnly={!sessionEditEnabled}
                        value={
                          sessionEditEnabled
                            ? sessionPromptOverride ?? previewInstructions
                            : previewInstructions
                        }
                        onChange={(e) => {
                          if (sessionEditEnabled) {
                            setSessionPromptOverride(e.target.value);
                          }
                        }}
                        rows={5}
                        className={cn(
                          "w-full rounded-md border px-3 py-2 text-xs font-mono leading-relaxed outline-none resize-y",
                          sessionEditEnabled
                            ? "border-primary-200 bg-white text-slate-700 focus:border-primary-300 focus:ring-2 focus:ring-primary-100"
                            : "border-slate-200 bg-slate-50 text-slate-600"
                        )}
                      />
                    </div>

                    <div className="flex flex-wrap items-center gap-3">
                      <Link
                        href={pipelineSettingsHref(clientId, "prompts")}
                        className="text-[10px] font-medium text-primary-700 hover:text-primary-800 underline-offset-2 hover:underline"
                      >
                        Edit tenant defaults in Pipeline →
                      </Link>
                      {sessionPromptOverride !== null && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 border border-amber-200">
                          Session override active
                        </span>
                      )}
                      {activeTemplateId && selectedTemplateId === activeTemplateId && sessionPromptOverride === null && (
                        <span className="text-[10px] text-emerald-700">Tenant default template</span>
                      )}
                      <label
                        className="flex items-center gap-2 cursor-pointer"
                        title="Edit prompt text for this session only (not saved to tenant config)"
                      >
                        <input
                          type="checkbox"
                          checked={sessionEditEnabled}
                          onChange={(e) => {
                            const on = e.target.checked;
                            setSessionEditEnabled(on);
                            if (on) {
                              setSessionPromptOverride(previewInstructions);
                            } else {
                              setSessionPromptOverride(null);
                            }
                          }}
                          className="rounded border-slate-300 text-primary-600 focus:ring-primary-500 w-3.5 h-3.5"
                        />
                        <span className="text-xs text-slate-600">Edit for this session</span>
                      </label>

                      {isAdmin && (
                        <button
                          type="button"
                          onClick={() => void handleSaveDefaultTemplate()}
                          disabled={!selectedTemplateId}
                          className={cn(
                            "text-xs px-3 py-1.5 rounded-md border transition-colors",
                            selectedTemplateId
                              ? "border-primary-200 bg-primary-50 text-primary-700 hover:bg-primary-100"
                              : "border-slate-200 text-slate-400 cursor-not-allowed"
                          )}
                        >
                          Save as default
                        </button>
                      )}
                      {activeTemplateId && selectedTemplateId === activeTemplateId && (
                        <span className="text-[10px] text-emerald-700">Tenant default</span>
                      )}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>

          <div className="flex items-end gap-3 max-w-4xl mx-auto">
            <label
              className="flex items-center gap-2 cursor-pointer shrink-0 pb-2"
              title="When ON, rewrites your query for better recall in multi-turn conversations. Turn OFF for precise lookups like invoice IDs."
            >
              <input
                type="checkbox"
                checked={autoRewrite}
                onChange={(e) => setAutoRewrite(e.target.checked)}
                className="rounded border-slate-300 text-primary-600 focus:ring-primary-500 w-3.5 h-3.5"
              />
              <span className="text-xs text-slate-600 whitespace-nowrap">Auto-rewrite</span>
            </label>
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
