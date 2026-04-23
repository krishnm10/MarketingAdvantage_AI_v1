"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Sparkles, Plus, Trash2, Edit3, Eye, Save, Copy, CheckCircle2,
  XCircle, Loader2, RefreshCw, Tag, FileText, MessageSquare, Zap,
  AlertTriangle, ChevronRight, Lock, Hash, Settings2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";

/* ─── Types ─── */
interface PromptTemplate {
  template_id: string;
  name: string;
  system_instructions: string;
  context_format: string;
  question_prefix: string;
  answer_prefix: string;
  citation_style: string;
  max_context_chars: number | null;
  version: string;
  tags: string[];
  created_at?: number;
  updated_at?: number;
}

interface PreviewResult {
  template_id: string;
  system_prompt: string;
  rendered_context: string;
  full_user_turn: string;
  estimated_chars: number;
  estimated_tokens: number;
}

const CITATION_OPTIONS = [
  { value: "inline_numeric", label: "Inline Numeric [1], [2]" },
  { value: "inline_source",  label: "Inline Source (Source: filename)" },
  { value: "none",           label: "No Citations" },
];

const STARTER_TEMPLATES = [
  {
    template_id:         "marketing-rag-v1",
    name:                "Marketing Intelligence (Default)",
    system_instructions: `You are a Marketing Intelligence Assistant for enterprise businesses.
Answer the question using ONLY the context passages provided below.
If the context does not contain sufficient information to answer, state clearly: "I don't have enough information in the provided context to answer this."
Cite the source number (e.g. [1], [2]) when referencing a specific passage.
Be concise, factual, and precise. Do not speculate beyond the context.`,
    citation_style: "inline_numeric",
    tags: ["marketing", "rag", "default"],
  },
  {
    template_id:         "campaign-analyst-v1",
    name:                "Campaign Performance Analyst",
    system_instructions: `You are a Campaign Performance Analyst with deep expertise in digital marketing metrics.
Analyse the provided context and answer with data-driven insights.
Always cite the data source [N] when making claims.
If asked about trends, reference specific numbers from the context.
Do not invent metrics — if data is absent, say so explicitly.`,
    citation_style: "inline_numeric",
    tags: ["campaigns", "analytics", "metrics"],
  },
  {
    template_id:         "brand-voice-v1",
    name:                "Brand Voice Compliance",
    system_instructions: `You are a Brand Compliance Expert. Your role is to answer questions about brand guidelines, tone, and style.
Use ONLY the provided brand documentation to answer.
When referencing a guideline, cite the document and section [N].
If no relevant guideline exists, state that clearly.
Maintain a professional, precise tone in all answers.`,
    citation_style: "inline_source",
    tags: ["brand", "compliance", "guidelines"],
  },
  {
    template_id:         "competitive-intel-v1",
    name:                "Competitive Intelligence",
    system_instructions: `You are a Competitive Intelligence Analyst. Provide objective, evidence-based competitive insights.
Base all analysis strictly on the context provided — do not use external knowledge.
Cite context passages [N] to support every claim.
Distinguish clearly between fact (from context) and inference.
If context is insufficient for a comparison, state that explicitly.`,
    citation_style: "inline_numeric",
    tags: ["competitive", "intelligence", "analysis"],
  },
];

function slugify(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 64);
}

export default function PromptBuilderPage() {
  const [templates,       setTemplates]       = useState<PromptTemplate[]>([]);
  const [selected,        setSelected]        = useState<PromptTemplate | null>(null);
  const [mode,            setMode]            = useState<"list" | "edit" | "create">("list");
  const [loading,         setLoading]         = useState(true);
  const [saving,          setSaving]          = useState(false);
  const [saveStatus,      setSaveStatus]      = useState<"idle" | "ok" | "err">("idle");
  const [preview,         setPreview]         = useState<PreviewResult | null>(null);
  const [previewLoading,  setPreviewLoading]  = useState(false);
  const [previewQuery,    setPreviewQuery]    = useState("What was our Q3 marketing ROI?");
  const [deleteConfirm,   setDeleteConfirm]   = useState<string | null>(null);

  /* ─── Edit form state ─── */
  const [editId,          setEditId]          = useState("");
  const [editName,        setEditName]        = useState("");
  const [editSystem,      setEditSystem]      = useState("");
  const [editContext,     setEditContext]     = useState("[{index}] {text}\n  [Source: {source}]");
  const [editQPrefix,     setEditQPrefix]     = useState("Question:");
  const [editAPrefix,     setEditAPrefix]     = useState("Answer:");
  const [editCitation,    setEditCitation]    = useState("inline_numeric");
  const [editMaxChars,    setEditMaxChars]    = useState<number | null>(null);
  const [editVersion,     setEditVersion]     = useState("1.0.0");
  const [editTags,        setEditTags]        = useState("");
  const [idError,         setIdError]         = useState("");

  const loadTemplates = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await apiClient.get(API.PROMPT_TEMPLATES.LIST());
      setTemplates(resp.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadTemplates(); }, [loadTemplates]);

  const populateForm = (t: Partial<PromptTemplate> & { template_id?: string }) => {
    setEditId(t.template_id ?? "");
    setEditName(t.name ?? "");
    setEditSystem(t.system_instructions ?? "");
    setEditContext(t.context_format ?? "[{index}] {text}\n  [Source: {source}]");
    setEditQPrefix(t.question_prefix ?? "Question:");
    setEditAPrefix(t.answer_prefix ?? "Answer:");
    setEditCitation(t.citation_style ?? "inline_numeric");
    setEditMaxChars(t.max_context_chars ?? null);
    setEditVersion(t.version ?? "1.0.0");
    setEditTags((t.tags ?? []).join(", "));
    setIdError("");
  };

  const handleNew = () => {
    populateForm({});
    setPreview(null);
    setMode("create");
  };

  const handleEdit = async (tmpl: PromptTemplate) => {
    try {
      const resp = await apiClient.get(API.PROMPT_TEMPLATES.GET(tmpl.template_id));
      populateForm(resp.data);
    } catch {
      populateForm(tmpl);
    }
    setSelected(tmpl);
    setPreview(null);
    setMode("edit");
  };

  const handleStarterTemplate = (starter: typeof STARTER_TEMPLATES[0]) => {
    populateForm({
      ...starter,
      context_format: "[{index}] {text}\n  [Source: {source}]",
      question_prefix: "Question:",
      answer_prefix: "Answer:",
      version: "1.0.0",
    });
    setMode("create");
  };

  const validateId = (id: string): boolean => {
    if (!/^[a-z0-9][a-z0-9\-]{0,62}[a-z0-9]$/.test(id)) {
      setIdError("Use lowercase alphanumeric + hyphens, 2–64 chars, no leading/trailing hyphens.");
      return false;
    }
    setIdError("");
    return true;
  };

  const handleSave = async () => {
    if (mode === "create" && !validateId(editId)) return;
    setSaving(true);
    setSaveStatus("idle");
    const payload = {
      template_id:         editId,
      name:                editName,
      system_instructions: editSystem,
      context_format:      editContext,
      question_prefix:     editQPrefix,
      answer_prefix:       editAPrefix,
      citation_style:      editCitation,
      max_context_chars:   editMaxChars,
      version:             editVersion,
      tags:                editTags.split(",").map(t => t.trim()).filter(Boolean),
    };
    try {
      if (mode === "create") {
        await apiClient.post(API.PROMPT_TEMPLATES.CREATE(), payload);
      } else {
        await apiClient.put(API.PROMPT_TEMPLATES.UPDATE(editId), payload);
      }
      setSaveStatus("ok");
      await loadTemplates();
      setTimeout(() => { setSaveStatus("idle"); setMode("list"); }, 1200);
    } catch {
      setSaveStatus("err");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (deleteConfirm !== id) { setDeleteConfirm(id); return; }
    try {
      await apiClient.delete(API.PROMPT_TEMPLATES.DELETE(id));
      setDeleteConfirm(null);
      await loadTemplates();
    } catch { }
  };

  const handlePreview = async () => {
    if (!editId) return;
    setPreviewLoading(true);
    try {
      const resp = await apiClient.post(API.PROMPT_TEMPLATES.PREVIEW(editId), {
        query: previewQuery,
        chunks: [
          { text: "Marketing ROI for Q3 2025 was 340% in the Mumbai region, driven by digital channels.", metadata: { source: "q3_report.pdf", page_number: 8 } },
          { text: "Campaign spend was ₹12.4M with a return of ₹42M, primarily from performance marketing.", metadata: { source: "q3_report.pdf", page_number: 9 } },
        ],
      });
      setPreview(resp.data);
    } catch { } finally {
      setPreviewLoading(false);
    }
  };

  const handleCopySystem = () => {
    navigator.clipboard.writeText(editSystem);
  };

  /* ─── List view ─── */
  if (mode === "list") {
    return (
      <div className="space-y-6 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-2">
              <Sparkles className="h-6 w-6 text-primary-400" />
              Prompt Library
            </h1>
            <p className="mt-1 text-sm text-slate-400">
              Build and manage enterprise RAG prompt templates. Templates are versioned, tagged, and reusable across client pipelines.
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={loadTemplates}
              className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-700 transition-colors"
            >
              <RefreshCw className="h-3.5 w-3.5" /> Refresh
            </button>
            <button
              onClick={handleNew}
              className="flex items-center gap-1.5 rounded-lg bg-primary-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-primary-700 transition-colors"
            >
              <Plus className="h-4 w-4" /> New Template
            </button>
          </div>
        </div>

        {/* Starter templates */}
        {templates.length === 0 && !loading && (
          <div className="rounded-xl border border-dashed border-slate-700 p-6 space-y-4">
            <p className="text-sm font-medium text-slate-300 text-center">No templates yet — start with a preset:</p>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {STARTER_TEMPLATES.map(starter => (
                <button
                  key={starter.template_id}
                  onClick={() => handleStarterTemplate(starter)}
                  className="rounded-lg border border-slate-700 bg-slate-800/50 p-4 text-left hover:border-primary-500/50 hover:bg-slate-800 transition-colors group"
                >
                  <p className="text-sm font-medium text-white group-hover:text-primary-300 transition-colors">{starter.name}</p>
                  <div className="mt-2 flex flex-wrap gap-1">
                    {starter.tags.map(t => (
                      <span key={t} className="rounded bg-slate-700 px-1.5 py-0.5 text-[10px] text-slate-400">{t}</span>
                    ))}
                  </div>
                  <p className="mt-2 text-xs text-primary-400 group-hover:text-primary-300">Use this preset →</p>
                </button>
              ))}
            </div>
          </div>
        )}

        {loading ? (
          <div className="flex items-center gap-2 text-slate-400">
            <Loader2 className="h-5 w-5 animate-spin" /> Loading templates…
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            {templates.map(tmpl => (
              <div key={tmpl.template_id} className="rounded-xl border border-slate-700/50 bg-slate-900 p-4 space-y-3">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="text-sm font-semibold text-white truncate">{tmpl.name}</p>
                    <p className="text-[10px] text-slate-500 font-mono mt-0.5 truncate">{tmpl.template_id}</p>
                  </div>
                  <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-400 font-mono flex-shrink-0">v{tmpl.version}</span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {(tmpl.tags || []).map(t => (
                    <span key={t} className="rounded-full bg-primary-500/10 text-primary-400 border border-primary-500/20 px-2 py-0.5 text-[10px]">{t}</span>
                  ))}
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-slate-500">{CITATION_OPTIONS.find(c => c.value === tmpl.citation_style)?.label}</span>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleEdit(tmpl)}
                    className="flex-1 flex items-center justify-center gap-1.5 rounded-lg bg-slate-800 py-1.5 text-xs text-slate-300 hover:bg-slate-700 hover:text-white transition-colors"
                  >
                    <Edit3 className="h-3.5 w-3.5" /> Edit
                  </button>
                  <button
                    onClick={() => handleDelete(tmpl.template_id)}
                    className={cn(
                      "flex items-center justify-center gap-1 rounded-lg px-3 py-1.5 text-xs transition-colors",
                      deleteConfirm === tmpl.template_id
                        ? "bg-rose-500/20 text-rose-400 border border-rose-500/30"
                        : "bg-slate-800 text-slate-500 hover:bg-rose-500/10 hover:text-rose-400"
                    )}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    {deleteConfirm === tmpl.template_id ? "Confirm" : ""}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  /* ─── Edit / Create view ─── */
  return (
    <div className="p-6 space-y-5">
      {/* Header */}
      <div className="flex items-center gap-2">
        <button onClick={() => { setMode("list"); setPreview(null); }} className="text-slate-400 hover:text-white transition-colors text-sm flex items-center gap-1">
          ← Back
        </button>
        <ChevronRight className="h-3.5 w-3.5 text-slate-600" />
        <h1 className="text-lg font-semibold text-white">
          {mode === "create" ? "New Prompt Template" : `Edit: ${editName}`}
        </h1>
      </div>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-5">
        {/* Left: form */}
        <div className="xl:col-span-3 space-y-4">
          <div className="rounded-xl border border-slate-700/50 bg-slate-900 p-5 space-y-4">
            <h2 className="text-sm font-semibold text-white flex items-center gap-2">
              <Settings2 className="h-4 w-4 text-primary-400" /> Template Identity
            </h2>
            <div className="grid grid-cols-2 gap-4">
              <div className="col-span-2 sm:col-span-1">
                <label className="block text-xs text-slate-400 mb-1">
                  Template ID <span className="text-rose-400">*</span>
                  {mode === "edit" && <Lock className="inline h-3 w-3 ml-1 text-slate-500" />}
                </label>
                <input
                  value={editId}
                  onChange={e => { setEditId(slugify(e.target.value)); setIdError(""); }}
                  disabled={mode === "edit"}
                  placeholder="marketing-rag-v1"
                  className={cn(
                    "w-full rounded-lg border px-3 py-1.5 text-sm font-mono focus:outline-none",
                    mode === "edit"
                      ? "border-slate-700 bg-slate-800/50 text-slate-500 cursor-not-allowed"
                      : idError
                      ? "border-rose-500 bg-slate-800 text-white focus:border-rose-400"
                      : "border-slate-700 bg-slate-800 text-white focus:border-primary-500"
                  )}
                />
                {idError && <p className="text-xs text-rose-400 mt-1">{idError}</p>}
              </div>
              <div className="col-span-2 sm:col-span-1">
                <label className="block text-xs text-slate-400 mb-1">Display Name <span className="text-rose-400">*</span></label>
                <input
                  value={editName}
                  onChange={e => setEditName(e.target.value)}
                  placeholder="Marketing Intelligence"
                  className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-white focus:border-primary-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">Version</label>
                <input
                  value={editVersion}
                  onChange={e => setEditVersion(e.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm font-mono text-white focus:border-primary-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">Citation Style</label>
                <select
                  value={editCitation}
                  onChange={e => setEditCitation(e.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-white focus:border-primary-500 focus:outline-none"
                >
                  {CITATION_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              </div>
              <div className="col-span-2">
                <label className="block text-xs text-slate-400 mb-1">Tags (comma-separated)</label>
                <input
                  value={editTags}
                  onChange={e => setEditTags(e.target.value)}
                  placeholder="marketing, rag, enterprise"
                  className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-white focus:border-primary-500 focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* System Instructions */}
          <div className="rounded-xl border border-slate-700/50 bg-slate-900 p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-white flex items-center gap-2">
                <MessageSquare className="h-4 w-4 text-violet-400" /> System Instructions
              </h2>
              <button onClick={handleCopySystem} className="flex items-center gap-1 text-xs text-slate-400 hover:text-white transition-colors">
                <Copy className="h-3 w-3" /> Copy
              </button>
            </div>
            <p className="text-[11px] text-slate-500">
              Define the assistant's persona, grounding rules, citation policy, and response format.
              Variables: <code className="text-slate-400">&#123;&#123;CONTEXT&#125;&#125;</code>, <code className="text-slate-400">&#123;&#123;QUESTION&#125;&#125;</code> are injected at runtime.
            </p>
            <textarea
              value={editSystem}
              onChange={e => setEditSystem(e.target.value)}
              rows={10}
              className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-white font-mono leading-relaxed focus:border-primary-500 focus:outline-none resize-y"
              placeholder={`You are a Marketing Intelligence Assistant.
Answer ONLY from the context provided.
If insufficient context, say so clearly.
Cite [1], [2] when referencing passages.`}
            />
            <div className="flex items-center gap-2 text-[10px] text-slate-500">
              <Hash className="h-3 w-3" />
              <span>~{Math.round(editSystem.length / 4)} tokens</span>
            </div>
          </div>

          {/* Advanced format */}
          <div className="rounded-xl border border-slate-700/50 bg-slate-900 p-5 space-y-4">
            <h2 className="text-sm font-semibold text-white flex items-center gap-2">
              <FileText className="h-4 w-4 text-teal-400" /> Context & Question Format
            </h2>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <label className="block text-xs text-slate-400 mb-1">Context Chunk Format</label>
                <textarea
                  value={editContext}
                  onChange={e => setEditContext(e.target.value)}
                  rows={3}
                  className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs font-mono text-white focus:border-primary-500 focus:outline-none"
                />
                <p className="text-[10px] text-slate-500 mt-1">
                  Variables: <code>&#123;index&#125;</code>, <code>&#123;text&#125;</code>, <code>&#123;source&#125;</code>, <code>&#123;page&#125;</code>
                </p>
              </div>
              <div className="space-y-3">
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Question Prefix</label>
                  <input
                    value={editQPrefix}
                    onChange={e => setEditQPrefix(e.target.value)}
                    className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-white focus:border-primary-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Answer Prefix</label>
                  <input
                    value={editAPrefix}
                    onChange={e => setEditAPrefix(e.target.value)}
                    className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-white focus:border-primary-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">
                    Max Context Chars <span className="text-slate-600">(blank = token budget governs)</span>
                  </label>
                  <input
                    type="number"
                    value={editMaxChars ?? ""}
                    onChange={e => setEditMaxChars(e.target.value ? parseInt(e.target.value) : null)}
                    placeholder="None"
                    className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-white focus:border-primary-500 focus:outline-none"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3">
            <button
              onClick={handleSave}
              disabled={saving || !editId || !editName || !editSystem}
              className={cn(
                "flex items-center gap-2 rounded-lg px-5 py-2 text-sm font-medium transition-colors",
                saving || !editId || !editName || !editSystem
                  ? "bg-slate-700 text-slate-500 cursor-not-allowed"
                  : "bg-primary-600 text-white hover:bg-primary-700"
              )}
            >
              {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
              {saving ? "Saving…" : mode === "create" ? "Create Template" : "Save Changes"}
            </button>
            {saveStatus === "ok"  && <span className="flex items-center gap-1 text-sm text-emerald-400"><CheckCircle2 className="h-4 w-4" /> Saved</span>}
            {saveStatus === "err" && <span className="flex items-center gap-1 text-sm text-rose-400"><XCircle className="h-4 w-4" /> Error saving</span>}
            <button onClick={() => { setMode("list"); setPreview(null); }} className="text-sm text-slate-400 hover:text-white transition-colors">
              Cancel
            </button>
          </div>
        </div>

        {/* Right: preview */}
        <div className="xl:col-span-2 space-y-4">
          <div className="rounded-xl border border-slate-700/50 bg-slate-900 p-5 space-y-3 sticky top-6">
            <h2 className="text-sm font-semibold text-white flex items-center gap-2">
              <Eye className="h-4 w-4 text-amber-400" /> Live Preview
            </h2>
            <div>
              <label className="block text-xs text-slate-400 mb-1">Sample Question</label>
              <input
                value={previewQuery}
                onChange={e => setPreviewQuery(e.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-sm text-white focus:border-primary-500 focus:outline-none"
              />
            </div>
            <button
              onClick={handlePreview}
              disabled={previewLoading || !editId}
              className={cn(
                "w-full flex items-center justify-center gap-2 rounded-lg py-2 text-sm font-medium transition-colors",
                previewLoading || !editId
                  ? "bg-slate-700 text-slate-500 cursor-not-allowed"
                  : "bg-amber-600/20 text-amber-300 border border-amber-500/30 hover:bg-amber-600/30"
              )}
            >
              {previewLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}
              {previewLoading ? "Rendering…" : "Preview Render"}
            </button>

            {preview && (
              <div className="space-y-3 text-xs">
                <div className="flex items-center gap-3 text-[10px] text-slate-400 bg-slate-800 rounded-lg p-2">
                  <span>~{preview.estimated_tokens} tokens</span>
                  <span>·</span>
                  <span>{preview.estimated_chars} chars</span>
                </div>
                <div>
                  <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1">System Prompt</p>
                  <pre className="whitespace-pre-wrap text-slate-300 bg-slate-800 rounded-lg p-3 leading-relaxed text-[11px] overflow-auto max-h-40">{preview.system_prompt}</pre>
                </div>
                <div>
                  <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1">User Turn (Context + Question)</p>
                  <pre className="whitespace-pre-wrap text-slate-300 bg-slate-800 rounded-lg p-3 leading-relaxed text-[11px] overflow-auto max-h-48">{preview.full_user_turn}</pre>
                </div>
              </div>
            )}

            {!preview && !previewLoading && editId && (
              <div className="rounded-lg border border-dashed border-slate-700 p-4 text-center">
                <Eye className="h-6 w-6 text-slate-600 mx-auto mb-1" />
                <p className="text-xs text-slate-500">
                  {mode === "create"
                    ? "Save the template first, then preview."
                    : "Click Preview Render to see the rendered output."}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
