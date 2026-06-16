"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { FileText, Library } from "lucide-react";
import { cn } from "@/lib/utils";

import { useTenant } from "@/contexts/TenantContext";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";
import { useUnsavedChanges } from "@/lib/useUnsavedChanges";
import {
  libraryIdForPromptPreset,
  PROMPT_PRESET_LIBRARY,
  type EffectiveTenantRuntime,
} from "@/lib/effectiveTenantRuntime";
import { RuntimeWarningBanner } from "@/components/runtime/RuntimeWarningBanner";
import { SettingsCard } from "@/components/ui/SettingsCard";
import { SaveBar } from "@/components/ui/SaveBar";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const SYSTEM_DEFAULT_SELECT_VALUE = "__system_default__";

const PROMPT_STRATEGIES = [
  { value: "rag_context", label: "RAG Context", desc: "Standard grounded Q&A" },
  { value: "cot", label: "Chain of Thought", desc: "Step-by-step reasoning" },
  { value: "few_shot", label: "Few Shot", desc: "Example-driven answers" },
  { value: "instruction_tuned", label: "Instruction Tuned", desc: "Task-specific instructions" },
  { value: "system", label: "System", desc: "System-style prompt" },
  { value: "custom", label: "Custom", desc: "Pick a library template manually" },
] as const;

interface PromptTemplateListItem {
  template_id: string;
  name: string;
}

interface PromptEngineeringState {
  enabled: boolean;
  promptType: string;
  maxTokensWarning: number;
  promptTemplateId: string;
}

function normalizeTemplateSelect(value: string | null | undefined): string {
  if (!value || !String(value).trim()) return SYSTEM_DEFAULT_SELECT_VALUE;
  return String(value).trim();
}

function persistTemplateId(selectValue: string): string | null {
  if (selectValue === SYSTEM_DEFAULT_SELECT_VALUE) return null;
  return selectValue;
}

function presetFromTemplateId(templateId: string | null): string {
  if (!templateId) return "rag_context";
  for (const [preset, id] of Object.entries(PROMPT_PRESET_LIBRARY)) {
    if (id === templateId) return preset;
  }
  return "custom";
}

export default function PromptEngineeringPage() {
  const { clientId } = useTenant();
  const [initial, setInitial] = useState<PromptEngineeringState | null>(null);
  const [state, setState] = useState<PromptEngineeringState | null>(null);
  const [templates, setTemplates] = useState<PromptTemplateListItem[]>([]);
  const [runtime, setRuntime] = useState<EffectiveTenantRuntime | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const { isDirty, markDirty, markClean } = useUnsavedChanges(false);

  const load = useCallback(async () => {
    if (!clientId) return;
    setLoading(true);
    setSaveError(null);
    try {
      const [pipelineRes, runtimeRes, templatesRes] = await Promise.all([
        apiClient.get(API.RAG_CONFIG.GET_PIPELINE(clientId)).catch(() => null),
        apiClient.get<EffectiveTenantRuntime>(API.MODELS.RUNTIME(clientId)).catch(() => null),
        apiClient.get(API.PROMPT_TEMPLATES.LIST()).catch(() => null),
      ]);

      const pipeline = pipelineRes?.data as
        | { retrieval?: { prompt_template_id?: string | null } }
        | undefined;
      const rt = runtimeRes?.data ?? null;
      setRuntime(rt);
      setTemplates(
        (templatesRes?.data as { templates?: PromptTemplateListItem[] })?.templates ?? []
      );

      const templateId = pipeline?.retrieval?.prompt_template_id ?? null;
      const promptType =
        rt?.prompt_node.configured_prompt_type ?? presetFromTemplateId(templateId);

      const next: PromptEngineeringState = {
        enabled: rt?.prompt_node.enabled ?? true,
        promptType,
        maxTokensWarning: 3000,
        promptTemplateId: normalizeTemplateSelect(templateId),
      };

      setInitial(next);
      setState(next);
      markClean();
    } finally {
      setLoading(false);
    }
  }, [clientId, markClean]);

  useEffect(() => {
    void load();
  }, [load]);

  const patchState = (patch: Partial<PromptEngineeringState>) => {
    setState((prev) => (prev ? { ...prev, ...patch } : prev));
    markDirty();
  };

  const handleStrategyChange = (promptType: string) => {
    const presetLibraryId = libraryIdForPromptPreset(promptType);
    patchState({
      promptType,
      ...(presetLibraryId ? { promptTemplateId: presetLibraryId } : {}),
    });
  };

  const handleDiscard = () => {
    if (initial) {
      setState(initial);
      markClean();
      setSaveError(null);
    } else {
      void load();
    }
  };

  const handleSave = useCallback(async () => {
    if (!clientId || !state) return;
    setSaving(true);
    setSaveError(null);
    try {
      const presetLibraryId = libraryIdForPromptPreset(state.promptType);
      const explicitTemplateId = persistTemplateId(state.promptTemplateId);

      const payload: Record<string, unknown> = {
        prompt: {
          enabled: state.enabled,
          prompt_type: state.promptType,
          max_tokens_warning: state.maxTokensWarning,
        },
        prompt_template_id: presetLibraryId ?? explicitTemplateId,
      };

      await apiClient.put(API.RAG_CONFIG.PUT_PIPELINE(clientId), payload);
      setInitial(state);
      markClean();

      const rtRes = await apiClient
        .get<EffectiveTenantRuntime>(API.MODELS.RUNTIME(clientId))
        .catch(() => null);
      if (rtRes?.data) setRuntime(rtRes.data);
    } catch (err: unknown) {
      const ax = err as { response?: { data?: { detail?: string } }; message?: string };
      setSaveError(
        ax?.response?.data?.detail ?? ax?.message ?? "Failed to save prompt settings."
      );
    } finally {
      setSaving(false);
    }
  }, [clientId, state, markClean]);

  const hasLoaded = useMemo(() => !loading && state != null, [loading, state]);
  const effective = state ?? {
    enabled: true,
    promptType: "rag_context",
    maxTokensWarning: 3000,
    promptTemplateId: SYSTEM_DEFAULT_SELECT_VALUE,
  };

  const effectiveTemplateLabel =
    runtime?.retrieval.prompt_ssot.effective_template_id ??
    runtime?.retrieval.prompt_template_id ??
    "—";

  return (
    <div className="space-y-6 pb-16">
      <div className="space-y-1">
        <h1 className="text-2xl font-bold text-slate-900">Prompt Engineering</h1>
        <p className="text-sm text-slate-500">
          Bind this tenant to a prompt strategy and Prompt Library template.
          Template content (system instructions, context format) is edited in the library.
        </p>
      </div>

      {!hasLoaded ? (
        <div className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-500">
          Loading prompt configuration…
        </div>
      ) : (
        <>
          {runtime?.warnings?.length ? (
            <RuntimeWarningBanner warnings={runtime.warnings} />
          ) : null}

          <SettingsCard
            title="Prompt Node"
            subtitle="Enable and choose a generation strategy"
            icon={FileText}
            jsonPaths={["prompt.enabled", "prompt.prompt_type", "prompt.max_tokens_warning"]}
            dirty={isDirty}
            helpText="The prompt node controls how retrieved context is assembled before the LLM call."
          >
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-semibold text-slate-600">Enabled</span>
              <button
                type="button"
                onClick={() => patchState({ enabled: !effective.enabled })}
                className={cn(
                  "relative h-5 w-9 rounded-full transition-colors",
                  effective.enabled ? "bg-violet-500" : "bg-slate-300"
                )}
              >
                <span
                  className={cn(
                    "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform",
                    effective.enabled ? "translate-x-4" : "translate-x-0.5"
                  )}
                />
              </button>
            </div>

            {effective.enabled && (
              <div className="space-y-4 border-t border-slate-100 pt-4">
                {runtime && (
                  <div className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs text-blue-900">
                    <span className="font-semibold">Effective template (SSOT):</span>{" "}
                    <code className="font-mono text-[11px] bg-blue-100/80 px-1 rounded">
                      {effectiveTemplateLabel}
                    </code>
                    <span className="text-blue-700/80 ml-1">
                      ({runtime.retrieval.prompt_ssot.source})
                    </span>
                  </div>
                )}

                <div>
                  <label className="text-xs font-semibold text-slate-600 mb-2 block">
                    Prompt Strategy
                  </label>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                    {PROMPT_STRATEGIES.map((opt) => {
                      const libId = libraryIdForPromptPreset(opt.value);
                      return (
                        <button
                          key={opt.value}
                          type="button"
                          onClick={() => handleStrategyChange(opt.value)}
                          className={cn(
                            "rounded-lg border px-3 py-2 text-left transition-all",
                            effective.promptType === opt.value
                              ? "border-violet-400 bg-violet-50 shadow-sm"
                              : "border-slate-200 bg-white hover:border-violet-200"
                          )}
                        >
                          <span className="text-xs font-semibold text-slate-800">{opt.label}</span>
                          <p className="text-[10px] text-slate-400 mt-0.5">{opt.desc}</p>
                          {libId && (
                            <p
                              className="text-[9px] font-mono text-violet-600 mt-1 truncate"
                              title={libId}
                            >
                              → {libId}
                            </p>
                          )}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div>
                  <label className="text-xs text-slate-600">Max token warning threshold</label>
                  <input
                    type="number"
                    value={effective.maxTokensWarning}
                    onChange={(e) =>
                      patchState({
                        maxTokensWarning: parseInt(e.target.value, 10) || 3000,
                      })
                    }
                    className="ml-2 w-24 h-8 rounded border border-slate-200 bg-slate-50 px-2 text-xs text-slate-700 focus:border-violet-400 focus:outline-none"
                  />
                </div>
              </div>
            )}
          </SettingsCard>

          <SettingsCard
            title="Prompt Library Binding"
            subtitle="Which template file backs generation"
            icon={Library}
            jsonPaths={["retrieval.prompt_template_id"]}
            dirty={isDirty}
            helpText="Presets auto-map to library ids on save. Custom strategies pick an explicit template."
          >
            <Select
              value={effective.promptTemplateId}
              onValueChange={(v) => patchState({ promptTemplateId: v })}
            >
              <SelectTrigger className="mt-1 bg-slate-50 text-slate-800 border-slate-200">
                <SelectValue placeholder="Select template" />
              </SelectTrigger>
              <SelectContent className="bg-white text-slate-900 border-slate-200">
                <SelectItem value={SYSTEM_DEFAULT_SELECT_VALUE}>
                  System default (builtin)
                </SelectItem>
                {templates.map((t) => (
                  <SelectItem key={t.template_id} value={t.template_id}>
                    {t.name} ({t.template_id})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {effective.promptType === "custom" && (
              <p className="mt-3 text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
                Inline <code className="text-[10px]">prompt.template</code> is not persisted.
                Create or edit templates in{" "}
                <Link href="/settings/prompt-builder" className="font-semibold underline">
                  Settings → Prompt Builder
                </Link>
                .
              </p>
            )}

            <p className="mt-3 text-[11px] text-slate-500">
              Full template CRUD (system instructions, context format, preview) lives in{" "}
              <Link href="/settings/prompt-builder" className="text-primary-600 underline">
                Prompt Builder
              </Link>
              .
            </p>
          </SettingsCard>

          {saveError && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
              {saveError}
            </div>
          )}
        </>
      )}

      <SaveBar
        isDirty={isDirty}
        onSave={handleSave}
        onDiscard={handleDiscard}
        saving={saving}
      />
    </div>
  );
}
