"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  Cpu,
  FileJson,
  FileSpreadsheet,
  FileText,
  Film,
  Globe,
  HardDrive,
  Image as ImageIcon,
  Mic,
  RefreshCw,
  ScanLine,
  Upload,
  Zap,
} from "lucide-react";
import { cn } from "@/lib/utils";

import { useTenant } from "@/contexts/TenantContext";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";
import { useUnsavedChanges } from "@/lib/useUnsavedChanges";
import { PARSER_HARDWARE, type ParserHardwareHint } from "@/lib/pipelineBlocks";

import { SettingsCard } from "@/components/ui/SettingsCard";
import { SaveBar } from "@/components/ui/SaveBar";
import InfoTooltip from "@/components/ui/InfoTooltip";

/* ─── Types & defaults ─── */

interface ParserDraft {
  enable_pdf: boolean;
  enable_docx: boolean;
  enable_xlsx: boolean;
  enable_csv: boolean;
  enable_pptx: boolean;
  enable_html: boolean;
  enable_json: boolean;
  enable_txt: boolean;
  enable_ocr: boolean;
  enable_audio: boolean;
  enable_video: boolean;
  enable_image: boolean;
  ocr_language: string;
  ocr_engine: string;
}

interface VisionDraft {
  ai_profile: "cpu" | "gpu" | "api" | "dist";
  video_vision_frames: number;
  video_max_duration_sec: number;
  enable_visual_explanation: boolean;
  enable_audio_language_detection: boolean;
  max_concurrent_media_tasks: number;
}

interface ConnectorsState {
  parsers: ParserDraft;
  max_file_size_mb: number;
  parse_workers: number;
  enable_visual_llm_explanation: boolean;
  visual_llm_concurrency: number;
  vision: VisionDraft;
}

const DEFAULT_PARSERS: ParserDraft = {
  enable_pdf: true,
  enable_docx: true,
  enable_xlsx: true,
  enable_csv: true,
  enable_pptx: true,
  enable_html: true,
  enable_json: true,
  enable_txt: true,
  enable_ocr: false,
  enable_audio: false,
  enable_video: false,
  enable_image: false,
  ocr_language: "eng",
  ocr_engine: "tesseract",
};

const DEFAULT_VISION: VisionDraft = {
  ai_profile: "cpu",
  video_vision_frames: 8,
  video_max_duration_sec: 1800,
  enable_visual_explanation: true,
  enable_audio_language_detection: true,
  max_concurrent_media_tasks: 2,
};

const DEFAULT_STATE: ConnectorsState = {
  parsers: DEFAULT_PARSERS,
  max_file_size_mb: 100,
  parse_workers: 4,
  enable_visual_llm_explanation: true,
  visual_llm_concurrency: 4,
  vision: DEFAULT_VISION,
};

type ParserToggleKey = keyof Pick<
  ParserDraft,
  | "enable_pdf"
  | "enable_docx"
  | "enable_xlsx"
  | "enable_csv"
  | "enable_pptx"
  | "enable_html"
  | "enable_json"
  | "enable_txt"
  | "enable_ocr"
  | "enable_audio"
  | "enable_video"
  | "enable_image"
>;

interface FormatMeta {
  key: ParserToggleKey;
  label: string;
  ext: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  group: "documents" | "web" | "media";
}

const FORMAT_CONNECTORS: FormatMeta[] = [
  { key: "enable_pdf", label: "PDF", ext: ".pdf", description: "Text extraction from PDF documents", icon: FileText, group: "documents" },
  { key: "enable_docx", label: "Word", ext: ".docx", description: "Microsoft Word documents", icon: FileText, group: "documents" },
  { key: "enable_pptx", label: "PowerPoint", ext: ".pptx", description: "Slide decks and presentations", icon: FileText, group: "documents" },
  { key: "enable_xlsx", label: "Excel", ext: ".xlsx", description: "Spreadsheets and tabular workbooks", icon: FileSpreadsheet, group: "documents" },
  { key: "enable_csv", label: "CSV", ext: ".csv", description: "Comma-separated tabular files", icon: FileSpreadsheet, group: "documents" },
  { key: "enable_html", label: "HTML", ext: ".html", description: "Web pages and markup exports", icon: Globe, group: "web" },
  { key: "enable_json", label: "JSON", ext: ".json", description: "Structured JSON payloads", icon: FileJson, group: "web" },
  { key: "enable_txt", label: "Plain text", ext: ".txt", description: "Unstructured text files", icon: FileText, group: "web" },
  { key: "enable_ocr", label: "OCR", ext: "scanned", description: "Optical character recognition for scans", icon: ScanLine, group: "media" },
  { key: "enable_image", label: "Images", ext: ".png/.jpg", description: "Image captioning via vision model", icon: ImageIcon, group: "media" },
  { key: "enable_audio", label: "Audio", ext: ".mp3/.wav", description: "Speech transcription (Whisper)", icon: Mic, group: "media" },
  { key: "enable_video", label: "Video", ext: ".mp4", description: "Frame sampling + audio extraction", icon: Film, group: "media" },
];

const DOCUMENT_KEYS = FORMAT_CONNECTORS.filter((f) => f.group === "documents").map((f) => f.key);
const WEB_KEYS = FORMAT_CONNECTORS.filter((f) => f.group === "web").map((f) => f.key);
const MEDIA_KEYS = FORMAT_CONNECTORS.filter((f) => f.group === "media").map((f) => f.key);

function mergeParsersFromApi(raw: unknown): ParserDraft {
  const base = { ...DEFAULT_PARSERS };
  if (!raw || typeof raw !== "object") return base;
  const o = raw as Record<string, unknown>;
  for (const key of Object.keys(DEFAULT_PARSERS) as (keyof ParserDraft)[]) {
    if (typeof o[key] === "boolean") {
      (base as Record<string, unknown>)[key] = o[key];
    } else if (key === "ocr_language" || key === "ocr_engine") {
      if (typeof o[key] === "string" && o[key].trim()) {
        (base as Record<string, unknown>)[key] = o[key].trim();
      }
    }
  }
  return base;
}

function mergeVisionFromApi(raw: unknown): VisionDraft {
  const base = { ...DEFAULT_VISION };
  if (!raw || typeof raw !== "object") return base;
  const o = raw as Record<string, unknown>;
  const profile = String(o.ai_profile ?? "").trim().toLowerCase();
  if (profile === "cpu" || profile === "gpu" || profile === "api" || profile === "dist") {
    base.ai_profile = profile;
  }
  if (o.video_vision_frames != null) {
    const n = Number(o.video_vision_frames);
    if (Number.isFinite(n) && n >= 1) base.video_vision_frames = Math.round(n);
  }
  if (o.video_max_duration_sec != null) {
    const n = Number(o.video_max_duration_sec);
    if (Number.isFinite(n) && n >= 1) base.video_max_duration_sec = Math.round(n);
  }
  if (typeof o.enable_visual_explanation === "boolean") {
    base.enable_visual_explanation = o.enable_visual_explanation;
  }
  if (typeof o.enable_audio_language_detection === "boolean") {
    base.enable_audio_language_detection = o.enable_audio_language_detection;
  }
  if (o.max_concurrent_media_tasks != null) {
    const n = Number(o.max_concurrent_media_tasks);
    if (Number.isFinite(n) && n >= 1) base.max_concurrent_media_tasks = Math.round(n);
  }
  return base;
}

function mergeConnectorsFromApi(parserRaw: unknown, ingestionRaw: unknown): ConnectorsState {
  const base = { ...DEFAULT_STATE, parsers: mergeParsersFromApi(parserRaw) };
  if (!ingestionRaw || typeof ingestionRaw !== "object") return base;
  const ing = ingestionRaw as Record<string, unknown>;

  if (ing.max_file_size_mb != null) {
    const n = Number(ing.max_file_size_mb);
    if (Number.isFinite(n) && n >= 1) base.max_file_size_mb = Math.round(n);
  }
  if (typeof ing.enable_visual_llm_explanation === "boolean") {
    base.enable_visual_llm_explanation = ing.enable_visual_llm_explanation;
  }
  if (ing.visual_llm_concurrency != null) {
    const n = Number(ing.visual_llm_concurrency);
    if (Number.isFinite(n) && n >= 1) base.visual_llm_concurrency = Math.min(32, Math.round(n));
  }
  base.vision = mergeVisionFromApi(ing.vision);

  const phantom = ing.phantom;
  if (phantom && typeof phantom === "object") {
    const pw = Number((phantom as Record<string, unknown>).parse_workers);
    if (Number.isFinite(pw) && pw >= 1) base.parse_workers = Math.round(pw);
  }

  return base;
}

function countActiveFormats(parsers: ParserDraft): number {
  return FORMAT_CONNECTORS.filter((f) => parsers[f.key]).length;
}

function anyMediaEnabled(parsers: ParserDraft): boolean {
  return MEDIA_KEYS.some((k) => parsers[k]);
}

/* ─── UI helpers ─── */

function HardwareBadge({ hint }: { hint: ParserHardwareHint }) {
  const styles: Record<ParserHardwareHint, string> = {
    none: "bg-slate-100 text-slate-600 border-slate-200",
    vision: "bg-violet-50 text-violet-700 border-violet-200",
    gpu: "bg-amber-50 text-amber-800 border-amber-200",
  };
  const labels: Record<ParserHardwareHint, string> = {
    none: "CPU-light",
    vision: "Vision",
    gpu: "GPU",
  };
  return (
    <span className={cn("rounded-full border px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide", styles[hint])}>
      {labels[hint]}
    </span>
  );
}

function ToggleSwitch({
  checked,
  onChange,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative h-5 w-9 shrink-0 rounded-full transition-colors disabled:opacity-50",
        checked ? "bg-primary-500" : "bg-slate-300"
      )}
    >
      <span
        className={cn(
          "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform",
          checked ? "translate-x-4" : "translate-x-0.5"
        )}
      />
    </button>
  );
}

function FormatConnectorRow({
  meta,
  checked,
  onChange,
}: {
  meta: FormatMeta;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  const Icon = meta.icon;
  const hw = PARSER_HARDWARE[meta.key] ?? "none";
  return (
    <div
      className={cn(
        "flex items-center gap-3 rounded-lg border px-3 py-2.5 transition-colors",
        checked ? "border-primary-200 bg-primary-50/40" : "border-slate-100 bg-slate-50/60"
      )}
    >
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg",
          checked ? "bg-primary-100 text-primary-700" : "bg-white text-slate-400"
        )}
      >
        <Icon className="h-4 w-4" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs font-semibold text-slate-800">{meta.label}</span>
          <code className="text-[10px] text-slate-400">{meta.ext}</code>
          <HardwareBadge hint={hw} />
        </div>
        <p className="text-[10px] text-slate-500 leading-snug mt-0.5">{meta.description}</p>
        <code className="text-[9px] text-slate-400 font-mono">parsers.{meta.key}</code>
      </div>
      <ToggleSwitch checked={checked} onChange={onChange} />
    </div>
  );
}

function NumberField({
  label,
  jsonPath,
  value,
  onChange,
  min,
  max,
  help,
}: {
  label: string;
  jsonPath: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  help?: string;
}) {
  return (
    <label className="block text-xs">
      <div className="flex items-center gap-1.5 mb-1">
        <span className="font-medium text-slate-700">{label}</span>
        {help ? <InfoTooltip text={help} /> : null}
      </div>
      <input
        type="number"
        min={min}
        max={max}
        value={value}
        onChange={(e) => {
          const n = parseInt(e.target.value, 10);
          if (Number.isFinite(n)) onChange(Math.min(max, Math.max(min, n)));
        }}
        className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200"
      />
      <code className="mt-1 block text-[10px] text-slate-400 font-mono">{jsonPath}</code>
    </label>
  );
}

function SummaryChip({ label, value, tone = "default" }: { label: string; value: string; tone?: "default" | "success" | "warning" }) {
  const toneClass =
    tone === "success"
      ? "text-emerald-700"
      : tone === "warning"
        ? "text-amber-700"
        : "text-slate-800";
  return (
    <div className="rounded-xl border border-slate-200/60 bg-white px-4 py-3 text-center shadow-sm">
      <p className={cn("text-lg font-bold", toneClass)}>{value}</p>
      <p className="text-[11px] text-slate-400 mt-0.5">{label}</p>
    </div>
  );
}

/** Short display labels for the live capability badge row. */
const CAPABILITY_BADGE_LABELS: Partial<Record<ParserToggleKey, string>> = {
  enable_docx: "Word",
  enable_pptx: "PPT",
  enable_xlsx: "Excel",
  enable_txt: "Text",
  enable_image: "Images",
};

type CapabilityTone = "document" | "web" | "media" | "vision";

interface ActiveCapability {
  id: string;
  label: string;
  tone: CapabilityTone;
}

function buildEnabledCapabilities(draft: ConnectorsState): ActiveCapability[] {
  const badges: ActiveCapability[] = [];

  for (const meta of FORMAT_CONNECTORS) {
    if (!draft.parsers[meta.key]) continue;
    badges.push({
      id: meta.key,
      label: CAPABILITY_BADGE_LABELS[meta.key] ?? meta.label,
      tone: meta.group === "web" ? "web" : meta.group === "media" ? "media" : "document",
    });
  }

  if (draft.vision.enable_visual_explanation || draft.enable_visual_llm_explanation) {
    badges.push({ id: "vision", label: "Vision", tone: "vision" });
  }

  return badges;
}

const CAPABILITY_TONE_CLASSES: Record<CapabilityTone, string> = {
  document: "bg-primary-50 text-primary-800 border-primary-200 ring-primary-100",
  web: "bg-sky-50 text-sky-800 border-sky-200 ring-sky-100",
  media: "bg-violet-50 text-violet-800 border-violet-200 ring-violet-100",
  vision: "bg-amber-50 text-amber-900 border-amber-200 ring-amber-100",
};

function ActiveCapabilityBadge({ label, tone }: { label: string; tone: CapabilityTone }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-bold tracking-wide",
        "shadow-sm ring-1 ring-inset transition-colors duration-150",
        CAPABILITY_TONE_CLASSES[tone]
      )}
    >
      {label}
    </span>
  );
}

function EnabledFormatsRow({ draft }: { draft: ConnectorsState }) {
  const capabilities = useMemo(() => buildEnabledCapabilities(draft), [draft]);

  return (
    <div className="rounded-xl border border-slate-200/60 bg-white px-4 py-3 shadow-sm">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-slate-500 shrink-0">
          Enabled formats
        </span>
        {capabilities.length === 0 ? (
          <span className="text-xs text-slate-400 italic">No connectors enabled</span>
        ) : (
          <div className="flex flex-wrap gap-1.5 min-w-0 flex-1">
            {capabilities.map((cap) => (
              <ActiveCapabilityBadge key={cap.id} label={cap.label} tone={cap.tone} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ConnectorsSummaryStrip({ draft }: { draft: ConnectorsState }) {
  const activeCount = countActiveFormats(draft.parsers);
  const mediaOn = anyMediaEnabled(draft.parsers);

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <SummaryChip label="Active connectors" value={`${activeCount} / ${FORMAT_CONNECTORS.length}`} tone="success" />
        <SummaryChip
          label="Multimodal"
          value={mediaOn ? "Enabled" : "Off"}
          tone={mediaOn ? "warning" : "default"}
        />
        <SummaryChip label="Max file size" value={`${draft.max_file_size_mb} MB`} />
        <SummaryChip label="Parse workers" value={String(draft.parse_workers)} />
      </div>
      <EnabledFormatsRow draft={draft} />
    </div>
  );
}

/* ─── Page ─── */

export default function IngestionConnectorsPage() {
  const { clientId } = useTenant();
  const [initial, setInitial] = useState<ConnectorsState | null>(null);
  const [state, setState] = useState<ConnectorsState | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const { isDirty, markDirty, markClean } = useUnsavedChanges(false);

  const load = useCallback(async () => {
    if (!clientId) return;
    setLoading(true);
    setSaveError(null);
    try {
      const res = await apiClient.get(API.RAG_CONFIG.PIPELINE_PLUGGABLE_GET(clientId));
      const data = res.data as { parser?: unknown; ingestion?: unknown };
      const next = mergeConnectorsFromApi(data.parser, data.ingestion);
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

  const patch = useCallback(
    (updater: (prev: ConnectorsState) => ConnectorsState) => {
      setState((prev) => {
        if (!prev) return prev;
        const next = updater(prev);
        markDirty();
        return next;
      });
    },
    [markDirty]
  );

  const patchParser = useCallback(
    (key: keyof ParserDraft, value: boolean | string) => {
      patch((prev) => ({
        ...prev,
        parsers: { ...prev.parsers, [key]: value },
      }));
    },
    [patch]
  );

  const setAllKeys = useCallback(
    (keys: ParserToggleKey[], enabled: boolean) => {
      patch((prev) => {
        const parsers = { ...prev.parsers };
        for (const k of keys) parsers[k] = enabled;
        return { ...prev, parsers };
      });
    },
    [patch]
  );

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
      await apiClient.patch(API.RAG_CONFIG.PIPELINE_PLUGGABLE_PATCH(clientId), {
        parser: { ...state.parsers },
        ingestion: {
          max_file_size_mb: state.max_file_size_mb,
          enable_visual_llm_explanation: state.enable_visual_llm_explanation,
          visual_llm_concurrency: state.visual_llm_concurrency,
          vision: { ...state.vision },
          phantom: { parse_workers: state.parse_workers },
        },
      });
      setInitial(state);
      markClean();
    } catch (err: unknown) {
      const ax = err as { response?: { data?: { detail?: string } }; message?: string };
      setSaveError(ax?.response?.data?.detail ?? ax?.message ?? "Failed to save connector settings.");
    } finally {
      setSaving(false);
    }
  }, [clientId, state, markClean]);

  const draft = state ?? DEFAULT_STATE;
  const hasLoaded = useMemo(() => !loading && state != null, [loading, state]);
  const mediaOn = anyMediaEnabled(draft.parsers);

  return (
    <div className="space-y-6 pb-16">
      <div className="space-y-1">
        <h1 className="text-2xl font-bold text-slate-900">Connectors</h1>
        <p className="text-sm text-slate-500 max-w-3xl">
          Configure which file formats and media types this tenant can ingest, plus intake limits and
          multimodal processing. Settings persist to{" "}
          <code className="text-xs">parsers.*</code> and{" "}
          <code className="text-xs">ingestion.*</code> in tenant JSON.
        </p>
      </div>

      {!hasLoaded ? (
        <div className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-500">
          Loading connector configuration…
        </div>
      ) : (
        <>
          <ConnectorsSummaryStrip draft={draft} />

          {/* Related workflows */}
          <div className="grid gap-3 sm:grid-cols-3">
            {[
              { href: "/ingestion/files", label: "Ingested Files", desc: "Browse uploaded documents", icon: Upload },
              { href: "/dashboard/sync", label: "Sync Monitor", desc: "DB ↔ vector store orphans", icon: RefreshCw },
              { href: "/ingestion/chunking-tokenization", label: "Chunking", desc: "Post-parse splitting rules", icon: FileText },
            ].map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="group flex items-center gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3 shadow-sm hover:border-primary-200 hover:shadow-md transition-all"
              >
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-100 text-slate-600 group-hover:bg-primary-50 group-hover:text-primary-600">
                  <link.icon className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-semibold text-slate-800">{link.label}</p>
                  <p className="text-[10px] text-slate-500">{link.desc}</p>
                </div>
                <ArrowRight className="h-3.5 w-3.5 text-slate-300 group-hover:text-primary-500" />
              </Link>
            ))}
          </div>

          {/* Intake gates */}
          <SettingsCard
            title="Intake Gates & Throughput"
            subtitle="Reject oversized files and tune parse parallelism"
            icon={HardDrive}
            jsonPaths={["ingestion.max_file_size_mb", "ingestion.phantom.parse_workers"]}
            dirty={isDirty}
            helpText="These limits apply before format-specific parsers run. Parse workers control how many files are parsed concurrently during bulk ingestion."
          >
            <div className="grid gap-4 sm:grid-cols-2">
              <NumberField
                label="Maximum file size (MB)"
                jsonPath="ingestion.max_file_size_mb"
                value={draft.max_file_size_mb}
                onChange={(v) => patch((p) => ({ ...p, max_file_size_mb: v }))}
                min={1}
                max={2048}
                help="Files larger than this are rejected at upload. Lower for memory-constrained deployments."
              />
              <NumberField
                label="Parse workers"
                jsonPath="ingestion.phantom.parse_workers"
                value={draft.parse_workers}
                onChange={(v) => patch((p) => ({ ...p, parse_workers: v }))}
                min={1}
                max={32}
                help="Parallel file parsing threads (PHANTOM). Increase on multi-core hosts; watch RAM when multimodal parsers are enabled."
              />
            </div>
          </SettingsCard>

          {/* Document formats */}
          <SettingsCard
            title="Document Format Connectors"
            subtitle="Office, spreadsheet, and plain-text parsers"
            icon={FileText}
            jsonPaths={["parsers.enable_pdf", "parsers.enable_docx", "parsers.enable_xlsx"]}
            dirty={isDirty}
            headerActions={
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setAllKeys(DOCUMENT_KEYS, true)}
                  className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-[10px] font-medium text-slate-600 hover:bg-slate-50"
                >
                  Enable all
                </button>
                <button
                  type="button"
                  onClick={() => setAllKeys(DOCUMENT_KEYS, false)}
                  className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-[10px] font-medium text-slate-600 hover:bg-slate-50"
                >
                  Disable all
                </button>
              </div>
            }
          >
            <div className="space-y-2">
              {FORMAT_CONNECTORS.filter((f) => f.group === "documents").map((meta) => (
                <FormatConnectorRow
                  key={meta.key}
                  meta={meta}
                  checked={draft.parsers[meta.key]}
                  onChange={(v) => patchParser(meta.key, v)}
                />
              ))}
            </div>
          </SettingsCard>

          {/* Web & structured */}
          <SettingsCard
            title="Web & Structured Data"
            subtitle="HTML, JSON, and plain-text connectors"
            icon={Globe}
            jsonPaths={["parsers.enable_html", "parsers.enable_json", "parsers.enable_txt"]}
            dirty={isDirty}
          >
            <div className="space-y-2">
              {FORMAT_CONNECTORS.filter((f) => f.group === "web").map((meta) => (
                <FormatConnectorRow
                  key={meta.key}
                  meta={meta}
                  checked={draft.parsers[meta.key]}
                  onChange={(v) => patchParser(meta.key, v)}
                />
              ))}
            </div>
          </SettingsCard>

          {/* Multimodal */}
          <SettingsCard
            title="Multimodal & Media Connectors"
            subtitle="OCR, vision, audio, and video — higher compute cost"
            icon={Film}
            jsonPaths={["parsers.enable_ocr", "parsers.enable_image", "parsers.enable_audio", "parsers.enable_video"]}
            dirty={isDirty}
            statusLabel={mediaOn ? "Heavy compute" : undefined}
            statusTone="warning"
            helpText="Vision and GPU badges indicate parsers that load local models or call cloud vision APIs. Disable formats you do not ingest to reduce latency and cost."
          >
            {mediaOn && (
              <div className="mb-4 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50/80 px-3 py-2.5 text-[11px] text-amber-900">
                <Zap className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                Multimodal connectors are active — ensure vision profile and secrets are configured under Pipeline Builder.
              </div>
            )}
            <div className="space-y-2">
              {FORMAT_CONNECTORS.filter((f) => f.group === "media").map((meta) => (
                <FormatConnectorRow
                  key={meta.key}
                  meta={meta}
                  checked={draft.parsers[meta.key]}
                  onChange={(v) => patchParser(meta.key, v)}
                />
              ))}
            </div>
          </SettingsCard>

          {/* OCR engine */}
          {draft.parsers.enable_ocr && (
            <SettingsCard
              title="OCR Engine"
              subtitle="Settings used when OCR connector is enabled"
              icon={ScanLine}
              jsonPaths={["parsers.ocr_language", "parsers.ocr_engine"]}
              dirty={isDirty}
            >
              <div className="grid gap-4 sm:grid-cols-2">
                <label className="block text-xs">
                  <span className="font-medium text-slate-700">OCR language</span>
                  <input
                    type="text"
                    value={draft.parsers.ocr_language}
                    onChange={(e) => patchParser("ocr_language", e.target.value)}
                    placeholder="eng"
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200"
                  />
                  <code className="mt-1 block text-[10px] text-slate-400 font-mono">parsers.ocr_language</code>
                </label>
                <label className="block text-xs">
                  <span className="font-medium text-slate-700">OCR engine</span>
                  <select
                    value={draft.parsers.ocr_engine}
                    onChange={(e) => patchParser("ocr_engine", e.target.value)}
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200"
                  >
                    <option value="tesseract">tesseract</option>
                  </select>
                  <code className="mt-1 block text-[10px] text-slate-400 font-mono">parsers.ocr_engine</code>
                </label>
              </div>
            </SettingsCard>
          )}

          {/* Vision & chart explanation */}
          <SettingsCard
            title="Vision & Chart Explanation"
            subtitle="Multimodal execution profile and visual LLM during ingest"
            icon={Cpu}
            jsonPaths={[
              "ingestion.vision.ai_profile",
              "ingestion.enable_visual_llm_explanation",
              "ingestion.visual_llm_concurrency",
            ]}
            dirty={isDirty}
            helpText="Controls how charts, tables, and media frames are explained before embedding. API profile uses cloud vision; cpu/gpu use local models."
          >
            <div className="space-y-4">
              <div className="flex items-center justify-between gap-4 rounded-lg border border-slate-100 bg-slate-50/80 px-3 py-2.5">
                <div>
                  <span className="text-xs font-semibold text-slate-700">Visual LLM explanation</span>
                  <code className="block text-[10px] text-slate-400 font-mono mt-0.5">
                    ingestion.enable_visual_llm_explanation
                  </code>
                </div>
                <ToggleSwitch
                  checked={draft.enable_visual_llm_explanation}
                  onChange={(v) => patch((p) => ({ ...p, enable_visual_llm_explanation: v }))}
                />
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <label className="block text-xs">
                  <span className="font-medium text-slate-700">Vision execution profile</span>
                  <select
                    value={draft.vision.ai_profile}
                    onChange={(e) =>
                      patch((p) => ({
                        ...p,
                        vision: {
                          ...p.vision,
                          ai_profile: e.target.value as VisionDraft["ai_profile"],
                        },
                      }))
                    }
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200"
                  >
                    <option value="cpu">cpu — local CPU models</option>
                    <option value="gpu">gpu — local GPU models</option>
                    <option value="api">api — cloud vision provider</option>
                    <option value="dist">dist — distributed workers</option>
                  </select>
                  <code className="mt-1 block text-[10px] text-slate-400 font-mono">ingestion.vision.ai_profile</code>
                </label>
                <NumberField
                  label="Visual LLM concurrency"
                  jsonPath="ingestion.visual_llm_concurrency"
                  value={draft.visual_llm_concurrency}
                  onChange={(v) => patch((p) => ({ ...p, visual_llm_concurrency: v }))}
                  min={1}
                  max={32}
                  help="Max concurrent chart/table explanation calls during ingestion."
                />
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <NumberField
                  label="Video vision frames"
                  jsonPath="ingestion.vision.video_vision_frames"
                  value={draft.vision.video_vision_frames}
                  onChange={(v) =>
                    patch((p) => ({ ...p, vision: { ...p.vision, video_vision_frames: v } }))
                  }
                  min={1}
                  max={64}
                  help="Frames sampled per video for vision captioning."
                />
                <NumberField
                  label="Max video duration (sec)"
                  jsonPath="ingestion.vision.video_max_duration_sec"
                  value={draft.vision.video_max_duration_sec}
                  onChange={(v) =>
                    patch((p) => ({ ...p, vision: { ...p.vision, video_max_duration_sec: v } }))
                  }
                  min={60}
                  max={7200}
                  help="Videos longer than this are truncated or rejected."
                />
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <div className="flex items-center justify-between gap-3 rounded-lg border border-slate-100 bg-slate-50/80 px-3 py-2.5">
                  <div>
                    <span className="text-xs font-semibold text-slate-700">Visual explanation</span>
                    <code className="block text-[10px] text-slate-400 font-mono">ingestion.vision.enable_visual_explanation</code>
                  </div>
                  <ToggleSwitch
                    checked={draft.vision.enable_visual_explanation}
                    onChange={(v) =>
                      patch((p) => ({ ...p, vision: { ...p.vision, enable_visual_explanation: v } }))
                    }
                  />
                </div>
                <div className="flex items-center justify-between gap-3 rounded-lg border border-slate-100 bg-slate-50/80 px-3 py-2.5">
                  <div>
                    <span className="text-xs font-semibold text-slate-700">Audio language detection</span>
                    <code className="block text-[10px] text-slate-400 font-mono">ingestion.vision.enable_audio_language_detection</code>
                  </div>
                  <ToggleSwitch
                    checked={draft.vision.enable_audio_language_detection}
                    onChange={(v) =>
                      patch((p) => ({
                        ...p,
                        vision: { ...p.vision, enable_audio_language_detection: v },
                      }))
                    }
                  />
                </div>
              </div>

              <NumberField
                label="Max concurrent media tasks"
                jsonPath="ingestion.vision.max_concurrent_media_tasks"
                value={draft.vision.max_concurrent_media_tasks}
                onChange={(v) =>
                  patch((p) => ({ ...p, vision: { ...p.vision, max_concurrent_media_tasks: v } }))
                }
                min={1}
                max={16}
                help="Cap parallel OCR / audio / video jobs to protect GPU and API rate limits."
              />
            </div>
          </SettingsCard>

          {saveError && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
              {saveError}
            </div>
          )}
        </>
      )}

      <SaveBar isDirty={isDirty} onSave={handleSave} onDiscard={handleDiscard} saving={saving} />
    </div>
  );
}
