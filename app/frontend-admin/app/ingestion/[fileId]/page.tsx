"use client";

import { useEffect, useState, type ComponentType } from "react";
import { useParams } from "next/navigation";
import Button from "@/components/ui/Button";
import Modal from "@/components/ui/Modal";
import apiClient from "@/lib/apiClient";
import { cn } from "@/lib/utils";
import { useFormatDate } from "@/lib/useHydrated";
import { useAuth } from "@/lib/useAuth";
import { AlertTriangle, ArrowLeft, FileText, Hash, Info, Layers3, Link2, Loader2, Pencil, Save, SearchCheck, Sparkles } from "lucide-react";

interface Chunk {
  id: string;
  chunk_index: number;
  page_number?: number | null;
  parent_chunk_id?: string | null;
  text: string;
  cleaned_text: string;
  tokens?: number;
  source_type?: string | null;
  semantic_hash: string;
  confidence: number;
  is_duplicate: boolean;
  duplicate_of?: string | null;
  similarity_score?: number | null;
  global_content_id: string | null;
  gci_occurrence_count: number | null;
  meta_data?: Record<string, any>;
  reasoning_ingestion?: Record<string, any>;
}

interface FileDetails {
  id: string;
  file_name: string;
  file_type: string;
  total_chunks: number;
  unique_chunks: number;
  duplicate_chunks: number;
  dedup_ratio?: number | null;
  status: string;
  updated_at?: string;
  created_at?: string;
}

interface DisplayRow {
  id: string;
  chunk_index: number | string;
  text: string;
  layer: "L1 Exact" | "L2 GCI" | "L3 Review" | "Unique";
  reference: string;
  semantics?: string;
  matchSurface?: string;
  isInferred?: boolean;
  duplicateCopies?: number;
}

function truncate(value: string, size = 150) {
  return value.length > size ? `${value.slice(0, size)}...` : value;
}

function shortId(value: string | null | undefined) {
  if (!value) return "n/a";
  return value.length > 18 ? `${value.slice(0, 8)}...${value.slice(-6)}` : value;
}

function toneClasses(layer: DisplayRow["layer"]) {
  if (layer === "L1 Exact") return "border-amber-200 bg-amber-100 text-amber-800";
  if (layer === "L2 GCI") return "border-blue-200 bg-blue-100 text-blue-800";
  if (layer === "L3 Review") return "border-rose-200 bg-rose-100 text-rose-800";
  return "border-emerald-200 bg-emerald-100 text-emerald-800";
}

function duplicateReason(layer: DisplayRow["layer"]) {
  if (layer === "L1 Exact") return "Exact same value repeated in this file";
  if (layer === "L2 GCI") return "Already exists in Global Content Index";
  if (layer === "L3 Review") return "Possible semantic duplicate needing review";
  return "Unique row kept after dedup";
}

function matchSurfaceLabel(layer: DisplayRow["layer"], fileName: string) {
  if (layer === "L1 Exact") return fileName;
  if (layer === "L2 GCI") return "Global Content Index";
  if (layer === "L3 Review") return "Current ingestion review";
  return fileName;
}

function matchingSemanticsText(layer: DisplayRow["layer"], reference: string, duplicateCopies?: number, isInferred?: boolean) {
  if (layer === "L1 Exact") {
    return isInferred
      ? `Same semantic hash repeated ${duplicateCopies ?? 0} more times`
      : `Exact semantic-hash repeat: ${reference}`;
  }
  if (layer === "L2 GCI") return `Cross-file match via GCI evidence: ${reference}`;
  if (layer === "L3 Review") return `Semantic review candidate using hash trail: ${reference}`;
  return `Unique semantic hash: ${reference}`;
}

export default function FileDetailPage() {
  const { fileId } = useParams<{ fileId: string }>();
  const { role } = useAuth();
  const canEditChunks = role === "admin" || role === "editor";
  const [file, setFile] = useState<FileDetails | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingChunk, setEditingChunk] = useState<Chunk | null>(null);
  const [draftText, setDraftText] = useState("");
  const [llmMode, setLlmMode] = useState<"" | "factual" | "creative">("");
  const [savingChunk, setSavingChunk] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const { formatDateTime } = useFormatDate();

  useEffect(() => {
    if (!fileId) return;
    (async () => {
      try {
        const [fRes, cRes] = await Promise.all([
          apiClient.get(`/api/v2/ingestion-admin/files/${fileId}`),
          apiClient.get(`/api/v2/ingestion-admin/files/${fileId}/chunks`),
        ]);
        setFile(fRes.data);
        setChunks(cRes.data ?? []);
      } catch (err) {
        console.error("Failed to fetch file details:", err);
      } finally {
        setLoading(false);
      }
    })();
  }, [fileId]);

  function openEditor(chunk: Chunk) {
    setEditingChunk(chunk);
    setDraftText(chunk.cleaned_text || chunk.text || "");
    setLlmMode("");
    setSaveMessage(null);
  }

  function closeEditor() {
    if (savingChunk) return;
    setEditingChunk(null);
    setDraftText("");
    setLlmMode("");
    setSaveMessage(null);
  }

  async function saveChunkEdit() {
    if (!editingChunk) return;
    const nextText = draftText.trim();
    if (!nextText) {
      setSaveMessage("Chunk text cannot be empty.");
      return;
    }

    setSavingChunk(true);
    setSaveMessage(null);
    try {
      await apiClient.put(`/api/v2/ingestion-admin/chunks/${editingChunk.id}`, {
        cleaned_text: nextText,
        llm_mode: llmMode || null,
      });

      setChunks((prev) =>
        prev.map((c) =>
          c.id === editingChunk.id
            ? {
                ...c,
                cleaned_text: nextText,
                meta_data: {
                  ...(c.meta_data || {}),
                  manually_edited: true,
                  llm_mode: llmMode || null,
                },
              }
            : c
        )
      );
      setSaveMessage("Chunk saved and re-embedded.");
      setTimeout(() => closeEditor(), 700);
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setSaveMessage(typeof detail === "string" ? detail : "Failed to save chunk.");
    } finally {
      setSavingChunk(false);
    }
  }

  const sortedChunks = [...chunks].sort((a, b) => a.chunk_index - b.chunk_index);
  const groups = new Map<string, Chunk[]>();
  sortedChunks.forEach((chunk) => {
    const current = groups.get(chunk.semantic_hash) ?? [];
    current.push(chunk);
    groups.set(chunk.semantic_hash, current);
  });

  const l1Groups = Array.from(groups.entries())
    .filter(([, members]) => members.length > 1)
    .map(([semanticHash, members]) => ({ semanticHash, members: members.sort((a, b) => a.chunk_index - b.chunk_index) }))
    .sort((a, b) => b.members.length - a.members.length);

  const l1RepeatIds = new Set<string>();
  l1Groups.forEach(group => group.members.slice(1).forEach(member => l1RepeatIds.add(member.id)));

  const l1DuplicateCount = l1Groups.reduce((sum, group) => sum + Math.max(group.members.length - 1, 0), 0);
  const l2Duplicates = sortedChunks.filter((chunk) => chunk.is_duplicate && Boolean(chunk.global_content_id));
  const l3Candidates = sortedChunks.filter((chunk) => chunk.is_duplicate && !chunk.global_content_id && !l1RepeatIds.has(chunk.id));

  if (loading) {
    return <div className="flex justify-center gap-3 py-20 text-slate-500"><Loader2 className="h-5 w-5 animate-spin" /> Loading dedup inspector...</div>;
  }

  if (!file) {
    return <p className="py-16 text-center text-slate-500">File not found.</p>;
  }

  const duplicateRows = sortedChunks.filter((chunk) => chunk.is_duplicate);
  const uniqueRows = sortedChunks.filter((chunk) => !chunk.is_duplicate);
  const hasDuplicates = file.duplicate_chunks > 0;
  const inferredL1Only =
    duplicateRows.length === 0 &&
    file.duplicate_chunks > 0 &&
    uniqueRows.length === 1 &&
    file.unique_chunks === 1;
  const inferredL1Rows: DisplayRow[] = inferredL1Only
    ? [{
        id: `inferred-${uniqueRows[0].id}`,
        chunk_index: `${uniqueRows[0].chunk_index} x${file.total_chunks}`,
        text: uniqueRows[0].cleaned_text || uniqueRows[0].text,
        layer: "L1 Exact",
        reference: `${shortId(uniqueRows[0].semantic_hash)} • repeated ${file.duplicate_chunks} duplicate copies`,
        isInferred: true,
        duplicateCopies: file.duplicate_chunks,
      }]
    : [];
  const duplicateTableRows: DisplayRow[] = duplicateRows.length > 0
    ? duplicateRows.map((chunk) => {
        const isL1Repeat = l1RepeatIds.has(chunk.id);
        return {
          id: chunk.id,
          chunk_index: chunk.chunk_index,
          text: chunk.cleaned_text || chunk.text,
          layer: isL1Repeat ? "L1 Exact" as const : chunk.global_content_id ? "L2 GCI" as const : "L3 Review" as const,
          reference: chunk.global_content_id
            ? `${shortId(chunk.global_content_id)} (${chunk.gci_occurrence_count ?? "n/a"})`
            : shortId(chunk.semantic_hash),
          isInferred: false,
        };
      })
    : inferredL1Rows;
  const l1DisplayCount = inferredL1Only ? file.duplicate_chunks : l1DuplicateCount;
  const l2DisplayCount = l2Duplicates.length;
  const l3DisplayCount = l3Candidates.length;
  const dominantLayer =
    l1DisplayCount >= l2DisplayCount && l1DisplayCount >= l3DisplayCount
      ? "L1 exact repeats"
      : l2DisplayCount >= l3DisplayCount
        ? "L2 GCI-linked duplicates"
        : "L3 semantic review items";
  const duplicateCoverage = file.total_chunks > 0 ? ((file.duplicate_chunks / file.total_chunks) * 100).toFixed(1) : "0.0";
  const duplicateDisplayRows: DisplayRow[] = duplicateRows.length > 0
    ? duplicateRows.map((chunk) => {
        const isL1Repeat = l1RepeatIds.has(chunk.id);
        return {
          id: chunk.id,
          chunk_index: chunk.chunk_index,
          text: chunk.cleaned_text || chunk.text,
          layer: isL1Repeat ? "L1 Exact" : chunk.global_content_id ? "L2 GCI" : "L3 Review",
          reference: chunk.global_content_id
            ? `${shortId(chunk.global_content_id)} (${chunk.gci_occurrence_count ?? "n/a"})`
            : shortId(chunk.semantic_hash),
        };
      })
    : inferredL1Rows;
  const uniqueDisplayRows: DisplayRow[] = uniqueRows.map((chunk) => ({
    id: chunk.id,
    chunk_index: chunk.chunk_index,
    text: chunk.cleaned_text || chunk.text,
    layer: "Unique",
    reference: shortId(chunk.semantic_hash),
  }));

  return (
    <div className="space-y-6">
      <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="space-y-3">
            <div className="inline-flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold uppercase tracking-[0.24em] text-slate-600">
              <Layers3 className="h-3.5 w-3.5" />
              L1 / L2 / L3 Dedup Inspector
            </div>
            <div className="flex items-start gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-slate-900 to-slate-700 text-white">
                <FileText className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-900">{file.file_name}</h1>
                <p className="mt-2 text-sm text-slate-500">
                  {file.file_type} • {file.status} • {formatDateTime(file.updated_at || file.created_at || null)}
                </p>
              </div>
            </div>
            <p className="max-w-3xl text-sm leading-6 text-slate-600">
              This screen reflects the existing backend dedup flow only. L1 shows exact repeated semantic hashes within this file,
              L2 shows duplicate rows already linked to Global Content Index evidence, and L3 shows remaining duplicate rows that need semantic review with the current admin payload.
            </p>
          </div>
          <Button variant="outline" onClick={() => window.history.back()} className="flex items-center gap-2 self-start">
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {[
            ["Total Chunks", String(file.total_chunks), "All stored rows"],
            ["Unique", String(file.unique_chunks), "Passed dedup"],
            ["Duplicates", String(file.duplicate_chunks), "Marked duplicate"],
            ["Dedup Ratio", `${Number(file.dedup_ratio ?? 0).toFixed(2)}%`, "Duplicate rows / total"],
          ].map(([label, value, helper]) => (
            <div key={label} className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{label}</p>
              <p className="mt-3 text-3xl font-bold text-slate-900">{value}</p>
              <p className="mt-2 text-xs text-slate-600">{helper}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <ProcessCard icon={Hash} title="L1 Exact Hash" count={l1DisplayCount} tone="amber" active={l1DisplayCount > 0} description={inferredL1Only ? "Likely repeated exact value inferred from file stats." : `${l1Groups.length} repeated hash groups found in this file.`} />
        <ProcessCard icon={Link2} title="L2 GCI Match" count={l2DisplayCount} tone="blue" active={l2DisplayCount > 0} description="Duplicate rows with a Global Content Index reference." />
        <ProcessCard icon={SearchCheck} title="L3 Semantic Review" count={l3DisplayCount} tone="rose" active={l3DisplayCount > 0} description="Duplicate rows not explained by L1 or exposed GCI links." />
      </div>

      <div className={cn(
        "rounded-3xl border p-5 shadow-sm",
        hasDuplicates ? "border-amber-200 bg-gradient-to-r from-amber-50 via-white to-rose-50" : "border-emerald-200 bg-gradient-to-r from-emerald-50 via-white to-teal-50"
      )}>
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 rounded-full bg-white/80 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-slate-600">
              {hasDuplicates ? <Sparkles className="h-3.5 w-3.5" /> : <Info className="h-3.5 w-3.5" />}
              Review Summary
            </div>
            <h2 className="text-lg font-semibold text-slate-900">
              {hasDuplicates
                ? `${file.duplicate_chunks} duplicate rows detected across ${duplicateCoverage}% of this file`
                : "No duplicate evidence was returned for this file"}
            </h2>
            <p className="max-w-3xl text-sm leading-6 text-slate-600">
              {hasDuplicates
                ? `Start with ${dominantLayer}. L1 is best for exact same-file repeats, L2 is best for prior committed content with GCI evidence, and L3 is the manual review lane when the current admin payload has no exposed target reference.`
                : "This file currently behaves like a clean ingestion baseline. Use it as a reference when comparing how duplicate-heavy files should appear in the same inspector."}
            </p>
          </div>
          <div className="grid min-w-[260px] gap-2 text-sm text-slate-600">
            <LegendItem tone="emerald" label="Unique" detail="No duplicate marker on the stored row" />
            <LegendItem tone="amber" label="L1 Exact" detail="Repeated semantic hash inside this file" />
            <LegendItem tone="blue" label="L2 GCI" detail="Duplicate row linked to Global Content Index" />
            <LegendItem tone="rose" label="L3 Review" detail="Duplicate row with no exposed GCI link in this payload" />
          </div>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className={cn(
          "rounded-3xl border p-5 shadow-sm",
          hasDuplicates ? "border-rose-200 bg-rose-50" : "border-slate-200 bg-slate-50"
        )}>
          <div className="flex items-center gap-3">
            <div className={cn(
              "flex h-11 w-11 items-center justify-center rounded-2xl",
              hasDuplicates ? "bg-rose-100 text-rose-700" : "bg-slate-200 text-slate-600"
            )}>
              <AlertTriangle className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-slate-900">Duplicates Found</h2>
              <p className="text-sm text-slate-600">These rows were flagged as duplicate by the backend.</p>
            </div>
          </div>
          <div className="mt-4 rounded-2xl bg-white/90 p-4">
            <p className={cn("text-4xl font-bold", hasDuplicates ? "text-rose-700" : "text-slate-500")}>
              {file.duplicate_chunks}
            </p>
            <p className="mt-2 text-sm text-slate-600">
              {hasDuplicates
                ? `Review these first. Most of the duplicate evidence is currently in ${dominantLayer}.`
                : "No duplicate rows were returned for this file."}
            </p>
          </div>
        </div>

        <div className="rounded-3xl border border-emerald-200 bg-emerald-50 p-5 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-emerald-100 text-emerald-700">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-slate-900">Unique Content</h2>
              <p className="text-sm text-slate-600">These rows passed dedup and remain usable content.</p>
            </div>
          </div>
          <div className="mt-4 rounded-2xl bg-white/90 p-4">
            <p className="text-4xl font-bold text-emerald-700">{file.unique_chunks}</p>
            <p className="mt-2 text-sm text-slate-600">
              {file.unique_chunks === 1
                ? "Only one row survived dedup for this file."
                : `${file.unique_chunks} rows are unique and safe to keep.`}
            </p>
          </div>
        </div>
      </div>

      <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Dedup Match Review</h2>
            <p className="mt-1 text-sm text-slate-600">
              This review table shows the ingestion file name, where the duplicate was matched, and the available matching semantics for L1, L2, and L3.
            </p>
            <p className="mt-2 text-xs text-slate-500">
              External matched file names are only shown if the current API payload exposes them. L2 and L3 currently use the best available reference evidence from the backend.
            </p>
          </div>
          <div className="text-sm text-slate-500">
            {duplicateTableRows.length} duplicate rows ready for review
          </div>
        </div>
        <div className="mt-5 overflow-x-auto rounded-2xl border border-slate-200">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead className="bg-slate-100">
              <tr>
                {["Lane", "Ingestion File", "Chunk", "Matched Against", "Matching Semantics", "Duplicate Value"].map((header) => (
                  <th key={header} className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-600">{header}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {duplicateTableRows.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-4 py-8 text-center text-sm text-slate-500">
                    No duplicate values found in this file.
                  </td>
                </tr>
              ) : (
                duplicateTableRows.map((row) => (
                  <tr
                    key={`review-${row.id}`}
                    className={cn(
                      row.layer === "L1 Exact" && "bg-amber-50/80",
                      row.layer === "L2 GCI" && "bg-blue-50/80",
                      row.layer === "L3 Review" && "bg-rose-50/80"
                    )}
                  >
                    <td className="px-4 py-3">
                      <span className={cn("rounded-full border px-2.5 py-1 text-[11px] font-semibold", toneClasses(row.layer))}>
                        {row.layer}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-slate-700">{file.file_name}</td>
                    <td className="px-4 py-3 font-medium text-slate-800">#{row.chunk_index}</td>
                    <td className="px-4 py-3 text-slate-700">{matchSurfaceLabel(row.layer, file.file_name)}</td>
                    <td className="px-4 py-3 text-slate-700">
                      <div>{matchingSemanticsText(row.layer, row.reference, row.duplicateCopies, row.isInferred)}</div>
                      <div className="mt-1 font-mono text-xs text-slate-500">{row.reference}</div>
                      {row.isInferred && <div className="mt-1 text-[11px] text-slate-500">Inferred from file totals</div>}
                    </td>
                    <td className="px-4 py-3 text-slate-700">{truncate(row.text, 220)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <EvidencePanel
          title="L1 Exact Duplicate Groups"
          icon={Hash}
          tone="amber"
          empty={inferredL1Only ? "Exact repeated values were inferred from file totals even though duplicate chunk rows were not returned in this payload." : "No exact repeated semantic hashes were found."}
          items={l1Groups.map((group) => ({
            title: `${group.members.length} chunks share ${shortId(group.semanticHash)}`,
            detail: `Anchor #${group.members[0].chunk_index} • ${Math.max(group.members.length - 1, 0)} duplicate copies`,
            body: group.members.map((member) => `#${member.chunk_index}: ${truncate(member.cleaned_text || member.text, 85)}`).join("\n"),
          }))}
        />
        <EvidencePanel
          title="L2 Cross-File GCI Evidence"
          icon={Link2}
          tone="blue"
          empty="No GCI-backed duplicates were returned for this file."
          items={l2Duplicates.map((chunk) => ({
            title: `Chunk #${chunk.chunk_index}`,
            detail: `GCI ${shortId(chunk.global_content_id)} • occurrences ${chunk.gci_occurrence_count ?? "n/a"}`,
            body: truncate(chunk.cleaned_text || chunk.text),
          }))}
        />
        <EvidencePanel
          title="L3 Semantic Review Queue"
          icon={AlertTriangle}
          tone="rose"
          empty="No L3 review items were inferred from the current admin payload."
          items={l3Candidates.map((chunk) => ({
            title: `Chunk #${chunk.chunk_index}`,
            detail: "Duplicate row without exact-hash or visible GCI evidence",
            body: truncate(chunk.cleaned_text || chunk.text),
          }))}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <LaneSummaryCard
          title="L1 Match View"
          tone="amber"
          active={l1DisplayCount > 0}
          fileName={file.file_name}
          matchSurface={matchSurfaceLabel("L1 Exact", file.file_name)}
          semantics={inferredL1Only
            ? matchingSemanticsText("L1 Exact", shortId(uniqueRows[0]?.semantic_hash), file.duplicate_chunks, true)
            : l1Groups[0]
              ? `Primary repeated hash ${shortId(l1Groups[0].semanticHash)} with ${Math.max(l1Groups[0].members.length - 1, 0)} duplicate copies`
              : "No L1 exact duplicate evidence returned for this file."}
        />
        <LaneSummaryCard
          title="L2 Match View"
          tone="blue"
          active={l2DisplayCount > 0}
          fileName={file.file_name}
          matchSurface={matchSurfaceLabel("L2 GCI", file.file_name)}
          semantics={l2Duplicates[0]
            ? matchingSemanticsText("L2 GCI", `${shortId(l2Duplicates[0].global_content_id)} (${l2Duplicates[0].gci_occurrence_count ?? "n/a"})`)
            : "No L2 GCI-backed duplicate evidence returned for this file."}
        />
        <LaneSummaryCard
          title="L3 Match View"
          tone="rose"
          active={l3DisplayCount > 0}
          fileName={file.file_name}
          matchSurface={matchSurfaceLabel("L3 Review", file.file_name)}
          semantics={l3Candidates[0]
            ? matchingSemanticsText("L3 Review", shortId(l3Candidates[0].semantic_hash))
            : "No L3 semantic review candidate was exposed by the current payload."}
        />
      </div>

      <div className="hidden rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Duplicate Values Table</h2>
            <p className="mt-1 text-sm text-slate-600">
              Simple table for non-technical review. Every duplicate row is highlighted and grouped by backend duplicate lane.
            </p>
          </div>
          <div className="text-sm text-slate-500">
            {duplicateRows.length} duplicate rows • {uniqueRows.length} unique rows
          </div>
        </div>
        <div className="mt-5 overflow-x-auto rounded-2xl border border-slate-200">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead className="bg-slate-100">
              <tr>
                {["Status", "Chunk", "Duplicate Type", "Reference", "Duplicate Value"].map((header) => (
                  <th key={header} className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-600">{header}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {duplicateRows.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-4 py-8 text-center text-sm text-slate-500">
                    No duplicate values found in this file.
                  </td>
                </tr>
              ) : (
                duplicateRows.map((chunk) => {
                  const isL1Repeat = l1RepeatIds.has(chunk.id);
                  const duplicateType = isL1Repeat ? "L1 Exact Duplicate" : chunk.global_content_id ? "L2 GCI Duplicate" : "L3 Semantic Review";
                  const reference = chunk.global_content_id
                    ? `${shortId(chunk.global_content_id)} (${chunk.gci_occurrence_count ?? "n/a"})`
                    : shortId(chunk.semantic_hash);
                  return (
                    <tr key={`dup-${chunk.id}`} className="bg-rose-50/70">
                      <td className="px-4 py-3">
                        <span className="rounded-full border border-rose-200 bg-rose-100 px-2.5 py-1 text-[11px] font-semibold text-rose-800">
                          Duplicate
                        </span>
                      </td>
                      <td className="px-4 py-3 font-medium text-slate-800">#{chunk.chunk_index}</td>
                      <td className="px-4 py-3 text-slate-700">{duplicateType}</td>
                      <td className="px-4 py-3 font-mono text-xs text-slate-600">{reference}</td>
                      <td className="px-4 py-3 text-slate-700">{truncate(chunk.cleaned_text || chunk.text, 180)}</td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Unique Values Table</h2>
            <p className="mt-1 text-sm text-slate-600">
              Rows that passed dedup. These are highlighted in green for quick comparison against the duplicate table above.
            </p>
          </div>
          <div className="text-sm text-slate-500">
            Showing first {Math.min(uniqueDisplayRows.length, 10)} unique rows
          </div>
        </div>
        <div className="mt-5 overflow-x-auto rounded-2xl border border-slate-200">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead className="bg-slate-100">
              <tr>
                {["Status", "Chunk", "Semantic Hash", "Unique Value"].map((header) => (
                  <th key={header} className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-600">{header}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {uniqueDisplayRows.length === 0 ? (
                <tr>
                  <td colSpan={4} className="px-4 py-8 text-center text-sm text-slate-500">
                    No unique values were returned for this file.
                  </td>
                </tr>
              ) : (
                uniqueDisplayRows.slice(0, 10).map((row) => (
                  <tr key={`unique-${row.id}`} className="bg-emerald-50/70">
                    <td className="px-4 py-3">
                      <span className="rounded-full border border-emerald-200 bg-emerald-100 px-2.5 py-1 text-[11px] font-semibold text-emerald-800">
                        Unique
                      </span>
                    </td>
                    <td className="px-4 py-3 font-medium text-slate-800">#{row.chunk_index}</td>
                    <td className="px-4 py-3 font-mono text-xs text-slate-600">{row.reference}</td>
                    <td className="px-4 py-3 text-slate-700">{truncate(row.text, 180)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Chunk Governance Console</h2>
            <p className="mt-1 text-sm text-slate-600">
              Operational governance view with page lineage, structural parent links, sentiment signals, and controlled edit actions.
            </p>
          </div>
          <div className="text-sm text-slate-500">
            {canEditChunks ? "Edit enabled for your role" : "Read-only role"}
          </div>
        </div>
        <div className="mt-5 overflow-x-auto rounded-2xl border border-slate-200">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead className="bg-slate-50">
              <tr>
                {["Chunk", "Page", "Resolution", "Parent", "Child", "Section", "Strategy", "Quality", "Sentiment", "Duplicate", "Governance", "Actions"].map((header) => (
                  <th key={header} className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{header}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {sortedChunks.map((chunk) => {
                const meta = chunk.meta_data || {};
                const reasoning = chunk.reasoning_ingestion || {};
                const sectionTitle = reasoning.section_context || reasoning.section_title || meta.section_title || "";
                const sentimentBucket = reasoning.sentiment_bucket || "neutral";
                const sentimentConfidence = Number(reasoning.sentiment_confidence ?? 0.5);
                const qualityScore = Number(reasoning.chunk_quality_score ?? 0);
                const strategy = reasoning.chunking_strategy || "semantic";

                // Infer resolution from available data
                const rawResolution = reasoning.chunk_resolution
                  || (chunk.parent_chunk_id ? "child" : null)
                  || (reasoning.child_chunk_ids ? "parent" : null);
                const resolution = rawResolution || "flat";

                // Infer quality gate from score when not explicitly set
                const qualityGate = reasoning.quality_gate_action
                  || (qualityScore > 0 ? (qualityScore >= 0.35 ? "passed" : "flagged") : "—");

                const hasGist = Boolean(reasoning.document_gist);
                const overlapMode = reasoning.overlap_mode || null;
                const boundaryCoherence = reasoning.boundary_coherence != null ? Number(reasoning.boundary_coherence) : null;

                // Infer content type for display
                const contentType = reasoning.content_type || reasoning.granularity || "";

                // Child chunk IDs for parent chunks
                const childIds: string[] = reasoning.child_chunk_ids || [];
                return (
                  <tr key={`gov-${chunk.id}`} className="align-top hover:bg-slate-50">
                    <td className="px-4 py-3 font-medium text-slate-800">#{chunk.chunk_index}</td>
                    <td className="px-4 py-3 text-slate-700">{chunk.page_number ?? "—"}</td>
                    <td className="px-4 py-3">
                      <span className={cn(
                        "rounded-full border px-2 py-0.5 text-[11px] font-semibold",
                        resolution === "child" && "border-sky-200 bg-sky-100 text-sky-800",
                        resolution === "parent" && "border-violet-200 bg-violet-100 text-violet-800",
                        resolution === "standalone" && "border-slate-200 bg-slate-100 text-slate-700",
                        resolution === "flat" && "border-slate-200 bg-slate-100 text-slate-500"
                      )}>
                        {resolution}
                      </span>
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-slate-600">
                      {shortId(chunk.parent_chunk_id ?? reasoning.parent_chunk_id)}
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-slate-600">
                      {childIds.length > 0
                        ? <div className="flex flex-col gap-0.5">{childIds.slice(0, 3).map((cid: string) => <span key={cid}>{shortId(cid)}</span>)}{childIds.length > 3 && <span className="text-slate-400">+{childIds.length - 3} more</span>}</div>
                        : <span className="text-slate-400">—</span>}
                    </td>
                    <td className="px-4 py-3 text-slate-700">
                      {sectionTitle
                        ? truncate(String(sectionTitle), 42)
                        : contentType
                          ? <span className="text-xs text-slate-400">{contentType}</span>
                          : <span className="text-xs text-slate-400">—</span>}
                    </td>
                    <td className="px-4 py-3">
                      <span className={cn(
                        "rounded-full border px-2 py-0.5 text-[11px] font-semibold",
                        strategy === "elite_v2" && "border-violet-200 bg-violet-100 text-violet-800",
                        strategy === "elite" && "border-indigo-200 bg-indigo-100 text-indigo-800",
                        (strategy.startsWith("structure") || strategy.startsWith("smart") || strategy.startsWith("rust") || strategy.startsWith("overlap") || strategy.startsWith("recursive")) && "border-sky-200 bg-sky-100 text-sky-800",
                        strategy === "semantic" && "border-slate-200 bg-slate-100 text-slate-600"
                      )}>
                        {strategy}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex flex-col gap-1">
                        <div className="flex items-center gap-1.5">
                          <div className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-200">
                            <div
                              className={cn(
                                "h-full rounded-full",
                                qualityScore >= 0.7 && "bg-emerald-500",
                                qualityScore >= 0.4 && qualityScore < 0.7 && "bg-amber-500",
                                qualityScore < 0.4 && "bg-rose-500"
                              )}
                              style={{ width: `${Math.round(qualityScore * 100)}%` }}
                            />
                          </div>
                          <span className="font-mono text-[11px] text-slate-500">
                            {qualityScore > 0 ? `${(qualityScore * 100).toFixed(0)}%` : "—"}
                          </span>
                        </div>
                        <span className={cn(
                          "w-fit rounded-full border px-1.5 py-0.5 text-[10px] font-semibold",
                          qualityGate === "passed" && "border-emerald-200 bg-emerald-50 text-emerald-700",
                          qualityGate === "flagged" && "border-rose-200 bg-rose-50 text-rose-700",
                          qualityGate === "—" && "border-slate-200 bg-slate-50 text-slate-400"
                        )}>
                          {qualityGate}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <span className={cn(
                          "rounded-full border px-2 py-0.5 text-[11px] font-semibold",
                          sentimentBucket === "positive" && "border-emerald-200 bg-emerald-100 text-emerald-800",
                          sentimentBucket === "negative" && "border-rose-200 bg-rose-100 text-rose-800",
                          sentimentBucket !== "positive" && sentimentBucket !== "negative" && "border-slate-200 bg-slate-100 text-slate-700"
                        )}>
                          {String(sentimentBucket)}
                        </span>
                        <span className="font-mono text-xs text-slate-500">{(sentimentConfidence * 100).toFixed(0)}%</span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      {chunk.is_duplicate ? (
                        <span className="rounded-full border border-rose-200 bg-rose-100 px-2 py-0.5 text-[11px] font-semibold text-rose-700">Duplicate</span>
                      ) : (
                        <span className="rounded-full border border-emerald-200 bg-emerald-100 px-2 py-0.5 text-[11px] font-semibold text-emerald-700">Unique</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-xs text-slate-600">
                      <div>type: {chunk.source_type || "—"}</div>
                      <div>tokens: {chunk.tokens ?? "—"}</div>
                      {hasGist && (
                        <div className="mt-0.5 flex items-center gap-1 text-violet-600">
                          <Sparkles className="h-3 w-3" />
                          <span>gist</span>
                        </div>
                      )}
                      {overlapMode && (
                        <div className={cn(
                          "mt-0.5",
                          overlapMode === "dynamic" && "text-amber-600",
                          overlapMode === "skipped_high_coherence" && "text-emerald-600"
                        )}>
                          overlap: {overlapMode === "skipped_high_coherence" ? "skipped" : overlapMode}
                          {boundaryCoherence != null && ` (${(boundaryCoherence * 100).toFixed(0)}%)`}
                        </div>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      {canEditChunks ? (
                        <button
                          onClick={() => openEditor(chunk)}
                          className="inline-flex items-center gap-1 rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-100"
                        >
                          <Pencil className="h-3.5 w-3.5" />
                          Edit
                        </button>
                      ) : (
                        <span className="text-xs text-slate-400">Read-only</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-900">Chunk Reference View</h2>
        <p className="mt-1 text-sm text-slate-600">
          Real chunks returned by the current admin API. This is the raw review surface for checking duplicate text, semantic hashes, and GCI references.
        </p>
        <div className="mt-4 grid gap-3 lg:grid-cols-3">
          <ReviewHint
            tone="amber"
            title="How to read L1"
            body="Look for repeated hashes with an anchor chunk and repeated copies. These are exact normalized-text repeats inside this file."
          />
          <ReviewHint
            tone="blue"
            title="How to read L2"
            body="Use the GCI reference and occurrence count to confirm the row already exists in prior committed ingestions."
          />
          <ReviewHint
            tone="rose"
            title="How to read L3"
            body="Treat this lane as analyst review. The current UI cannot expose the semantic-match target unless the backend adds that field."
          />
        </div>
        <div className="mt-5 overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead className="bg-slate-50">
              <tr>
                {["Chunk", "Layer View", "Semantic Hash", "GCI Reference", "Text"].map((header) => (
                  <th key={header} className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{header}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {sortedChunks.map((chunk) => {
                const isL1Repeat = l1RepeatIds.has(chunk.id);
                const layer = isL1Repeat ? "L1 Exact" : chunk.is_duplicate && chunk.global_content_id ? "L2 GCI" : chunk.is_duplicate ? "L3 Review" : "Unique";
                return (
                  <tr
                    key={chunk.id}
                    className={cn(
                      "align-top hover:bg-slate-50",
                      layer === "L1 Exact" && "bg-amber-50/40",
                      layer === "L2 GCI" && "bg-blue-50/40",
                      layer === "L3 Review" && "bg-rose-50/40"
                    )}
                  >
                    <td className="px-4 py-3 font-medium text-slate-800">#{chunk.chunk_index}</td>
                    <td className="px-4 py-3">
                      <span className={cn(
                        "rounded-full border px-2.5 py-1 text-[11px] font-semibold",
                        layer === "L1 Exact" && "border-amber-200 bg-amber-100 text-amber-800",
                        layer === "L2 GCI" && "border-blue-200 bg-blue-100 text-blue-800",
                        layer === "L3 Review" && "border-rose-200 bg-rose-100 text-rose-800",
                        layer === "Unique" && "border-emerald-200 bg-emerald-100 text-emerald-800"
                      )}>
                        {layer}
                      </span>
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-slate-600">{shortId(chunk.semantic_hash)}</td>
                    <td className="px-4 py-3 text-slate-600">
                      {chunk.global_content_id ? `${shortId(chunk.global_content_id)} (${chunk.gci_occurrence_count ?? "n/a"})` : "—"}
                    </td>
                    <td className="px-4 py-3 text-slate-700">{truncate(chunk.cleaned_text || chunk.text, 160)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <Modal
        open={Boolean(editingChunk)}
        onClose={closeEditor}
        title={editingChunk ? `Edit Chunk #${editingChunk.chunk_index}` : "Edit Chunk"}
      >
        <div className="space-y-4">
          <div className="text-xs text-slate-500">
            Chunk ID: <span className="font-mono">{editingChunk?.id}</span>
          </div>
          <textarea
            rows={10}
            value={draftText}
            onChange={(e) => setDraftText(e.target.value)}
            className="w-full rounded-lg border border-slate-300 p-3 text-sm text-slate-800 outline-none focus:border-primary-500"
          />
          <div>
            <label className="mb-1 block text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Normalization Mode</label>
            <select
              value={llmMode}
              onChange={(e) => setLlmMode(e.target.value as "" | "factual" | "creative")}
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-700 outline-none focus:border-primary-500"
            >
              <option value="">None</option>
              <option value="factual">Factual</option>
              <option value="creative">Creative</option>
            </select>
          </div>
          {saveMessage && (
            <div className={cn(
              "rounded-md px-3 py-2 text-sm",
              saveMessage.toLowerCase().includes("failed") || saveMessage.toLowerCase().includes("empty")
                ? "bg-rose-50 text-rose-700"
                : "bg-emerald-50 text-emerald-700"
            )}>
              {saveMessage}
            </div>
          )}
          <div className="flex items-center justify-end gap-2">
            <Button variant="outline" onClick={closeEditor} disabled={savingChunk}>Cancel</Button>
            <Button onClick={saveChunkEdit} loading={savingChunk} className="flex items-center gap-2">
              <Save className="h-4 w-4" />
              Save Chunk
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

function ProcessCard({ icon: Icon, title, count, description, tone, active }: { icon: ComponentType<{ className?: string }>; title: string; count: number; description: string; tone: "amber" | "blue" | "rose"; active: boolean; }) {
  return (
    <div className={cn(
      "rounded-3xl border p-5 shadow-sm transition-all",
      tone === "amber" && (active ? "border-amber-300 bg-amber-50 ring-2 ring-amber-200" : "border-amber-100 bg-amber-50/40 opacity-60"),
      tone === "blue" && (active ? "border-blue-300 bg-blue-50 ring-2 ring-blue-200" : "border-blue-100 bg-blue-50/40 opacity-60"),
      tone === "rose" && (active ? "border-rose-300 bg-rose-50 ring-2 ring-rose-200" : "border-rose-100 bg-rose-50/40 opacity-60")
    )}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-slate-900">{title}</p>
          <p className="mt-1 text-xs text-slate-600">{description}</p>
        </div>
        <div className="rounded-2xl bg-white/80 p-3"><Icon className="h-5 w-5 text-slate-900" /></div>
      </div>
      <p className="mt-5 text-3xl font-bold text-slate-900">{count}</p>
    </div>
  );
}

function LegendItem({ tone, label, detail }: { tone: "emerald" | "amber" | "blue" | "rose"; label: string; detail: string; }) {
  return (
    <div className="flex items-start gap-3 rounded-2xl border border-white/70 bg-white/80 px-3 py-2.5">
      <span
        className={cn(
          "mt-0.5 h-2.5 w-2.5 rounded-full",
          tone === "emerald" && "bg-emerald-500",
          tone === "amber" && "bg-amber-500",
          tone === "blue" && "bg-blue-500",
          tone === "rose" && "bg-rose-500"
        )}
      />
      <div>
        <p className="text-sm font-semibold text-slate-900">{label}</p>
        <p className="text-xs leading-5 text-slate-500">{detail}</p>
      </div>
    </div>
  );
}

function ReviewHint({ tone, title, body }: { tone: "amber" | "blue" | "rose"; title: string; body: string; }) {
  return (
    <div className={cn(
      "rounded-2xl border p-4",
      tone === "amber" && "border-amber-200 bg-amber-50",
      tone === "blue" && "border-blue-200 bg-blue-50",
      tone === "rose" && "border-rose-200 bg-rose-50"
    )}>
      <p className="text-sm font-semibold text-slate-900">{title}</p>
      <p className="mt-2 text-xs leading-5 text-slate-600">{body}</p>
    </div>
  );
}

function LaneSummaryCard({
  title,
  tone,
  active,
  fileName,
  matchSurface,
  semantics,
}: {
  title: string;
  tone: "amber" | "blue" | "rose";
  active: boolean;
  fileName: string;
  matchSurface: string;
  semantics: string;
}) {
  return (
    <div
      className={cn(
        "rounded-3xl border p-5 shadow-sm transition-all",
        tone === "amber" && (active ? "border-amber-300 bg-amber-50" : "border-amber-100 bg-amber-50/40 opacity-70"),
        tone === "blue" && (active ? "border-blue-300 bg-blue-50" : "border-blue-100 bg-blue-50/40 opacity-70"),
        tone === "rose" && (active ? "border-rose-300 bg-rose-50" : "border-rose-100 bg-rose-50/40 opacity-70"),
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
        <span className={cn(
          "rounded-full border px-2.5 py-1 text-[11px] font-semibold",
          tone === "amber" && "border-amber-200 bg-amber-100 text-amber-800",
          tone === "blue" && "border-blue-200 bg-blue-100 text-blue-800",
          tone === "rose" && "border-rose-200 bg-rose-100 text-rose-800",
        )}>
          {active ? "Active" : "No match"}
        </span>
      </div>
      <div className="mt-4 space-y-3 text-sm text-slate-700">
        <div className="rounded-2xl border border-white/80 bg-white/80 p-3">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Ingestion File</div>
          <div className="mt-1 font-medium text-slate-900">{fileName}</div>
        </div>
        <div className="rounded-2xl border border-white/80 bg-white/80 p-3">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Matched Against</div>
          <div className="mt-1 text-slate-900">{matchSurface}</div>
        </div>
        <div className="rounded-2xl border border-white/80 bg-white/80 p-3">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">Matching Semantics</div>
          <div className="mt-1 leading-6 text-slate-900">{semantics}</div>
        </div>
      </div>
    </div>
  );
}

function EvidencePanel({ title, icon: Icon, tone, empty, items }: { title: string; icon: ComponentType<{ className?: string }>; tone: "amber" | "blue" | "rose"; empty: string; items: Array<{ title: string; detail: string; body: string }>; }) {
  return (
    <div className={cn("rounded-3xl border p-5 shadow-sm", tone === "amber" && "border-amber-200 bg-amber-50", tone === "blue" && "border-blue-200 bg-blue-50", tone === "rose" && "border-rose-200 bg-rose-50")}>
      <div className="flex items-center gap-2"><Icon className="h-4 w-4 text-slate-900" /><h3 className="text-sm font-semibold text-slate-900">{title}</h3></div>
      <div className="mt-4 space-y-3">
        {items.length === 0 ? <div className="rounded-2xl border border-white/80 bg-white/80 p-4 text-sm text-slate-500">{empty}</div> : items.slice(0, 6).map((item, index) => (
          <div key={`${item.title}-${index}`} className="rounded-2xl border border-white/80 bg-white/80 p-4">
            <p className="text-sm font-semibold text-slate-900">{item.title}</p>
            <p className="mt-1 text-xs text-slate-500">{item.detail}</p>
            <p className="mt-3 whitespace-pre-line text-sm leading-6 text-slate-700">{item.body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
