"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Eye,
  Cpu,
  Monitor,
  Cloud,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Loader2,
  Zap,
  Image,
  Video,
  FileText,
  BarChart3,
  Layers,
  Settings,
} from "lucide-react";
import { cn } from "@/lib/utils";
import apiClient from "@/lib/apiClient";

/* ─── Types ─── */
interface EncoderInfo {
  name: string;
  tier: "cpu" | "gpu" | "api";
  model: string;
  status: "available" | "unavailable" | "active" | "checking";
  description: string;
  icon: any;
  gradient: string;
  capabilities: string[];
  details: Record<string, string>;
}

interface VisionConfig {
  ai_profile: string;
  vision_model_cpu: string;
  vision_model_gpu: string;
  vision_api_provider: string;
  vision_api_model: string;
  vision_quantize: string;
  vision_flash_attention: string;
  vision_max_pixels: string;
  vision_batch_size_cpu: string;
  vision_batch_size_gpu: string;
}

function StatusBadge({ status }: { status: string }) {
  if (status === "checking") {
    return <Loader2 className="h-4 w-4 text-slate-400 animate-spin" />;
  }
  if (status === "active") {
    return (
      <span className="flex items-center gap-1.5 text-xs font-semibold text-emerald-600 bg-emerald-50 rounded-full px-2.5 py-0.5">
        <CheckCircle2 className="h-3 w-3" /> Active
      </span>
    );
  }
  if (status === "available") {
    return (
      <span className="flex items-center gap-1.5 text-xs font-medium text-blue-600 bg-blue-50 rounded-full px-2.5 py-0.5">
        <CheckCircle2 className="h-3 w-3" /> Available
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1.5 text-xs font-medium text-slate-400 bg-slate-100 rounded-full px-2.5 py-0.5">
      <XCircle className="h-3 w-3" /> Not Installed
    </span>
  );
}

const MODE_ICONS: Record<string, any> = {
  Caption: Image,
  OCR: FileText,
  Chart: BarChart3,
  Document: FileText,
  VQA: Eye,
  Embed: Layers,
  Video: Video,
};

export default function VisionEncoderPage() {
  const [config, setConfig] = useState<VisionConfig | null>(null);
  const [encoders, setEncoders] = useState<EncoderInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [checking, setChecking] = useState(false);

  const loadConfig = useCallback(async () => {
    try {
      const res = await apiClient.get("/api/v2/config/");
      const cfg = res.data?.config || {};
      const parsed: VisionConfig = {
        ai_profile:           cfg.AI_PROFILE || "cpu",
        vision_model_cpu:     cfg.VISION_MODEL_CPU || "Qwen/Qwen2.5-VL-3B-Instruct",
        vision_model_gpu:     cfg.VISION_MODEL_GPU || "Qwen/Qwen2.5-VL-7B-Instruct",
        vision_api_provider:  cfg.VISION_API_PROVIDER || "openai",
        vision_api_model:     cfg.VISION_API_MODEL || "gpt-4o",
        vision_quantize:      cfg.VISION_QUANTIZE || "4bit",
        vision_flash_attention: cfg.VISION_FLASH_ATTENTION || "true",
        vision_max_pixels:    cfg.VISION_MAX_PIXELS || "1003520",
        vision_batch_size_cpu: cfg.VISION_BATCH_SIZE_CPU || "1",
        vision_batch_size_gpu: cfg.VISION_BATCH_SIZE_GPU || "4",
      };
      setConfig(parsed);
      buildEncoderList(parsed);
    } catch {
      // Use defaults if config API not reachable
      const defaults: VisionConfig = {
        ai_profile: "cpu",
        vision_model_cpu: "Qwen/Qwen2.5-VL-3B-Instruct",
        vision_model_gpu: "Qwen/Qwen2.5-VL-7B-Instruct",
        vision_api_provider: "openai",
        vision_api_model: "gpt-4o",
        vision_quantize: "4bit",
        vision_flash_attention: "true",
        vision_max_pixels: "1003520",
        vision_batch_size_cpu: "1",
        vision_batch_size_gpu: "4",
      };
      setConfig(defaults);
      buildEncoderList(defaults);
    } finally {
      setLoading(false);
    }
  }, []);

  const buildEncoderList = (cfg: VisionConfig) => {
    const profile = cfg.ai_profile;

    const list: EncoderInfo[] = [
      /* ─ CPU Tier ─ */
      {
        name: "Qwen2.5-VL-3B",
        tier: "cpu",
        model: "Qwen/Qwen2.5-VL-3B-Instruct",
        status: profile === "cpu" && cfg.vision_model_cpu.includes("Qwen") ? "active" : "available",
        description: "Best open-source quality on CPU. Excellent chart and document understanding.",
        icon: Cpu,
        gradient: "from-primary-500 to-primary-700 shadow-primary-600/20",
        capabilities: ["Caption", "OCR", "Chart", "Document", "VQA", "Video"],
        details: {
          "Parameters": "3B",
          "Load Time": "~30-60s",
          "RAM": "~6 GB",
          "Precision": "float32",
          "Package": "transformers + qwen-vl-utils",
        },
      },
      {
        name: "moondream2",
        tier: "cpu",
        model: "vikhyatk/moondream2",
        status: profile === "cpu" && cfg.vision_model_cpu.includes("moondream") ? "active" : "available",
        description: "Ultra-lightweight 1.8B fallback. Fastest CPU option with minimal RAM.",
        icon: Cpu,
        gradient: "from-slate-500 to-slate-700 shadow-slate-600/20",
        capabilities: ["Caption", "VQA"],
        details: {
          "Parameters": "1.8B",
          "Load Time": "~10s",
          "RAM": "~2 GB",
          "Precision": "float32",
          "Package": "moondream",
        },
      },

      /* ─ GPU Tier ─ */
      {
        name: "Qwen2.5-VL-7B",
        tier: "gpu",
        model: "Qwen/Qwen2.5-VL-7B-Instruct",
        status: profile === "gpu" && cfg.vision_model_gpu.includes("7B") ? "active" : "available",
        description: "Sweet spot for GPU: high quality with BF16 + Flash Attention 2 on 24GB VRAM.",
        icon: Monitor,
        gradient: "from-violet-500 to-violet-700 shadow-violet-600/20",
        capabilities: ["Caption", "OCR", "Chart", "Document", "VQA", "Video", "Embed"],
        details: {
          "Parameters": "7B",
          "VRAM": "~16 GB (BF16) / ~6 GB (4-bit)",
          "Quantization": cfg.vision_quantize,
          "Flash Attention": cfg.vision_flash_attention,
          "Batch Size": cfg.vision_batch_size_gpu,
        },
      },
      {
        name: "Qwen2.5-VL-72B",
        tier: "gpu",
        model: "Qwen/Qwen2.5-VL-72B-Instruct",
        status: profile === "gpu" && cfg.vision_model_gpu.includes("72B") ? "active" : "available",
        description: "Maximum quality. Requires 80GB+ VRAM (or 40GB with 4-bit quantization).",
        icon: Monitor,
        gradient: "from-purple-500 to-purple-700 shadow-purple-600/20",
        capabilities: ["Caption", "OCR", "Chart", "Document", "VQA", "Video", "Embed"],
        details: {
          "Parameters": "72B",
          "VRAM": "~80 GB (BF16) / ~40 GB (4-bit)",
          "Quantization": cfg.vision_quantize,
          "Batch Size": cfg.vision_batch_size_gpu,
        },
      },
      {
        name: "LLaVA-OneVision-7B",
        tier: "gpu",
        model: "lmms-lab/llava-onevision-qwen2-7b-ov-hf",
        status: profile === "gpu" && cfg.vision_model_gpu.includes("llava") ? "active" : "available",
        description: "Strong open model, excellent for video frame understanding.",
        icon: Monitor,
        gradient: "from-indigo-500 to-indigo-700 shadow-indigo-600/20",
        capabilities: ["Caption", "OCR", "Video", "VQA"],
        details: {
          "Parameters": "7B",
          "VRAM": "~16 GB",
          "License": "Apache 2.0",
        },
      },
      {
        name: "InternVL3-8B",
        tier: "gpu",
        model: "OpenGVLab/InternVL3-8B",
        status: profile === "gpu" && cfg.vision_model_gpu.includes("InternVL") ? "active" : "available",
        description: "MTEB benchmark leader. Best for dense document understanding.",
        icon: Monitor,
        gradient: "from-cyan-500 to-cyan-700 shadow-cyan-600/20",
        capabilities: ["Caption", "OCR", "Document", "Chart"],
        details: {
          "Parameters": "8B",
          "VRAM": "~18 GB",
          "Specialty": "Dense Documents",
        },
      },

      /* ─ API Tier ─ */
      {
        name: "GPT-4o Vision",
        tier: "api",
        model: "gpt-4o",
        status: profile === "api" && cfg.vision_api_provider === "openai" ? "active" : "available",
        description: "Best API quality. Excellent at chart reading, OCR, and visual Q&A.",
        icon: Cloud,
        gradient: "from-emerald-500 to-emerald-700 shadow-emerald-600/20",
        capabilities: ["Caption", "OCR", "Chart", "Document", "VQA", "Video"],
        details: {
          "Provider": "OpenAI",
          "Env Key": "OPENAI_API_KEY",
          "Cost": "~$0.01/image",
        },
      },
      {
        name: "Claude 3.5 Sonnet Vision",
        tier: "api",
        model: "claude-3-5-sonnet-20241022",
        status: profile === "api" && cfg.vision_api_provider === "anthropic" ? "active" : "available",
        description: "Best for structured documents, tables, and chart understanding.",
        icon: Cloud,
        gradient: "from-amber-500 to-amber-700 shadow-amber-600/20",
        capabilities: ["Caption", "OCR", "Chart", "Document", "VQA"],
        details: {
          "Provider": "Anthropic",
          "Env Key": "ANTHROPIC_API_KEY",
          "Cost": "~$0.008/image",
        },
      },
      {
        name: "Gemini 2.0 Flash",
        tier: "api",
        model: "gemini-2.0-flash",
        status: profile === "api" && cfg.vision_api_provider === "google" ? "active" : "available",
        description: "Fast and cost-effective. Strong at general vision tasks.",
        icon: Cloud,
        gradient: "from-blue-400 to-blue-600 shadow-blue-500/20",
        capabilities: ["Caption", "OCR", "Chart", "Document", "VQA"],
        details: {
          "Provider": "Google",
          "Env Key": "GOOGLE_API_KEY",
          "Cost": "~$0.002/image",
        },
      },
    ];

    setEncoders(list);
  };

  const checkStatus = useCallback(async () => {
    setChecking(true);
    try {
      const res = await apiClient.get("/api/v2/config/vision-status");
      const data = res.data;
      setEncoders((prev) =>
        prev.map((enc) => {
          if (enc.tier === "cpu") {
            const installed = data?.cpu_packages_installed ?? false;
            if (enc.status === "active") return enc;
            return { ...enc, status: installed ? "available" : "unavailable" };
          }
          return enc;
        })
      );
    } catch {
      // Endpoint may not exist yet — leave as-is
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  useEffect(() => {
    if (!loading) checkStatus();
  }, [loading, checkStatus]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-8 w-8 text-primary-500 animate-spin" />
      </div>
    );
  }

  const profile = config?.ai_profile || "cpu";
  const tiers = [
    { id: "cpu", label: "CPU (Local)", icon: Cpu, color: "text-primary-600", active: profile === "cpu" },
    { id: "gpu", label: "GPU (CUDA)", icon: Monitor, color: "text-violet-600", active: profile === "gpu" },
    { id: "api", label: "Cloud API", icon: Cloud, color: "text-emerald-600", active: profile === "api" },
  ];

  return (
    <div className="space-y-6">
      {/* ─── Header ─── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Multimodal Vision Encoder</h1>
          <p className="mt-1 text-sm text-slate-500">
            Neural vision models for image, chart, document, and video understanding
          </p>
        </div>
        <button
          onClick={() => { loadConfig(); checkStatus(); }}
          disabled={checking}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={cn("h-3.5 w-3.5", checking && "animate-spin")} />
          Refresh
        </button>
      </div>

      {/* ─── Active Profile Banner ─── */}
      <div className="rounded-xl border border-slate-200/60 bg-white shadow-card overflow-hidden">
        <div className="flex items-center gap-4 px-6 py-4 border-b border-slate-100">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-rose-500 to-rose-700 shadow-lg shadow-rose-600/20">
            <Eye className="h-5 w-5 text-white" />
          </div>
          <div className="flex-1">
            <h2 className="text-sm font-semibold text-slate-900">Active Configuration</h2>
            <p className="text-xs text-slate-400">Current runtime profile and selected model</p>
          </div>
          <a
            href="/settings"
            className="inline-flex items-center gap-1.5 rounded-lg bg-primary-50 px-3 py-1.5 text-xs font-semibold text-primary-700 hover:bg-primary-100 transition-colors"
          >
            <Settings className="h-3.5 w-3.5" />
            Edit in Settings
          </a>
        </div>
        <div className="px-6 py-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div>
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Profile</span>
            <p className="mt-0.5 text-sm font-mono font-semibold text-slate-900">{profile.toUpperCase()}</p>
          </div>
          <div>
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Active Model</span>
            <p className="mt-0.5 text-sm font-mono text-slate-700 truncate">
              {profile === "cpu" ? config?.vision_model_cpu :
               profile === "gpu" ? config?.vision_model_gpu :
               `${config?.vision_api_provider}/${config?.vision_api_model}`}
            </p>
          </div>
          <div>
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Quantization</span>
            <p className="mt-0.5 text-sm font-mono text-slate-700">{profile === "gpu" ? config?.vision_quantize : "N/A"}</p>
          </div>
          <div>
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Batch Size</span>
            <p className="mt-0.5 text-sm font-mono text-slate-700">
              {profile === "cpu" ? config?.vision_batch_size_cpu :
               profile === "gpu" ? config?.vision_batch_size_gpu : "async"}
            </p>
          </div>
        </div>
      </div>

      {/* ─── Tier Tabs ─── */}
      <div className="flex items-center gap-2">
        {tiers.map((tier) => (
          <div
            key={tier.id}
            className={cn(
              "flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors",
              tier.active
                ? "bg-slate-900 text-white shadow-sm"
                : "bg-white text-slate-500 border border-slate-200"
            )}
          >
            <tier.icon className={cn("h-4 w-4", tier.active ? "text-white" : tier.color)} />
            {tier.label}
            {tier.active && (
              <span className="ml-1 flex h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
            )}
          </div>
        ))}
      </div>

      {/* ─── Encoder Cards Grid ─── */}
      {(["cpu", "gpu", "api"] as const).map((tier) => {
        const tierEncoders = encoders.filter((e) => e.tier === tier);
        if (tierEncoders.length === 0) return null;

        const tierLabel = tier === "cpu" ? "CPU — Local Models" : tier === "gpu" ? "GPU — CUDA Accelerated" : "API — Cloud Providers";
        const tierDesc = tier === "cpu"
          ? "Runs on any machine. No GPU required. Auto-fallback chain: Qwen → moondream."
          : tier === "gpu"
          ? "CUDA/ROCm accelerated. BF16, Flash Attention 2, 4-bit quantization support."
          : "Cloud-hosted vision APIs. Zero local model loading. Pay-per-image pricing.";

        return (
          <div key={tier} className="space-y-4">
            <div>
              <h3 className="text-lg font-bold text-slate-900">{tierLabel}</h3>
              <p className="text-sm text-slate-500">{tierDesc}</p>
            </div>
            <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
              {tierEncoders.map((enc) => (
                <div
                  key={enc.model}
                  className={cn(
                    "rounded-xl border bg-white shadow-card hover:shadow-card-hover transition-shadow duration-300 overflow-hidden",
                    enc.status === "active"
                      ? "border-primary-200 ring-1 ring-primary-100"
                      : "border-slate-200/60"
                  )}
                >
                  {/* Card Header */}
                  <div className="flex items-center justify-between border-b border-slate-100 px-6 py-4">
                    <div className="flex items-center gap-3">
                      <div className={cn("flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br shadow-lg", enc.gradient)}>
                        <enc.icon className="h-5 w-5 text-white" />
                      </div>
                      <div>
                        <h4 className="text-sm font-semibold text-slate-900">{enc.name}</h4>
                        <p className="text-[11px] font-mono text-slate-400">{enc.model}</p>
                      </div>
                    </div>
                    <StatusBadge status={enc.status} />
                  </div>

                  {/* Description */}
                  <div className="px-6 py-3 border-b border-slate-50">
                    <p className="text-xs text-slate-500 leading-relaxed">{enc.description}</p>
                  </div>

                  {/* Capabilities */}
                  <div className="px-6 py-3 border-b border-slate-50">
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Supported Modes</span>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {enc.capabilities.map((cap) => {
                        const CapIcon = MODE_ICONS[cap] || Eye;
                        return (
                          <span
                            key={cap}
                            className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-medium text-slate-600"
                          >
                            <CapIcon className="h-2.5 w-2.5" />
                            {cap}
                          </span>
                        );
                      })}
                    </div>
                  </div>

                  {/* Details */}
                  <div className="px-6 py-3 space-y-2">
                    {Object.entries(enc.details).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between text-sm">
                        <span className="text-slate-500 text-xs">{key}</span>
                        <span className="rounded-lg bg-slate-100 px-2.5 py-0.5 text-xs font-mono text-slate-600">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })}

      {/* ─── Installation Guide ─── */}
      <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-card">
        <h3 className="text-sm font-semibold text-slate-900 mb-3">Installation Guide</h3>
        <p className="text-sm text-slate-500 mb-4">
          Install the packages for your target profile. The system auto-detects available packages at startup.
        </p>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          {[
            {
              title: "CPU Profile",
              gradient: "from-primary-500 to-primary-700",
              commands: [
                "pip install transformers qwen-vl-utils torch accelerate",
                "pip install moondream  # fallback",
              ],
            },
            {
              title: "GPU Profile",
              gradient: "from-violet-500 to-violet-700",
              commands: [
                "pip install transformers qwen-vl-utils torch accelerate",
                "pip install bitsandbytes  # quantization",
                "pip install flash-attn --no-build-isolation",
              ],
            },
            {
              title: "API Profile",
              gradient: "from-emerald-500 to-emerald-700",
              commands: [
                "pip install httpx  # likely already installed",
                "# Set env: OPENAI_API_KEY or",
                "# ANTHROPIC_API_KEY or GOOGLE_API_KEY",
              ],
            },
          ].map((profile) => (
            <div key={profile.title} className="rounded-lg border border-slate-200 overflow-hidden">
              <div className={cn("px-4 py-2 bg-gradient-to-r text-white text-xs font-semibold", profile.gradient)}>
                {profile.title}
              </div>
              <div className="p-4 space-y-1.5 bg-slate-50">
                {profile.commands.map((cmd, i) => (
                  <code key={i} className="block text-[11px] font-mono text-slate-600 leading-relaxed">
                    {cmd}
                  </code>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
