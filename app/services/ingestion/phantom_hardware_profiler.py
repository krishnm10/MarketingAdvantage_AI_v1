# =============================================================================
# phantom_hardware_profiler.py
# PHANTOM Protocol — Phase 0: Hardware Profiler
#
# Detects the host hardware at startup (AMD ROCm, CUDA, or CPU) and computes
# all PHANTOM tuning parameters: batch sizes, worker counts, Bloom filter
# capacity, thread pool size, and stage-collapse thresholds.
#
# Zero dependencies beyond the standard library + optional torch/rocm probing.
# If GPU libraries aren't installed it degrades gracefully to CPU-safe values.
#
# Usage:
#   from app.services.ingestion.phantom_hardware_profiler import phantom_profile
#   profile = phantom_profile()                 # call once at startup
#   print(profile.embed_batch_size)             # 256 on GPU, 64 on CPU
#
# Wire into FastAPI via phantom_config_bridge.phantom_startup()
# =============================================================================

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("phantom.hardware_profiler")


# ─────────────────────────────────────────────────────────────────────────────
# Hardware tier enum
# ─────────────────────────────────────────────────────────────────────────────

class HardwareTier(str, Enum):
    AMD_ROCM   = "amd_rocm"    # AMD GPU with ROCm (primary target hardware)
    CUDA       = "cuda"        # NVIDIA CUDA GPU
    APPLE_MPS  = "apple_mps"   # Apple Silicon Metal Performance Shaders
    CPU_HIGH   = "cpu_high"    # ≥16 GB RAM, ≥8 cores — high-end CPU only
    CPU_MID    = "cpu_mid"     # 8–16 GB RAM
    CPU_LOW    = "cpu_low"     # <8 GB RAM — conservative settings


# ─────────────────────────────────────────────────────────────────────────────
# Profile dataclass — all PHANTOM parameters in one place
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PHANTOMHardwareProfile:
    """
    All PHANTOM tuning parameters, derived from detected hardware at startup.

    Every number here is consumed by phantom_config_bridge.py and injected
    into PHANTOM components.  Nothing is hardcoded in the hot path.
    """

    # ── detection results ────────────────────────────────────────────────────
    tier:               HardwareTier = HardwareTier.CPU_MID
    gpu_name:           str          = "none"
    gpu_vram_gb:        float        = 0.0
    system_ram_gb:      float        = 0.0
    cpu_cores:          int          = 4
    cpu_threads:        int          = 8

    # ── embedding batching ───────────────────────────────────────────────────
    embed_batch_size:   int   = 64    # chunks sent to embedder per call
    embed_prefetch:     int   = 2     # batches to prefetch ahead of upsert

    # ── vector DB upsert ────────────────────────────────────────────────────
    upsert_batch_size:  int   = 128   # vectors upserted per VectorDB call
    upsert_concurrency: int   = 2     # parallel upsert workers

    # ── ingestion pipeline ──────────────────────────────────────────────────
    ingest_workers:     int   = 4     # parallel file ingestion tasks
    parse_workers:      int   = 4     # parallel parser calls
    io_thread_pool:     int   = 8     # ThreadPoolExecutor size for sync I/O

    # ── deduplication ────────────────────────────────────────────────────────
    bloom_capacity:     int   = 2_000_000  # Bloom filter expected element count
    bloom_error_rate:   float = 0.001      # 0.1% false-positive rate
    l2_dedup_batch:     int   = 32         # chunks batched for L2 embedding dedup

    # ── gravity clustering (Phase 3) ────────────────────────────────────────
    gravity_clusters:   int   = 16    # number of topic clusters per batch
    gravity_batch_size: int   = 512   # records per gravity round

    # ── stage collapse (Phase 5) ─────────────────────────────────────────────
    stage_collapse_concurrency: int = 4   # parallel stage-collapse kernels

    # ── diagnostics ─────────────────────────────────────────────────────────
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        gpu_line = (
            f"GPU: {self.gpu_name} ({self.gpu_vram_gb:.1f} GB VRAM)"
            if self.gpu_vram_gb > 0 else "GPU: none"
        )
        return (
            f"[PHANTOM Hardware Profile]\n"
            f"  Tier         : {self.tier.value}\n"
            f"  {gpu_line}\n"
            f"  RAM          : {self.system_ram_gb:.1f} GB\n"
            f"  CPU          : {self.cpu_cores}c / {self.cpu_threads}t\n"
            f"  embed_batch  : {self.embed_batch_size}\n"
            f"  upsert_batch : {self.upsert_batch_size}\n"
            f"  ingest_workers: {self.ingest_workers}\n"
            f"  io_thread_pool: {self.io_thread_pool}\n"
            f"  bloom_capacity: {self.bloom_capacity:,}\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _system_ram_gb() -> float:
    """Read total RAM in GB. Works on Linux / macOS / Windows."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    # Linux fallback — parse /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except Exception:
        pass
    return 8.0  # safe default


def _cpu_counts() -> tuple[int, int]:
    """Returns (physical_cores, logical_threads)."""
    logical = multiprocessing.cpu_count() or 4
    # Try psutil for physical core count
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or logical
        return physical, logical
    except ImportError:
        return max(1, logical // 2), logical


def _probe_rocm() -> tuple[bool, str, float]:
    """
    Returns (found, gpu_name, vram_gb).

    Tries three paths in order of reliability:
      1. torch.cuda with ROCm build (AMD exposes itself via CUDA API on ROCm)
      2. rocm-smi CLI
      3. /sys/class/drm/card*/device/mem_info_vram_total (Linux sysfs)
    """
    # Path 1 — torch ROCm (most reliable when torch is installed)
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            name  = props.name
            vram  = props.total_memory / (1024 ** 3)
            is_rocm = (
                "AMD"    in name.upper() or
                "RADEON" in name.upper() or
                "GFX"    in name.upper() or
                hasattr(torch.version, "hip")          # ROCm build flag
            )
            if is_rocm:
                return True, name, vram
    except Exception:
        pass

    # Path 2 — rocm-smi CLI
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and "VRAM" in result.stdout:
            lines = [l for l in result.stdout.splitlines() if l.strip() and "GPU" not in l]
            for line in lines:
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        vram_bytes = int(parts[-1].strip())
                        return True, "AMD GPU (rocm-smi)", vram_bytes / (1024 ** 3)
                    except ValueError:
                        pass
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # Path 3 — Linux sysfs
    try:
        import glob
        for path in glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"):
            with open(path) as f:
                vram_bytes = int(f.read().strip())
                if vram_bytes > 0:
                    return True, "AMD GPU (sysfs)", vram_bytes / (1024 ** 3)
    except Exception:
        pass

    return False, "", 0.0


def _probe_cuda() -> tuple[bool, str, float]:
    """Returns (found, gpu_name, vram_gb) for NVIDIA CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            name  = props.name
            vram  = props.total_memory / (1024 ** 3)
            if "NVIDIA" in name.upper() or "TESLA" in name.upper():
                return True, name, vram
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            line = result.stdout.strip().splitlines()[0]
            parts = line.split(",")
            if len(parts) >= 2:
                name = parts[0].strip()
                vram = float(parts[1].strip()) / 1024  # MiB → GiB
                return True, name, vram
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    return False, "", 0.0


def _probe_mps() -> bool:
    """Apple Silicon Metal Performance Shaders."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Parameter computation — maps hardware to PHANTOM tuning values
# ─────────────────────────────────────────────────────────────────────────────

def _compute_params(
    tier: HardwareTier,
    vram_gb: float,
    ram_gb: float,
    cpu_cores: int,
    cpu_threads: int,
) -> dict:
    """
    Derives all PHANTOM batch sizes and concurrency values from detected hardware.

    AMD ROCm notes (primary target):
      - ROCm memory bandwidth favours larger batches than CUDA for the same VRAM.
      - We add a 10% bump on batch sizes vs equivalent CUDA VRAM tier.
      - Upsert concurrency is kept lower because ROCm HIP context switches are
        more expensive than CUDA — pipelining embed→upsert is more effective.
    """
    warnings = []

    if tier in (HardwareTier.AMD_ROCM, HardwareTier.CUDA, HardwareTier.APPLE_MPS):
        # ── GPU path ────────────────────────────────────────────────────────
        # embed_batch_size: fill ~30% of VRAM per batch (safe headroom for model weights)
        # Rule of thumb: 1 GB VRAM ≈ 512 fp32 embeddings at dim=1536
        base = max(64, int(vram_gb * 512 * 0.30))
        # Round to nearest power of 2 for GPU efficiency
        embed_batch = 2 ** round(math.log2(base)) if base >= 2 else 64

        # AMD ROCm: bandwidth favours slightly larger batches
        if tier == HardwareTier.AMD_ROCM:
            embed_batch = int(embed_batch * 1.1)
            embed_batch = min(embed_batch, 1024)  # safety cap

        upsert_batch      = min(embed_batch * 2, 512)
        upsert_concurrency = 2 if tier == HardwareTier.AMD_ROCM else 3
        ingest_workers    = min(cpu_cores, 8)
        parse_workers     = min(cpu_cores, 8)
        io_thread_pool    = cpu_threads * 2
        l2_dedup_batch    = min(embed_batch, 64)
        gravity_batch     = min(embed_batch * 2, 1024)
        gravity_clusters  = 32
        stage_collapse_concurrency = min(cpu_cores // 2, 8)

        if vram_gb < 4:
            warnings.append(
                f"GPU VRAM is only {vram_gb:.1f} GB — embed_batch_size capped at 64. "
                "Consider reducing model size or using CPU offload."
            )
            embed_batch = 64

    elif tier == HardwareTier.CPU_HIGH:
        # ── High-end CPU (≥16 GB RAM, ≥8 cores) ────────────────────────────
        # Batch size limited by RAM; target ~2 GB per embed batch
        embed_batch        = 128
        upsert_batch       = 256
        upsert_concurrency = 2
        ingest_workers     = min(cpu_cores // 2, 6)
        parse_workers      = min(cpu_cores // 2, 6)
        io_thread_pool     = cpu_threads
        l2_dedup_batch     = 32
        gravity_batch      = 256
        gravity_clusters   = 16
        stage_collapse_concurrency = min(cpu_cores // 2, 4)

    elif tier == HardwareTier.CPU_MID:
        # ── Mid CPU (8–16 GB RAM) ─────────────────────────────────────────
        embed_batch        = 64
        upsert_batch       = 128
        upsert_concurrency = 1
        ingest_workers     = min(cpu_cores // 2, 4)
        parse_workers      = min(cpu_cores // 2, 4)
        io_thread_pool     = cpu_threads
        l2_dedup_batch     = 16
        gravity_batch      = 128
        gravity_clusters   = 8
        stage_collapse_concurrency = 2

    else:  # CPU_LOW
        embed_batch        = 32
        upsert_batch       = 64
        upsert_concurrency = 1
        ingest_workers     = 2
        parse_workers      = 2
        io_thread_pool     = max(cpu_threads, 4)
        l2_dedup_batch     = 8
        gravity_batch      = 64
        gravity_clusters   = 4
        stage_collapse_concurrency = 1
        warnings.append(
            "Low RAM detected — PHANTOM running in conservative mode. "
            "Throughput targets may not be achievable. "
            "Consider adding RAM or using a machine with GPU."
        )

    # Bloom filter: size for the expected records count
    # We scale from 1M baseline; larger RAM → larger filter → lower false-positives
    bloom_capacity = max(1_000_000, int(ram_gb * 500_000))

    return dict(
        embed_batch_size          = embed_batch,
        embed_prefetch            = 2,
        upsert_batch_size         = upsert_batch,
        upsert_concurrency        = upsert_concurrency,
        ingest_workers            = ingest_workers,
        parse_workers             = parse_workers,
        io_thread_pool            = io_thread_pool,
        bloom_capacity            = bloom_capacity,
        bloom_error_rate          = 0.001,
        l2_dedup_batch            = l2_dedup_batch,
        gravity_clusters          = gravity_clusters,
        gravity_batch_size        = gravity_batch,
        stage_collapse_concurrency = stage_collapse_concurrency,
        warnings                  = warnings,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def phantom_profile(verbose: bool = True) -> PHANTOMHardwareProfile:
    """
    Detect hardware and return a fully populated PHANTOMHardwareProfile.

    Call once at application startup via phantom_startup() in
    phantom_config_bridge.py.  The result is cached in the bridge module and
    shared across all PHANTOM components.

    Args:
        verbose: If True, logs the profile summary at INFO level.

    Returns:
        PHANTOMHardwareProfile with all PHANTOM tuning parameters populated.
    """
    ram_gb               = _system_ram_gb()
    cpu_cores, cpu_threads = _cpu_counts()

    # ── GPU detection (try ROCm first — that's the target hardware) ─────────
    rocm_found, rocm_name, rocm_vram = _probe_rocm()
    cuda_found, cuda_name, cuda_vram = (False, "", 0.0)
    mps_found = False

    if rocm_found:
        tier     = HardwareTier.AMD_ROCM
        gpu_name = rocm_name
        vram_gb  = rocm_vram
    else:
        cuda_found, cuda_name, cuda_vram = _probe_cuda()
        if cuda_found:
            tier     = HardwareTier.CUDA
            gpu_name = cuda_name
            vram_gb  = cuda_vram
        elif _probe_mps():
            tier     = HardwareTier.APPLE_MPS
            gpu_name = "Apple Silicon (MPS)"
            # MPS shares system RAM; use 25% of RAM as effective VRAM estimate
            vram_gb  = ram_gb * 0.25
        else:
            # CPU only — pick tier by RAM
            gpu_name = "none"
            vram_gb  = 0.0
            if ram_gb >= 16 and cpu_cores >= 8:
                tier = HardwareTier.CPU_HIGH
            elif ram_gb >= 8:
                tier = HardwareTier.CPU_MID
            else:
                tier = HardwareTier.CPU_LOW

    params = _compute_params(tier, vram_gb, ram_gb, cpu_cores, cpu_threads)

    profile = PHANTOMHardwareProfile(
        tier           = tier,
        gpu_name       = gpu_name,
        gpu_vram_gb    = vram_gb,
        system_ram_gb  = ram_gb,
        cpu_cores      = cpu_cores,
        cpu_threads    = cpu_threads,
        **params,
    )

    if verbose:
        for line in profile.summary().splitlines():
            logger.info(line)
        for w in profile.warnings:
            logger.warning("[PHANTOM] %s", w)

    return profile
