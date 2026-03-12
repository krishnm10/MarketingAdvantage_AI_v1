# =============================================================================
# phantom_config_bridge.py
# PHANTOM Protocol — Phase 0: Config Bridge
#
# Single source of truth for all PHANTOM runtime parameters.
# Wraps phantom_hardware_profiler at startup, then exposes every tuning
# value through a clean API so all PHANTOM phases can import from one place.
#
# ── Drop-in wiring ───────────────────────────────────────────────────────────
# In your FastAPI main.py (or app/__init__.py), add ONE call to startup:
#
#   from contextlib import asynccontextmanager
#   from app.services.ingestion.phantom_config_bridge import phantom_startup
#
#   @asynccontextmanager
#   async def lifespan(app: FastAPI):
#       phantom_startup()          # ← add this line
#       yield
#
#   app = FastAPI(lifespan=lifespan)
#
# Every other PHANTOM component then imports the singleton:
#
#   from app.services.ingestion.phantom_config_bridge import phantom_cfg
#   batch = phantom_cfg.embed_batch_size
#
# ── Zero risk ────────────────────────────────────────────────────────────────
# This module does NOT modify any existing class.  It reads, computes, and
# exposes values.  Existing code continues to work identically if this module
# is never imported.
# =============================================================================

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("phantom.config_bridge")

# ── lazy import to keep startup fast when hardware probe isn't needed ─────────
_profile = None          # PHANTOMHardwareProfile singleton


# ─────────────────────────────────────────────────────────────────────────────
# Config bridge class
# ─────────────────────────────────────────────────────────────────────────────

class PHANTOMConfig:
    """
    Unified config object.  All PHANTOM phases read from this — never from
    raw os.getenv() for performance-critical parameters.

    Attribute access falls back to safe defaults if phantom_startup() was not
    called (e.g., in unit tests), so nothing crashes.
    """

    def __init__(self):
        self._profile = None
        self._initialized = False

    def _init(self, profile) -> None:
        self._profile = profile
        self._initialized = True

    def _require(self):
        if not self._initialized:
            logger.warning(
                "[PHANTOM] phantom_startup() was not called before accessing "
                "PHANTOMConfig — using safe defaults. "
                "Add phantom_startup() to your FastAPI lifespan."
            )

    # ── hardware tier ────────────────────────────────────────────────────────

    @property
    def tier(self) -> str:
        self._require()
        return self._profile.tier.value if self._profile else "cpu_mid"

    @property
    def gpu_name(self) -> str:
        self._require()
        return self._profile.gpu_name if self._profile else "none"

    @property
    def gpu_vram_gb(self) -> float:
        self._require()
        return self._profile.gpu_vram_gb if self._profile else 0.0

    @property
    def system_ram_gb(self) -> float:
        self._require()
        return self._profile.system_ram_gb if self._profile else 8.0

    @property
    def cpu_cores(self) -> int:
        self._require()
        return self._profile.cpu_cores if self._profile else 4

    # ── embedding ────────────────────────────────────────────────────────────

    @property
    def embed_batch_size(self) -> int:
        """
        Replaces the hardcoded BATCH_SIZE = 256 in ingestion_service_v2.py.

        ingestion_service_v2.py Phase 1 change:
          Was:  BATCH_SIZE: int = 256
          Now:  from app.services.ingestion.phantom_config_bridge import phantom_cfg
                BATCH_SIZE = phantom_cfg.embed_batch_size
        """
        self._require()
        # Allow env override for manual tuning
        env = os.getenv("PHANTOM_EMBED_BATCH_SIZE")
        if env:
            return int(env)
        return self._profile.embed_batch_size if self._profile else 64

    @property
    def embed_prefetch(self) -> int:
        """Number of embedding batches to prefetch ahead of upsert."""
        self._require()
        return self._profile.embed_prefetch if self._profile else 2

    # ── vector DB ───────────────────────────────────────────────────────────

    @property
    def upsert_batch_size(self) -> int:
        self._require()
        env = os.getenv("PHANTOM_UPSERT_BATCH_SIZE")
        if env:
            return int(env)
        return self._profile.upsert_batch_size if self._profile else 128

    @property
    def upsert_concurrency(self) -> int:
        self._require()
        return self._profile.upsert_concurrency if self._profile else 2

    # ── ingestion workers ────────────────────────────────────────────────────

    @property
    def ingest_workers(self) -> int:
        """
        Number of parallel file ingestion tasks.
        Used by Phase 1 to replace sequential file processing.
        """
        self._require()
        env = os.getenv("PHANTOM_INGEST_WORKERS")
        if env:
            return int(env)
        return self._profile.ingest_workers if self._profile else 4

    @property
    def parse_workers(self) -> int:
        self._require()
        return self._profile.parse_workers if self._profile else 4

    @property
    def io_thread_pool(self) -> int:
        """
        Size of the ThreadPoolExecutor for sync I/O offloading.
        Used in BUG-2 fixes (run_in_executor) and Phase 2 Bloom gate.
        """
        self._require()
        return self._profile.io_thread_pool if self._profile else 8

    # ── deduplication ────────────────────────────────────────────────────────

    @property
    def bloom_capacity(self) -> int:
        """
        Expected max element count for the Bloom filter gate (Phase 2).
        Sized to RAM to keep false-positive rate at 0.1%.
        """
        self._require()
        env = os.getenv("PHANTOM_BLOOM_CAPACITY")
        if env:
            return int(env)
        return self._profile.bloom_capacity if self._profile else 2_000_000

    @property
    def bloom_error_rate(self) -> float:
        return self._profile.bloom_error_rate if self._profile else 0.001

    @property
    def l2_dedup_batch(self) -> int:
        """Chunk batch size for L2 embedding dedup (Phase 2 batch L2 fix)."""
        self._require()
        return self._profile.l2_dedup_batch if self._profile else 32

    # ── Phase 3: Gravity clustering ─────────────────────────────────────────

    @property
    def gravity_clusters(self) -> int:
        self._require()
        return self._profile.gravity_clusters if self._profile else 16

    @property
    def gravity_batch_size(self) -> int:
        self._require()
        return self._profile.gravity_batch_size if self._profile else 512

    # ── Phase 5: Stage collapse ──────────────────────────────────────────────

    @property
    def stage_collapse_concurrency(self) -> int:
        self._require()
        return self._profile.stage_collapse_concurrency if self._profile else 4

    # ── diagnostics ─────────────────────────────────────────────────────────

    @property
    def is_gpu(self) -> bool:
        """True if any GPU (ROCm, CUDA, or MPS) was detected."""
        self._require()
        if not self._profile:
            return False
        from app.services.ingestion.phantom_hardware_profiler import HardwareTier
        return self._profile.tier in (
            HardwareTier.AMD_ROCM,
            HardwareTier.CUDA,
            HardwareTier.APPLE_MPS,
        )

    @property
    def is_rocm(self) -> bool:
        """True specifically on AMD ROCm — primary target hardware."""
        self._require()
        if not self._profile:
            return False
        from app.services.ingestion.phantom_hardware_profiler import HardwareTier
        return self._profile.tier == HardwareTier.AMD_ROCM

    def as_dict(self) -> dict:
        """Return all config values as a plain dict (for /phantom/stats endpoint)."""
        return {
            "tier":                     self.tier,
            "gpu_name":                 self.gpu_name,
            "gpu_vram_gb":              self.gpu_vram_gb,
            "system_ram_gb":            self.system_ram_gb,
            "cpu_cores":                self.cpu_cores,
            "embed_batch_size":         self.embed_batch_size,
            "embed_prefetch":           self.embed_prefetch,
            "upsert_batch_size":        self.upsert_batch_size,
            "upsert_concurrency":       self.upsert_concurrency,
            "ingest_workers":           self.ingest_workers,
            "parse_workers":            self.parse_workers,
            "io_thread_pool":           self.io_thread_pool,
            "bloom_capacity":           self.bloom_capacity,
            "bloom_error_rate":         self.bloom_error_rate,
            "l2_dedup_batch":           self.l2_dedup_batch,
            "gravity_clusters":         self.gravity_clusters,
            "gravity_batch_size":       self.gravity_batch_size,
            "stage_collapse_concurrency": self.stage_collapse_concurrency,
            "is_gpu":                   self.is_gpu,
            "is_rocm":                  self.is_rocm,
        }

    def __repr__(self) -> str:
        if not self._initialized:
            return "PHANTOMConfig(uninitialized — call phantom_startup() first)"
        return (
            f"PHANTOMConfig("
            f"tier={self.tier!r}, "
            f"embed_batch={self.embed_batch_size}, "
            f"upsert_batch={self.upsert_batch_size}, "
            f"workers={self.ingest_workers}"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton — every PHANTOM component imports this
# ─────────────────────────────────────────────────────────────────────────────

phantom_cfg = PHANTOMConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Startup hook — call once from FastAPI lifespan
# ─────────────────────────────────────────────────────────────────────────────

def phantom_startup(verbose: bool = True) -> PHANTOMConfig:
    """
    Run hardware detection and populate the global phantom_cfg singleton.

    Call this ONCE from your FastAPI lifespan context manager:

        from contextlib import asynccontextmanager
        from app.services.ingestion.phantom_config_bridge import phantom_startup

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            phantom_startup()
            yield

        app = FastAPI(lifespan=lifespan)

    Subsequent imports of `phantom_cfg` anywhere in the codebase will return
    the fully populated singleton with hardware-tuned values.

    Args:
        verbose: If True (default) logs the hardware summary at startup.

    Returns:
        The populated PHANTOMConfig singleton (same object as `phantom_cfg`).
    """
    global phantom_cfg

    logger.info("[PHANTOM] Initializing hardware profile...")

    from app.services.ingestion.phantom_hardware_profiler import phantom_profile
    profile = phantom_profile(verbose=verbose)

    phantom_cfg._init(profile)

    logger.info(
        "[PHANTOM] Config bridge ready. "
        "embed_batch=%d  upsert_batch=%d  workers=%d  tier=%s",
        phantom_cfg.embed_batch_size,
        phantom_cfg.upsert_batch_size,
        phantom_cfg.ingest_workers,
        phantom_cfg.tier,
    )

    # Emit any hardware warnings at startup so they appear prominently in logs
    for w in profile.warnings:
        logger.warning("[PHANTOM] %s", w)

    return phantom_cfg


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI router — /phantom/stats endpoint (optional, wire in Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

def make_phantom_stats_router():
    """
    Returns a FastAPI APIRouter with a GET /phantom/stats endpoint.

    Wire into main.py in Phase 4:
        from app.services.ingestion.phantom_config_bridge import make_phantom_stats_router
        app.include_router(make_phantom_stats_router(), prefix="/phantom", tags=["phantom"])

    Response example:
        {
          "tier": "amd_rocm",
          "gpu_name": "AMD Radeon RX 6800 XT",
          "embed_batch_size": 256,
          ...
        }
    """
    try:
        from fastapi import APIRouter
        router = APIRouter()

        @router.get("/stats", summary="PHANTOM hardware profile and tuning parameters")
        async def phantom_stats():
            return {
                "status":  "ok" if phantom_cfg._initialized else "uninitialized",
                "config":  phantom_cfg.as_dict(),
            }

        return router
    except ImportError:
        logger.warning("[PHANTOM] FastAPI not installed — stats router not created.")
        return None
