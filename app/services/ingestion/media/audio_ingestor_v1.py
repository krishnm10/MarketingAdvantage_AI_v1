# =============================================
# audio_ingestor_v1.py
#
# Enterprise Audio → Semantic Text Ingestor
#
# BUG FIXES (this revision):
#
#   BUG-4 [CRITICAL] — DB session held open during Whisper transcription
#     BEFORE: async with get_async_session() as db: wrapped the entire function
#             including the blocking Whisper.transcribe() call (minutes of CPU).
#             A single DB connection was held for the full transcription window.
#             Under concurrent load this exhausts the connection pool entirely.
#     AFTER:  Three-phase session discipline:
#               Phase 1 — short-lived session: dedup check ONLY.
#               Phase 2 — NO session: validate + Whisper transcription.
#               Phase 3 — short-lived session: file record upsert + pipeline.
#             DB connections are held for milliseconds, not minutes.
#
#   BUG-6 [HIGH] — Duration check only worked for .wav
#     BEFORE: wave.open() is wav-only. All other formats hit except: pass,
#             bypassing the duration guard silently. A 5-hour MP3 passed.
#     AFTER:  Uses mutagen for multi-format duration (mp3, m4a, flac, etc).
#             Falls back with a logged warning rather than a silent pass.
#
#   BUG-8 [HIGH] — SUPPORTED_EXTENSIONS mismatch with watcher_ingestor_v2
#     BEFORE: {".wav", ".mp3", ".m4a", ".flac"} — 4 extensions (.aac, .ogg,
#             .opus, .wma) were routed here by watcher but always failed with
#             ValueError: Unsupported audio format.
#     AFTER:  Extended to match watcher AUDIO_EXTENSIONS exactly.
#
# Architecture guarantees (unchanged):
#   - Hard safety guards (size, duration, format)
#   - Singleton Whisper model
#   - Semantic segmentation
#   - Retrieval-safe chunking
#   - Governance-ready metadata
#   - NO direct DB/vector writes (delegated to IngestionServiceV2._run_pipeline)
# =============================================

import os
import re
import asyncio
from typing import Optional, List
from datetime import datetime

from app.utils.logger import log_info, log_warning
from app.db.models.ingested_file_v2 import IngestedFileV2
from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2
from app.db.session_v2 import get_async_session
from app.services.ingestion.media.media_hash_utils import MediaHashComputer
from sqlalchemy import select


# =============================================
# HARD LIMITS (ENTERPRISE)
# =============================================

MAX_AUDIO_MB           = 50
MAX_AUDIO_DURATION_SEC = 30 * 60  # 30 minutes

# BUG-8 FIX: Extended to match watcher_ingestor_v2.AUDIO_EXTENSIONS exactly.
# Previously {".wav", ".mp3", ".m4a", ".flac"} — 4 formats were silently failing.
SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".aac",   # NEW — was routed by watcher but rejected here
    ".ogg",   # NEW
    ".opus",  # NEW
    ".wma",   # NEW
}


# =============================================
# Whisper Singleton
# =============================================

_WHISPER_MODEL = None
_WHISPER_LOCK  = asyncio.Lock()


async def get_whisper_model(model_size: str = "base"):
    global _WHISPER_MODEL
    async with _WHISPER_LOCK:
        if _WHISPER_MODEL is None:
            log_info("[AudioIngestorV1] Loading Whisper model (singleton)")
            import whisper
            _WHISPER_MODEL = whisper.load_model(model_size)
        return _WHISPER_MODEL


# =============================================
# Audio Ingestor
# =============================================

class AudioIngestorV1:
    """
    Enterprise-grade audio ingestion adapter.

    Session Discipline (BUG-4 fix):
        Phase 1 — dedup check only         (short DB session, closed before CPU)
        Phase 2 — validate + transcribe    (NO DB connection held during Whisper)
        Phase 3 — file record + pipeline   (short DB session for writes only)

    This ensures DB connections are held for milliseconds, not minutes.
    Under concurrent load the connection pool is never exhausted by Whisper.
    """

    async def ingest(
        self,
        file_id: str,
        audio_path: str,
        business_id: Optional[str] = None,
    ) -> dict:
        """
        Full audio ingestion pipeline with acoustic fingerprint deduplication.
        """
        log_info(f"[AudioIngestorV1] Processing audio: {audio_path}")

        # ============================================================
        # PHASE 1: Short-lived DB session — dedup check ONLY.
        # Session is closed before any CPU-intensive work begins.
        # BUG-4 FIX: Previously this session wrapped the entire function.
        # ============================================================
        acoustic_hash, byte_hash = MediaHashComputer.compute_audio_hash(audio_path)

        async with get_async_session() as db:
            result = await db.execute(
                select(IngestedFileV2).where(
                    IngestedFileV2.media_hash == acoustic_hash
                )
            )
            existing_file = result.scalar_one_or_none()
        # DB SESSION CLOSED — before any blocking CPU work.

        if existing_file:
            log_info(
                f"[AudioIngestorV1] DUPLICATE DETECTED: "
                f"{os.path.basename(audio_path)} matches {existing_file.file_name} "
                f"(ID: {existing_file.id}, Hash: {acoustic_hash[:12]}...)"
            )
            return {
                "status":        "duplicate_skipped",
                "duplicate_of":  str(existing_file.id),
                "original_file": existing_file.file_name,
                "acoustic_hash": acoustic_hash[:16],
                "message": (
                    f"Audio is duplicate of: {existing_file.file_name}"
                ),
            }

        # ============================================================
        # PHASE 2: CPU-intensive work — NO DB connection held.
        # Validate first (cheap), then transcribe (expensive).
        # BUG-4 FIX: This block previously ran inside the DB session.
        # ============================================================

        # 2a. Validate (no DB needed — pure filesystem + format check)
        try:
            self._validate_audio(audio_path)
        except ValueError as e:
            log_warning(f"[AudioIngestorV1] Validation failed: {e}")
            return {"status": "failed", "error": str(e)}

        # 2b. Transcribe (executor-offloaded — does not block event loop)
        log_info(f"[AudioIngestorV1] Unique audio confirmed — starting transcription")
        model = await get_whisper_model()
        loop  = asyncio.get_running_loop()

        try:
            whisper_result = await loop.run_in_executor(
                None,
                lambda: model.transcribe(audio_path),
            )
        except Exception as e:
            log_warning(f"[AudioIngestorV1] Transcription failed: {e}")
            return {"status": "failed", "error": f"Transcription failed: {e}"}

        transcript = (whisper_result.get("text") or "").strip()
        if not transcript:
            log_warning("[AudioIngestorV1] Empty transcript — no speech detected")
            return {
                "status": "failed",
                "error":  "Empty transcript — no speech detected",
            }

        # 2c. Clean and segment (pure Python — no I/O)
        cleaned  = self._clean_transcript(transcript)
        segments = self._segment_transcript(cleaned)

        if not segments:
            log_warning("[AudioIngestorV1] No semantic segments produced")
            return {"status": "failed", "error": "No semantic segments extracted"}

        parsed_payload = {
            "raw_text": "\n".join(segments),
            "meta": {
                "media_type":           "audio",
                "confidence_source":    "model",
                "transcription_model":  "whisper",
                "segment_count":        len(segments),
                "acoustic_fingerprint": acoustic_hash,
                "byte_hash":            byte_hash,
            },
        }

        # ============================================================
        # PHASE 3: New short-lived DB session for writes + pipeline.
        # DB connection acquired here — AFTER all CPU work is complete.
        # BUG-4 FIX: Previously this reused the same session opened in Phase 1.
        # ============================================================
        async with get_async_session() as db:

            # 3a. Upsert file record
            file_record = await IngestionServiceV2._get_file_record(db, file_id)

            if not file_record:
                file_record = IngestedFileV2(
                    id=file_id,
                    file_name=os.path.basename(audio_path),
                    file_type="audio",
                    file_path=audio_path,
                    business_id=business_id,
                    media_hash=acoustic_hash,
                    meta_data={
                        "source_type":          "audio",
                        "ingested_via":         "audio_ingestor_v1",
                        "acoustic_fingerprint": acoustic_hash,
                        "byte_hash":            byte_hash,
                        "dedup_method":         "chromaprint_or_mfcc",
                    },
                    status="uploaded",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                db.add(file_record)
                await db.commit()
            else:
                # Backfill media_hash if the record predates this feature
                if not file_record.media_hash:
                    file_record.media_hash = acoustic_hash
                    meta = dict(file_record.meta_data or {})
                    meta.update({
                        "acoustic_fingerprint": acoustic_hash,
                        "byte_hash":            byte_hash,
                        "dedup_method":         "chromaprint_or_mfcc",
                    })
                    file_record.meta_data = meta
                    await db.commit()

            # 3b. Run through ingestion pipeline
            await IngestionServiceV2._run_pipeline(
                db=db,
                file_record=file_record,
                parsed_payload=parsed_payload,
            )

        log_info(f"[AudioIngestorV1] Completed audio ingestion: {file_id}")

        return {
            "status":        "success",
            "file_id":       file_id,
            "acoustic_hash": acoustic_hash[:16],
            "segment_count": len(segments),
            "message":       "Audio ingested successfully",
        }

    # =============================================
    # Helpers
    # =============================================

    def _validate_audio(self, path: str) -> None:
        """
        Validate audio file. Must be called before any DB interaction or transcription.

        BUG-6 FIX: Original used wave.open() (wav-only).
          For .mp3/.m4a/.flac/.aac/.ogg/.opus/.wma, wave.open() raised an exception
          that was caught by `except: pass`, silently bypassing the duration guard.
          A 5-hour MP3 would pass validation and be sent to Whisper.

          Fix: Uses mutagen.File() which reads duration from the container header
          without decoding audio, and supports all 8 SUPPORTED_EXTENSIONS.
          If mutagen is unavailable, a warning is logged — never a silent pass.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format: '{ext}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > MAX_AUDIO_MB:
            raise ValueError(
                f"Audio file too large: {size_mb:.1f} MB "
                f"(limit: {MAX_AUDIO_MB} MB)"
            )

        # Duration check — multi-format via mutagen
        try:
            from mutagen import File as MutaGenFile
            audio_meta = MutaGenFile(path)
            if audio_meta is not None and hasattr(audio_meta, "info"):
                duration = getattr(audio_meta.info, "length", None)
                if duration is not None and duration > MAX_AUDIO_DURATION_SEC:
                    raise ValueError(
                        f"Audio duration {duration:.0f}s exceeds limit of "
                        f"{MAX_AUDIO_DURATION_SEC}s "
                        f"({MAX_AUDIO_DURATION_SEC // 60} min)"
                    )
        except ValueError:
            raise  # Re-raise our own ValidationErrors unchanged
        except ImportError:
            log_warning(
                "[AudioIngestorV1] mutagen not installed — duration check skipped. "
                "Install with: pip install mutagen"
            )
        except Exception as e:
            # Corrupt or unrecognised header — log, let Whisper fail naturally
            log_warning(
                f"[AudioIngestorV1] Duration metadata read failed for "
                f"'{os.path.basename(path)}': {e}"
            )

    def _clean_transcript(self, text: str) -> str:
        """Remove timestamp markers, filler words, and extra whitespace."""
        text = re.sub(r"\[\d+:\d+(:\d+)?\]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\b(uh|um|ah)\b", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _segment_transcript(self, text: str) -> List[str]:
        """
        Deterministic sentence-boundary segmentation for retrieval.
        Produces ~400-char chunks aligned with the embedding window.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        buf:    List[str] = []

        for sentence in sentences:
            buf.append(sentence)
            if len(" ".join(buf)) > 400:
                chunks.append(" ".join(buf).strip())
                buf = []

        if buf:
            chunks.append(" ".join(buf).strip())

        return [c for c in chunks if c]