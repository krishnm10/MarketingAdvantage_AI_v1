# =============================================
# video_ingestor_v1.py
#
# Enterprise Video → High-Quality Semantic Text Ingestor
#
# Features:
# - Premium visual understanding with scene detection
# - Clean, deduplicated transcript processing
# - Semantic scene segmentation
# - Advanced text quality enhancement
# - No duplicate/redundant content
# =============================================

import os
import asyncio
import tempfile
import re
from typing import Optional, List, Dict, Any, Set
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

from app.utils.logger import log_info, log_warning
from app.db.models.ingested_file_v2 import IngestedFileV2
from app.services.ingestion.ingestion_service_v2 import IngestionServiceV2
from app.db.session_v2 import get_async_session
from app.services.ingestion.media.media_hash_utils import MediaHashComputer
from sqlalchemy import select

# -----------------------------
# HARD LIMITS (ENTERPRISE)
# -----------------------------
MAX_VIDEO_MB = 2048  # 2GB max
MAX_VIDEO_DURATION_SEC = 60 * 60  # 1 hour max
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
KEYFRAMES_TO_EXTRACT = 12
AUDIO_CHUNK_DURATION = 300  # 5 minutes
SCENE_CHANGE_THRESHOLD = 30.0  # Threshold for scene change detection

# -----------------------------
# Lazy imports
# -----------------------------
def _import_video_libs():
    try:
        import cv2
        import numpy as np
        return cv2, np
    except ImportError as e:
        raise ImportError(
            "Video processing requires opencv-python. "
            "Install with: pip install opencv-python"
        ) from e


def _import_audio_extractor():
    try:
        from moviepy.editor import VideoFileClip
        return VideoFileClip
    except ImportError:
        log_warning("[VideoIngestorV1] moviepy not available, audio extraction disabled")
        return None


# -----------------------------
# Whisper Singleton
# -----------------------------
_WHISPER_MODEL = None
_WHISPER_LOCK = asyncio.Lock()

async def get_whisper_model(model_size: str = "base"):
    global _WHISPER_MODEL
    async with _WHISPER_LOCK:
        if _WHISPER_MODEL is None:
            log_info("[VideoIngestorV1] Loading Whisper model (singleton)")
            import whisper
            _WHISPER_MODEL = whisper.load_model(model_size)
    return _WHISPER_MODEL


# -----------------------------
# Text Quality Enhancement Utilities
# -----------------------------
class TextQualityEnhancer:
    """Ensures high-quality, deduplicated text output"""
    
    @staticmethod
    def clean_transcript(text: str) -> str:
        """Remove noise, timestamps, and artifacts from transcript"""
        if not text:
            return ""
        
        # Remove timestamp markers like [00:00:00]
        text = re.sub(r'\[\d+:\d+:\d+\]|\[\d+:\d+\]', '', text)
        
        # Remove filler words (uh, um, ah)
        text = re.sub(r'\b(uh|um|ah|er|hmm)\b', '', text, flags=re.IGNORECASE)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove music/sound effect markers
        text = re.sub(r'\[.*?(music|sound|applause|laughter).*?\]', '', text, flags=re.IGNORECASE)
        
        # Fix common transcription errors
        text = text.replace(' i ', ' I ')
        text = text.replace(" i'm ", " I'm ")
        
        return text.strip()
    
    @staticmethod
    def deduplicate_sentences(text: str) -> str:
        """Remove duplicate sentences while preserving order"""
        if not text:
            return ""
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Use OrderedDict to preserve order while removing duplicates
        seen = OrderedDict()
        for sentence in sentences:
            sentence_clean = sentence.strip().lower()
            if sentence_clean and sentence_clean not in seen:
                seen[sentence_clean] = sentence.strip()
        
        return ' '.join(seen.values())
    
    @staticmethod
    def remove_redundant_phrases(text: str) -> str:
        """Remove repetitive phrases that don't add value"""
        if not text:
            return ""
        
        # Common redundant phrases in videos
        redundant_patterns = [
            r'please like and subscribe',
            r'don\'t forget to subscribe',
            r'hit the bell icon',
            r'comment below',
            r'check the description',
            r'link in the description',
            r'thanks for watching',
        ]
        
        for pattern in redundant_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    @staticmethod
    def segment_by_topic(text: str) -> List[str]:
        """Segment text into topical chunks for better semantic search"""
        if not text:
            return []
        
        # Split by paragraph breaks or major punctuation
        segments = re.split(r'\n\n+|(?<=[.!?])\s{2,}', text)
        
        # Filter out very short segments (likely noise)
        quality_segments = [
            seg.strip() 
            for seg in segments 
            if len(seg.strip()) > 50  # At least 50 chars
        ]
        
        return quality_segments


# -----------------------------
# Scene Analysis for High-Quality Visual Understanding
# -----------------------------
class SceneAnalyzer:
    """Advanced scene detection and description"""
    
    @staticmethod
    def detect_scene_changes(cap, frame_count: int, fps: float) -> List[int]:
        """Detect scene changes using histogram comparison"""
        cv2, np = _import_video_libs()
        
        scene_changes = [0]  # Start is always a scene
        prev_hist = None
        check_interval = max(int(fps * 2), 1)  # Check every 2 seconds
        
        for frame_idx in range(0, frame_count, check_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Calculate histogram
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            if prev_hist is not None:
                # Compare with previous frame
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                
                # If difference is significant, it's a scene change
                if diff > SCENE_CHANGE_THRESHOLD / 100:
                    scene_changes.append(frame_idx)
            
            prev_hist = hist
            del frame
        
        return scene_changes[:KEYFRAMES_TO_EXTRACT]  # Limit to keyframe count
    
    @staticmethod
    def analyze_frame_quality(frame) -> Dict[str, Any]:
        """Analyze frame for quality metrics"""
        cv2, np = _import_video_libs()
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate metrics
        brightness = np.mean(gray)
        contrast = gray.std()
        
        # Edge detection for content complexity
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        
        # Color analysis
        color_variance = np.var(frame)
        
        # Dominant colors
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180])
        dominant_color_idx = np.argmax(hue_hist)
        
        color_names = ["Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Purple", "Magenta"]
        dominant_color = color_names[dominant_color_idx]
        
        return {
            "brightness": brightness,
            "contrast": contrast,
            "edge_density": edge_density,
            "color_variance": color_variance,
            "dominant_color": dominant_color,
            "is_bright": brightness > 127,
            "is_high_contrast": contrast > 50,
            "is_complex": edge_density > 0.1,
        }
    
    @staticmethod
    def generate_scene_description(analysis: Dict[str, Any], timestamp: float) -> str:
        """Generate natural language scene description"""
        parts = [f"Scene at {timestamp:.1f}s:"]
        
        # Lighting
        if analysis["is_bright"]:
            parts.append("well-lit")
        else:
            parts.append("low-light")
        
        # Content complexity
        if analysis["is_complex"]:
            parts.append("detailed visual content")
        else:
            parts.append("simple composition")
        
        # Color information
        parts.append(f"dominant {analysis['dominant_color'].lower()} tones")
        
        # Contrast
        if analysis["is_high_contrast"]:
            parts.append("high contrast")
        
        return ", ".join(parts) + "."


# -----------------------------
# Video Ingestor with Premium Quality
# -----------------------------
class VideoIngestorV1:
    """
    Enterprise-grade video ingestion with premium text quality.
    
    Features:
    - Advanced scene detection and analysis
    - Clean, deduplicated transcripts
    - High-quality semantic descriptions
    - Streaming processing for large files
    - No duplicate content in output
    """
    
    def __init__(self):
        self.text_enhancer = TextQualityEnhancer()
        self.scene_analyzer = SceneAnalyzer()
        
    async def ingest(
        self,
        file_id: str,
        video_path: str,
        business_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Enterprise-grade video ingestion with deduplication.

        Session Discipline (BUG-5 fix):
            Phase 1 — dedup check only         (short DB session, closed before CPU)
            Phase 2 — validate + process       (NO DB connection during cv2 + Whisper)
            Phase 3 — file record + pipeline   (short DB session for writes only)

        BUG-5 FIX: Previously the entire function ran inside one
        `async with get_async_session() as db:` block, holding a DB connection
        across cv2 scene analysis + moviepy audio extraction + Whisper transcription.
        A long video could hold the connection for hours, exhausting the pool.
        """
        log_info(f"[VideoIngestorV1] Processing video: {video_path}")

        # ============================================================
        # PHASE 1: Short-lived DB session — dedup check ONLY.
        # Session is closed before any CPU-intensive work begins.
        # ============================================================
        video_hash, byte_hash = MediaHashComputer.compute_video_hash(video_path)

        async with get_async_session() as db:
            result = await db.execute(
                select(IngestedFileV2).where(
                    IngestedFileV2.media_hash == video_hash
                )
            )
            existing_file = result.scalar_one_or_none()
        # DB SESSION CLOSED — before any blocking CPU work.

        if existing_file:
            log_info(
                f"[VideoIngestorV1] DUPLICATE DETECTED: "
                f"{os.path.basename(video_path)} matches {existing_file.file_name}"
            )
            return {
                "status":        "duplicate_skipped",
                "duplicate_of":  str(existing_file.id),
                "original_file": existing_file.file_name,
                "video_hash":    video_hash[:16],
                "message":       f"Video is duplicate of: {existing_file.file_name}",
            }

        log_info(f"[VideoIngestorV1] Unique video confirmed")

        # ============================================================
        # PHASE 2: CPU-intensive work — NO DB connection held.
        # BUG-5 FIX: This block previously ran inside the DB session.
        # ============================================================

        # 2a. Validate (no DB needed — filesystem + format + duration)
        try:
            self._validate_video(video_path)
        except ValueError as e:
            log_warning(f"[VideoIngestorV1] Validation failed: {e}")
            return {"status": "failed", "error": str(e)}

        # 2b. Extract metadata (cv2)
        metadata = await self._extract_metadata_streaming(video_path)

        # 2c. Scene-based keyframe extraction (cv2)
        keyframes_text = await self._extract_scenes_premium(video_path, metadata)

        # 2d. Audio transcription (moviepy + Whisper)
        audio_transcript = await self._extract_audio_premium(video_path, metadata)

        # 2e. Combine with quality enhancement
        semantic_text = self._combine_premium_quality(
            keyframes_text=keyframes_text,
            audio_transcript=audio_transcript,
            metadata=metadata,
        )

        if not semantic_text.strip():
            log_warning("[VideoIngestorV1] No quality content extracted")
            return {"status": "failed", "error": "No semantic content extracted"}

        parsed_payload = {
            "raw_text": semantic_text,
            "meta": {
                "media_type":        "video",
                "confidence_source": "model",
                "duration_seconds":  metadata.get("duration", 0),
                "fps":               metadata.get("fps", 0),
                "resolution":        metadata.get("resolution", "unknown"),
                "has_audio":         bool(audio_transcript),
                "scenes_analyzed":   metadata.get("scenes_count", 0),
                "video_hash":        video_hash,
                "text_quality":      "premium",
                "deduplicated":      True,
            },
        }

        # ============================================================
        # PHASE 3: New short-lived DB session for writes + pipeline.
        # DB connection acquired AFTER all CPU work is complete.
        # ============================================================
        async with get_async_session() as db:

            # 3a. Upsert file record
            file_record = await IngestionServiceV2._get_file_record(db, file_id)

            if not file_record:
                size_mb = os.path.getsize(video_path) / (1024 * 1024)
                file_record = IngestedFileV2(
                    id=file_id,
                    file_name=os.path.basename(video_path),
                    file_type="video",
                    file_path=video_path,
                    business_id=business_id,
                    media_hash=video_hash,
                    meta_data={
                        "source_type":   "video",
                        "ingested_via":  "video_ingestor_v1",
                        "video_hash":    video_hash,
                        "byte_hash":     byte_hash,
                        "dedup_method":  "keyframe_perceptual_hash",
                        "file_size_mb":  round(size_mb, 2),
                        "quality_mode":  "premium",
                    },
                    status="uploaded",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                db.add(file_record)
                await db.commit()

            # 3b. Run through ingestion pipeline
            await IngestionServiceV2._run_pipeline(
                db=db,
                file_record=file_record,
                parsed_payload=parsed_payload,
            )

        log_info(f"[VideoIngestorV1] Completed premium video ingestion: {file_id}")

        return {
            "status":          "success",
            "file_id":         file_id,
            "video_hash":      video_hash[:16],
            "duration":        metadata.get("duration", 0),
            "scenes_analyzed": metadata.get("scenes_count", 0),
            "text_quality":    "premium",
            "message":         "Video ingested with premium quality text",
        }
    
    # ============================================
    # HELPER METHODS
    # ============================================
    
    def _validate_video(self, path: str):
        """
        Validate video file before any DB interaction or processing.

        BUG-7 FIX: MAX_VIDEO_DURATION_SEC = 3600 was defined at module level
        but was NEVER used in this method. A 5-hour 4K video passed validation.
        Duration check now uses cv2 (already a hard dependency of this module)
        to read the frame count and fps from the container header without
        decoding any frames.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format: {ext}")

        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > MAX_VIDEO_MB:
            raise ValueError(
                f"File too large: {size_mb:.1f} MB (limit: {MAX_VIDEO_MB} MB)"
            )

        # Duration check via cv2 — reads container metadata, no frame decoding
        try:
            cv2, _ = _import_video_libs()
            cap = cv2.VideoCapture(path)
            fps         = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if fps > 0 and frame_count > 0:
                duration = frame_count / fps
                if duration > MAX_VIDEO_DURATION_SEC:
                    raise ValueError(
                        f"Video duration {duration:.0f}s exceeds limit of "
                        f"{MAX_VIDEO_DURATION_SEC}s "
                        f"({MAX_VIDEO_DURATION_SEC // 60} min)"
                    )
        except ValueError:
            raise  # Re-raise our own ValidationErrors unchanged
        except Exception as e:
            log_warning(
                f"[VideoIngestorV1] Duration check failed for "
                f"'{os.path.basename(path)}': {e}"
            )

        log_info(f"[VideoIngestorV1] Video size: {size_mb:.2f} MB")
    
    async def _extract_metadata_streaming(self, video_path: str) -> Dict[str, Any]:
        cv2, np = _import_video_libs()
        loop = asyncio.get_running_loop()
        
        def _get_metadata():
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            return {
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "resolution": f"{width}x{height}",
                "file_size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2),
            }
        
        return await loop.run_in_executor(None, _get_metadata)
    
    async def _extract_scenes_premium(self, video_path: str, metadata: Dict) -> str:
        """Premium scene extraction with quality analysis"""
        cv2, np = _import_video_libs()
        loop = asyncio.get_running_loop()
        
        def _extract():
            cap = cv2.VideoCapture(video_path)
            
            # Detect scene changes
            scene_frames = self.scene_analyzer.detect_scene_changes(
                cap, 
                metadata["frame_count"],
                metadata["fps"]
            )
            
            descriptions = []
            for frame_idx in scene_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    timestamp = frame_idx / metadata["fps"]
                    analysis = self.scene_analyzer.analyze_frame_quality(frame)
                    description = self.scene_analyzer.generate_scene_description(analysis, timestamp)
                    descriptions.append(description)
                    del frame
            
            cap.release()
            metadata["scenes_count"] = len(descriptions)
            return "\n".join(descriptions)
        
        scenes = await loop.run_in_executor(None, _extract)
        log_info(f"[VideoIngestorV1] Analyzed {metadata.get('scenes_count', 0)} scenes (premium)")
        return scenes
    
    async def _extract_audio_premium(self, video_path: str, metadata: Dict) -> str:
        """Extract and clean audio transcript"""
        duration = metadata.get("duration", 0)
        
        # Get raw transcript
        if duration > AUDIO_CHUNK_DURATION:
            raw_transcript = await self._process_audio_chunks(video_path, duration)
        else:
            raw_transcript = await self._extract_audio_full(video_path)
        
        if not raw_transcript:
            return ""
        
        # Premium text enhancement
        cleaned = self.text_enhancer.clean_transcript(raw_transcript)
        deduplicated = self.text_enhancer.deduplicate_sentences(cleaned)
        final = self.text_enhancer.remove_redundant_phrases(deduplicated)
        
        log_info(f"[VideoIngestorV1] Audio: {len(raw_transcript)}→{len(final)} chars (enhanced)")
        return final
    
    async def _process_audio_chunks(self, video_path: str, duration: float) -> str:
        """Process audio in chunks"""
        transcripts = []
        chunk_count = int(duration / AUDIO_CHUNK_DURATION) + 1
        
        for chunk_idx in range(chunk_count):
            start = chunk_idx * AUDIO_CHUNK_DURATION
            end = min((chunk_idx + 1) * AUDIO_CHUNK_DURATION, duration)
            
            audio_chunk = await self._extract_audio_chunk(video_path, start, end, chunk_idx)
            
            if audio_chunk and os.path.exists(audio_chunk):
                chunk_text = await self._transcribe_audio_file(audio_chunk)
                if chunk_text:
                    transcripts.append(chunk_text)
                
                try:
                    os.remove(audio_chunk)
                except:
                    pass
        
        return " ".join(transcripts)
    
    async def _extract_audio_full(self, video_path: str) -> str:
        """Extract full audio for short videos"""
        try:
            VideoFileClip = _import_audio_extractor()
            if not VideoFileClip:
                return ""
            
            temp_audio = os.path.join(tempfile.gettempdir(), f"video_audio_{os.getpid()}.wav")
            loop = asyncio.get_running_loop()
            
            def _extract():
                clip = VideoFileClip(video_path)
                if clip.audio:
                    clip.audio.write_audiofile(temp_audio, codec='pcm_s16le', logger=None)
                    clip.close()
                    return temp_audio
                clip.close()
                return None
            
            audio_path = await loop.run_in_executor(None, _extract)
            
            if audio_path:
                transcript = await self._transcribe_audio_file(audio_path)
                try:
                    os.remove(audio_path)
                except:
                    pass
                return transcript
            
            return ""
        except Exception as e:
            log_warning(f"[VideoIngestorV1] Audio extraction failed: {e}")
            return ""
    
    async def _extract_audio_chunk(self, video_path: str, start: float, end: float, idx: int) -> Optional[str]:
        """Extract audio chunk"""
        try:
            VideoFileClip = _import_audio_extractor()
            if not VideoFileClip:
                return None
            
            chunk_path = os.path.join(tempfile.gettempdir(), f"video_chunk_{idx}_{os.getpid()}.wav")
            loop = asyncio.get_running_loop()
            
            def _extract():
                clip = VideoFileClip(video_path)
                if clip.audio:
                    audio = clip.audio.subclip(start, end)
                    audio.write_audiofile(chunk_path, codec='pcm_s16le', logger=None)
                    clip.close()
                    return chunk_path
                clip.close()
                return None
            
            return await loop.run_in_executor(None, _extract)
        except Exception as e:
            log_warning(f"[VideoIngestorV1] Chunk extraction failed: {e}")
            return None
    
    async def _transcribe_audio_file(self, audio_path: str) -> str:
        """Transcribe using Whisper"""
        try:
            model = await get_whisper_model()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: model.transcribe(audio_path))
            return (result.get("text") or "").strip()
        except Exception as e:
            log_warning(f"[VideoIngestorV1] Transcription failed: {e}")
            return ""
    
    def _combine_premium_quality(
        self,
        keyframes_text: str,
        audio_transcript: str,
        metadata: Dict
    ) -> str:
        """Combine visual + audio with quality enhancement"""
        parts = [
            f"Video Analysis Summary:",
            f"Duration: {metadata['duration']:.1f}s, Resolution: {metadata['resolution']}, Size: {metadata['file_size_mb']:.2f}MB",
            ""
        ]
        
        if keyframes_text:
            parts.extend(["Premium Visual Analysis:", keyframes_text, ""])
        
        if audio_transcript:
            # Segment transcript by topic for better search
            segments = self.text_enhancer.segment_by_topic(audio_transcript)
            if segments:
                parts.extend(["Enhanced Audio Content:"] + segments + [""])
            else:
                parts.extend(["Enhanced Audio Content:", audio_transcript, ""])
        
        final_text = "\n".join(parts)
        
        # Final deduplication pass
        return self.text_enhancer.deduplicate_sentences(final_text)