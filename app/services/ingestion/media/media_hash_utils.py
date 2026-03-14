# =============================================
# media_hash_utils.py
# 
# Enterprise Media Deduplication Utilities
# Supports: Images, Audio, Video
# =============================================

import hashlib
import os
from typing import Optional, Tuple
from PIL import Image
import imagehash
from app.utils.logger import log_info, log_warning


class MediaHashComputer:
    """
    Enterprise-grade media deduplication using perceptual hashing.
    Supports images, audio, and video with configurable similarity thresholds.
    """
    
    # Hamming distance thresholds for duplicate detection
    IMAGE_DUPLICATE_THRESHOLD = 5  # bits difference for images (0-5 = very similar)
    AUDIO_DUPLICATE_THRESHOLD = 10  # similarity threshold for audio
    
    @staticmethod
    def compute_image_hash(image_path: str) -> Tuple[str, str]:
        """
        Compute perceptual hash for images using dHash algorithm.
        
        Returns:
            Tuple[str, str]: (perceptual_hash, byte_hash)
            - perceptual_hash: Robust to minor edits, resizing, compression
            - byte_hash: Exact file hash for identical files
        """
        try:
            with Image.open(image_path) as img:
                # dHash - difference hash, robust to minor edits
                dhash = imagehash.dhash(img, hash_size=16)  # 256-bit hash
                perceptual_hash = str(dhash)
            
            # Also compute byte-level hash for exact duplicates
            byte_hash = MediaHashComputer._compute_file_hash(image_path)
            
            log_info(f"[MediaHash] Image hash computed: {perceptual_hash[:12]}... (file: {os.path.basename(image_path)})")
            return perceptual_hash, byte_hash
            
        except Exception as e:
            log_warning(f"[MediaHash] Failed to hash image {image_path}: {e}")
            # Fallback to file hash only
            byte_hash = MediaHashComputer._compute_file_hash(image_path)
            return byte_hash, byte_hash
    
    @staticmethod
    def compute_audio_hash(audio_path: str) -> Tuple[str, str]:
        """
        Compute acoustic fingerprint for audio files.
        
        Returns:
            Tuple[str, str]: (acoustic_fingerprint, byte_hash)
        """
        # Try Chromaprint first (most robust)
        try:
            import acoustid
            
            duration, fingerprint = acoustid.fingerprint_file(audio_path)
            acoustic_hash = hashlib.sha256(fingerprint.encode()).hexdigest()
            byte_hash = MediaHashComputer._compute_file_hash(audio_path)
            
            log_info(f"[MediaHash] Audio fingerprint (Chromaprint) computed: {acoustic_hash[:12]}...")
            return acoustic_hash, byte_hash
            
        except Exception as e:
            log_warning(f"[MediaHash] Chromaprint failed, using librosa fallback: {e}")
            
            # Fallback: use librosa for MFCC-based fingerprint
            try:
                import librosa
                import numpy as np
                
                # Load first 30 seconds (enough for fingerprinting)
                y, sr = librosa.load(audio_path, sr=22050, duration=30, mono=True)
                
                # Extract MFCC features (Mel-frequency cepstral coefficients)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                
                # Compute hash from MFCC
                mfcc_hash = hashlib.sha256(mfcc.tobytes()).hexdigest()
                byte_hash = MediaHashComputer._compute_file_hash(audio_path)
                
                log_info(f"[MediaHash] Audio fingerprint (librosa) computed: {mfcc_hash[:12]}...")
                return mfcc_hash, byte_hash
                
            except Exception as e2:
                log_warning(f"[MediaHash] All audio hashing methods failed: {e2}")
                # Last resort: use file hash only
                byte_hash = MediaHashComputer._compute_file_hash(audio_path)
                return byte_hash, byte_hash
    
    @staticmethod
    def compute_video_hash(video_path: str) -> Tuple[str, str]:
        """
        Compute combined hash for video (keyframe-based perceptual hash).
        
        Returns:
            Tuple[str, str]: (combined_hash, byte_hash)
        """
        try:
            import cv2
            
            # Extract 5 keyframes evenly distributed
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count == 0:
                raise ValueError("Video has no frames")
            
            keyframe_hashes = []
            for i in range(5):
                frame_idx = int((i / 4) * max(frame_count - 1, 0))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert frame to PIL Image for perceptual hashing
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    frame_hash = str(imagehash.dhash(pil_img, hash_size=8))
                    keyframe_hashes.append(frame_hash)
            
            cap.release()
            
            if not keyframe_hashes:
                raise ValueError("No frames could be extracted")
            
            # Combine keyframe hashes
            combined = "_".join(keyframe_hashes)
            video_hash = hashlib.sha256(combined.encode()).hexdigest()
            byte_hash = MediaHashComputer._compute_file_hash(video_path)
            
            log_info(f"[MediaHash] Video hash computed: {video_hash[:12]}... ({len(keyframe_hashes)} keyframes)")
            return video_hash, byte_hash
            
        except Exception as e:
            log_warning(f"[MediaHash] Video hashing failed: {e}")
            # Fallback to file hash
            byte_hash = MediaHashComputer._compute_file_hash(video_path)
            return byte_hash, byte_hash
    
    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        """
        Compute SHA256 hash of file bytes.
        Used for exact duplicate detection.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    @staticmethod
    def is_duplicate_image(hash1: str, hash2: str) -> bool:
        """
        Check if two image hashes represent duplicates.
        Uses Hamming distance for perceptual hash comparison.
        """
        try:
            # If exact match, definitely duplicate
            if hash1 == hash2:
                return True
            
            # Both hashes must be same length for comparison
            if len(hash1) != len(hash2):
                return False
            
            # Calculate Hamming distance (number of differing bits)
            distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
            is_dup = distance <= MediaHashComputer.IMAGE_DUPLICATE_THRESHOLD
            
            if is_dup:
                log_info(f"[MediaHash] Image duplicate detected (distance: {distance})")
            
            return is_dup
            
        except Exception as e:
            log_warning(f"[MediaHash] Error comparing hashes: {e}")
            return hash1 == hash2
    
    @staticmethod
    def is_duplicate_audio(hash1: str, hash2: str) -> bool:
        """
        Check if two audio hashes represent duplicates.
        Acoustic fingerprints are already robust, so exact match is used.
        """
        return hash1 == hash2
    
    @staticmethod
    def is_duplicate_video(hash1: str, hash2: str) -> bool:
        """
        Check if two video hashes represent duplicates.
        Keyframe-based hashes use exact matching.
        """
        return hash1 == hash2


# =============================================
# Convenience function for routing
# =============================================

def compute_media_hash(file_path: str, media_type: str = None) -> Tuple[str, str]:
    """
    Unified media hash computation with automatic type detection.
    
    Args:
        file_path: Path to media file
        media_type: Optional media type hint ('image', 'audio', 'video')
    
    Returns:
        Tuple[str, str]: (perceptual_hash, byte_hash)
    """
    if media_type is None:
        # Auto-detect from extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}:
            media_type = 'image'
        elif ext in {'.wav', '.mp3', '.m4a', '.aac', '.ogg', '.opus', '.wma', '.flac'}:
            media_type = 'audio'
        elif ext in {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}:
            media_type = 'video'
        else:
            log_warning(f"[MediaHash] Unknown media type for {file_path}, using file hash")
            file_hash = MediaHashComputer._compute_file_hash(file_path)
            return file_hash, file_hash
    
    # Route to appropriate hasher
    if media_type == 'image':
        return MediaHashComputer.compute_image_hash(file_path)
    elif media_type == 'audio':
        return MediaHashComputer.compute_audio_hash(file_path)
    elif media_type == 'video':
        return MediaHashComputer.compute_video_hash(file_path)
    else:
        file_hash = MediaHashComputer._compute_file_hash(file_path)
        return file_hash, file_hash
