# =============================================
# image_caption_cpu_v1.py
#
# CPU-only Image Caption + OCR provider
#
# Characteristics:
# - CPU safe
# - No GPU required
# - Lazy loading
# - Implements ImageCaptioner contract
# - Advanced pre-processing for charts/infographics
# - Multi-pass OCR (PSM 3, 6, 11) with merge
# - OpenCV contrast/threshold pipeline
# - Color region text extraction for bar charts
# =============================================

import asyncio
import re
from typing import Dict, Any, List

from app.ai.contracts import ImageCaptioner
from app.utils.logger import log_info


class ImageCaptionerCPU(ImageCaptioner):
    """
    CPU-only image captioner using advanced OCR pre-processing + heuristics.
    Handles plain photos, charts, infographics, bar charts with colored regions.
    """

    def __init__(self):
        self._ocr_ready = False
        self._cv2_ready = False

    # -------------------------------------------------
    # Lazy OCR + OpenCV init
    # -------------------------------------------------
    def _init_ocr(self):
        if self._ocr_ready:
            return

        try:
            import pytesseract  # noqa
            from PIL import Image  # noqa

            self._ocr_ready = True
            log_info("[ImageCaptionerCPU] OCR available")
        except Exception:
            self._ocr_ready = False
            log_info("[ImageCaptionerCPU] OCR unavailable, running caption-only mode")

        try:
            import cv2  # noqa
            import numpy  # noqa

            self._cv2_ready = True
            log_info("[ImageCaptionerCPU] OpenCV available — advanced pre-processing enabled")
        except Exception:
            self._cv2_ready = False
            log_info("[ImageCaptionerCPU] OpenCV unavailable — basic OCR only")

    # -------------------------------------------------
    # Caption image (main entry)
    # -------------------------------------------------
    async def caption(self, image_path: str) -> Dict[str, Any]:
        """
        Extract OCR text using advanced multi-pass pipeline and detect chart-like visuals.
        """
        self._init_ocr()

        loop = asyncio.get_running_loop()

        def _run():
            from PIL import Image
            import pytesseract

            with Image.open(image_path) as img:
                # Convert to RGB if needed (RGBA, P, L → RGB)
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                ocr_text = ""
                data_values: List[str] = []

                if self._ocr_ready:
                    if self._cv2_ready:
                        # Advanced pipeline: pre-process + multi-pass OCR
                        ocr_text, data_values = self._advanced_ocr(img, image_path)
                    else:
                        # Fallback: plain pytesseract
                        try:
                            ocr_text = pytesseract.image_to_string(img)
                        except Exception:
                            ocr_text = ""

                # Chart detection: expanded keyword list + structural signals
                lower = ocr_text.lower()
                chart_keywords = [
                    "%", "year", "202", "201", "200", "axis", "total",
                    "growth", "revenue", "crore", "million", "billion",
                    "population", "rate", "increase", "decrease",
                    "bar", "pie", "graph", "chart", "figure", "legend",
                    "source:", "avg", "average", "max", "min",
                ]
                keyword_hits = sum(1 for k in chart_keywords if k in lower)
                has_numbers = bool(re.search(r"\d+\.?\d*", ocr_text))

                is_chart = keyword_hits >= 2 or (keyword_hits >= 1 and has_numbers and len(data_values) > 0)

                # Richer caption based on detection
                if is_chart and data_values:
                    caption = (
                        "This image is a chart or data figure with extracted values: "
                        + ", ".join(data_values[:20])
                    )
                elif is_chart:
                    caption = "This image appears to be a chart or data figure."
                else:
                    caption = "This image appears to be a photograph or illustration."

                # Append extracted data values to OCR text if they added new info
                if data_values:
                    values_block = "\n[Extracted data values: " + ", ".join(data_values[:30]) + "]"
                    if values_block.strip() not in ocr_text:
                        ocr_text = ocr_text.strip() + "\n" + values_block

                return {
                    "caption": caption,
                    "ocr_text": ocr_text.strip(),
                    "objects": data_values if data_values else None,
                    "is_chart": is_chart,
                }

        return await loop.run_in_executor(None, _run)

    # =================================================
    # ADVANCED OCR PIPELINE
    # =================================================
    def _advanced_ocr(self, pil_img, image_path: str) -> tuple:
        """
        Multi-pass OCR with OpenCV pre-processing for charts/infographics.

        Returns:
            (merged_ocr_text, data_values_list)
        """
        import cv2
        import numpy as np
        import pytesseract
        from PIL import Image

        img_array = np.array(pil_img)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        all_texts: List[str] = []
        data_values: List[str] = []

        # ─── Pass 1: Original image, PSM 3 (auto) ────────
        try:
            text1 = pytesseract.image_to_string(pil_img, config="--psm 3")
            if text1.strip():
                all_texts.append(text1.strip())
        except Exception:
            pass

        # ─── Pass 2: Contrast-enhanced grayscale, PSM 6 ──
        try:
            # CLAHE (Adaptive histogram equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            enhanced_pil = Image.fromarray(enhanced)
            text2 = pytesseract.image_to_string(enhanced_pil, config="--psm 6")
            if text2.strip():
                all_texts.append(text2.strip())
        except Exception:
            pass

        # ─── Pass 3: Binary threshold (Otsu), PSM 6 ──────
        try:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_pil = Image.fromarray(binary)
            text3 = pytesseract.image_to_string(binary_pil, config="--psm 6")
            if text3.strip():
                all_texts.append(text3.strip())
        except Exception:
            pass

        # ─── Pass 4: Inverted binary, PSM 6 ──────────────
        # Catches light text on dark backgrounds
        try:
            inverted = cv2.bitwise_not(binary)
            inverted_pil = Image.fromarray(inverted)
            text4 = pytesseract.image_to_string(inverted_pil, config="--psm 6")
            if text4.strip():
                all_texts.append(text4.strip())
        except Exception:
            pass

        # ─── Pass 5: Upscaled 2x + sharpened, PSM 11 ────
        # Sparse text mode — catches isolated numbers in charts
        try:
            h, w = gray.shape[:2]
            upscaled = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(upscaled, -1, kernel)
            sharp_pil = Image.fromarray(sharpened)
            text5 = pytesseract.image_to_string(sharp_pil, config="--psm 11")
            if text5.strip():
                all_texts.append(text5.strip())
        except Exception:
            pass

        # ─── Pass 6: Color region extraction ─────────────
        # Isolate dominant color regions (bars in bar charts) and OCR each
        try:
            color_texts, color_values = self._extract_color_region_text(img_array)
            all_texts.extend(color_texts)
            data_values.extend(color_values)
        except Exception:
            pass

        # ─── Merge all passes: fuzzy-deduplicate + filter noise ───
        merged = self._merge_ocr_passes(all_texts)

        # ─── Extract numeric data values from merged text ─
        found_numbers = re.findall(r"\b\d+[\.,]?\d*\s*(?:%|crore|million|billion|lakh|k|m|b)?\b", merged, re.IGNORECASE)
        for n in found_numbers:
            n = n.strip()
            if n and n not in data_values:
                data_values.append(n)

        return merged, data_values

    # -------------------------------------------------
    # Color region text extraction
    # -------------------------------------------------
    def _extract_color_region_text(self, img_array) -> tuple:
        """
        Segment image by dominant colors (HSV clustering) and OCR each region.
        This catches text ON colored bars that standard OCR misses.
        """
        import cv2
        import numpy as np
        import pytesseract
        from PIL import Image

        texts: List[str] = []
        values: List[str] = []

        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        h, w = img_array.shape[:2]

        # Define color ranges for common chart bar colors
        color_ranges = [
            ("red",    (np.array([0, 70, 50]),   np.array([10, 255, 255]))),
            ("red2",   (np.array([170, 70, 50]), np.array([180, 255, 255]))),
            ("blue",   (np.array([100, 70, 50]), np.array([130, 255, 255]))),
            ("green",  (np.array([35, 70, 50]),  np.array([85, 255, 255]))),
            ("yellow", (np.array([20, 70, 50]),  np.array([35, 255, 255]))),
            ("orange", (np.array([10, 70, 50]),  np.array([20, 255, 255]))),
            ("purple", (np.array([130, 70, 50]), np.array([170, 255, 255]))),
        ]

        for color_name, (lower, upper) in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)

            # Only process if this color covers a meaningful area (> 2% of image)
            coverage = cv2.countNonZero(mask) / (h * w)
            if coverage < 0.02 or coverage > 0.80:
                continue

            # Find contours of colored regions
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, cw, ch = cv2.boundingRect(contour)
                # Skip tiny regions
                if cw < 20 or ch < 15:
                    continue

                # Expand ROI slightly to catch adjacent labels
                pad = 10
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w, x + cw + pad)
                y2 = min(h, y + ch + pad)

                roi = img_array[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # Convert to grayscale and threshold for OCR
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                _, roi_bin = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Also try inverted
                roi_inv = cv2.bitwise_not(roi_bin)

                best_text = ""
                for variant in [roi_bin, roi_inv]:
                    try:
                        roi_pil = Image.fromarray(variant)
                        text = pytesseract.image_to_string(roi_pil, config="--psm 7").strip()
                        if text and len(text) > len(best_text):
                            best_text = text
                    except Exception:
                        pass

                # Quality gate: only keep color region text that has
                # meaningful content (numbers or 3+ alpha chars)
                if best_text and self._is_meaningful_color_text(best_text):
                    texts.append(f"[{color_name} region] {best_text}")
                    nums = re.findall(r"\b\d+[\.,]?\d*\b", best_text)
                    for n in nums:
                        if n not in values:
                            values.append(n)

        return texts, values

    @staticmethod
    def _is_meaningful_color_text(text: str) -> bool:
        """
        Returns True only if the OCR text from a color region contains
        actual data (numbers with optional units, or 3+ consecutive alpha chars).
        Filters out garbage like 'BB', 'r |', 'YY]', '(20)', 'p25)'.
        """
        cleaned = re.sub(r"[^a-zA-Z0-9%.,:]", " ", text).strip()
        if not cleaned:
            return False
        # Must contain at least one number with optional unit
        has_number = bool(re.search(r"\b\d+\.?\d*\s*(%|crore|million|billion|lakh|k|m|b)?\b", cleaned, re.IGNORECASE))
        # Or contains a meaningful word (3+ alpha chars)
        has_word = bool(re.search(r"[a-zA-Z]{3,}", cleaned))
        if not has_number and not has_word:
            return False
        # Reject if mostly non-alphanumeric in original
        alnum = sum(c.isalnum() for c in text)
        if alnum / max(len(text), 1) < 0.5:
            return False
        return True

    # -------------------------------------------------
    # Merge OCR passes (deduplicate lines)
    # -------------------------------------------------
    def _merge_ocr_passes(self, text_list: List[str]) -> str:
        """
        Merge text from multiple OCR passes with:
        1. Noise line filtering (single chars, symbols, short garbage)
        2. Fuzzy deduplication (catches OCR typo variants of the same line)
        3. Structural dedup (catches lines with same word structure but different numbers)
        """
        seen_keys: List[str] = []       # normalized keys for fuzzy matching
        seen_structs: Dict[str, str] = {}  # structural signature → best line
        seen_prefixes: Dict[str, str] = {}  # label prefix → best line
        merged_lines: List[str] = []    # actual output lines

        for text in text_list:
            for line in text.splitlines():
                normalized = line.strip()
                if not normalized:
                    continue

                # ── Noise filter: skip garbage lines ──────────
                if self._is_noise_line(normalized):
                    continue

                # ── Fuzzy dedup: skip if similar line already seen ─
                key = re.sub(r"\s+", " ", normalized.lower())
                if self._is_fuzzy_duplicate(key, seen_keys):
                    continue

                # ── Structural dedup: lines that share the same word
                #    skeleton but differ only in OCR'd numbers ──
                struct_key = self._structural_signature(normalized)
                if struct_key and struct_key in seen_structs:
                    # Keep the variant with higher alphanumeric ratio (cleaner OCR)
                    existing = seen_structs[struct_key]
                    if self._alnum_ratio(normalized) > self._alnum_ratio(existing):
                        # Replace the old line with the better variant
                        idx = merged_lines.index(existing)
                        merged_lines[idx] = normalized
                        seen_structs[struct_key] = normalized
                    continue

                # ── Prefix dedup: lines starting with the same label
                #    (e.g. "Source:", "Growth Rate:") are the same item ──
                prefix_key = self._label_prefix(normalized)
                if prefix_key and prefix_key in seen_prefixes:
                    existing = seen_prefixes[prefix_key]
                    if self._alnum_ratio(normalized) > self._alnum_ratio(existing):
                        idx = merged_lines.index(existing)
                        merged_lines[idx] = normalized
                        seen_prefixes[prefix_key] = normalized
                    continue

                seen_keys.append(key)
                if struct_key:
                    seen_structs[struct_key] = normalized
                if prefix_key:
                    seen_prefixes[prefix_key] = normalized
                merged_lines.append(normalized)

        return "\n".join(merged_lines)

    @staticmethod
    def _structural_signature(line: str) -> str:
        """
        Build a 'structural signature' by abstracting word lengths and numbers.
        Two lines with the same signature are structurally the same even if OCR
        garbled words differently.

        'Growth Pate: 21.5% 248% 23.9%' → 'W6 W4: N% N% N%'
        'Growth Rate: 215% 248% 23.9%'  → 'W6 W4: N% N% N%'
        'Source: Censusof india' → 'W6: W8 W5'
        'Source: Gensusat India' → 'W6: W8 W5'  (same!)

        Returns empty string for lines that are purely numeric or too short.
        """
        # Must have at least 2 alphabetic words to be worth comparing
        alpha_words = re.findall(r"[a-zA-Z]{2,}", line)
        if len(alpha_words) < 2:
            return ""

        # Strip trailing dashes/punctuation that OCR sometimes adds
        line = line.rstrip(" —-–")

        # Build pattern: words→W+len, numbers→N, keep punctuation
        pattern_parts = []
        for token in re.split(r"(\s+)", line):
            token_stripped = token.strip()
            if not token_stripped:
                continue
            # Number (possibly with decimals, commas)
            if re.match(r"^\d[\d,.]*%?$", token_stripped):
                pattern_parts.append("N%" if token_stripped.endswith("%") else "N")
            # Word (possibly with trailing punctuation like ':')
            elif re.match(r"^[a-zA-Z]", token_stripped):
                alpha_len = sum(c.isalpha() for c in token_stripped)
                trailing = token_stripped[len(token_stripped.rstrip(":,;.—-")):]
                pattern_parts.append(f"W{alpha_len}{trailing}")
            else:
                pattern_parts.append(token_stripped)

        sig = " ".join(pattern_parts)
        return sig if len(pattern_parts) >= 3 else ""

    @staticmethod
    def _alnum_ratio(line: str) -> float:
        """Ratio of alphanumeric characters in a line."""
        total = len(line.replace(" ", ""))
        if total == 0:
            return 0.0
        return sum(c.isalnum() for c in line) / total

    @staticmethod
    def _label_prefix(line: str) -> str:
        """
        Extract a normalised label prefix from lines like 'Source: ...',
        'Growth Rate: ...', 'Note: ...'. Returns empty string if no label found.
        """
        m = re.match(r"^([a-zA-Z][a-zA-Z\s]{2,20})\s*:\s*\S", line)
        if m:
            return m.group(1).strip().lower()
        return ""

    @staticmethod
    def _is_noise_line(line: str) -> bool:
        """
        Returns True if the line is OCR noise rather than real content.
        Catches: single chars, symbol-only lines, short non-alphanumeric junk,
        truncated/orphaned words, garbled OCR, copyright lines, pipe artifacts.
        """
        # Strip all non-alphanumeric to measure real content
        alnum_only = re.sub(r"[^a-zA-Z0-9]", "", line)

        # Too short to be meaningful (< 3 alphanumeric chars)
        if len(alnum_only) < 3:
            return True

        # Line is mostly symbols/punctuation (< 50% alphanumeric)
        stripped = line.replace(" ", "")
        if stripped and len(alnum_only) / len(stripped) < 0.50:
            return True

        # Copyright / watermark / logo lines
        if re.search(r"[©®™]", line):
            return True

        # Lines starting with pipe/underscore sequences (OCR of chart axes/borders)
        if re.match(r"^[\|_\-—–]{2,}", line.strip()):
            return True

        # Garbled OCR: >30% of tokens are single characters (e.g. "e , e INDIA")
        tokens = line.split()
        if len(tokens) >= 4:
            single_char_tokens = sum(1 for t in tokens if len(t.strip(".,;:!?")) <= 1)
            if single_char_tokens / len(tokens) > 0.30:
                return True

        # Mismatched brackets — strong signal of OCR garbage (e.g. "(P}", "(%]")
        opens = line.count("(") + line.count("[") + line.count("{")
        closes_matching = line.count(")") + line.count("]") + line.count("}")
        if opens > 0 and closes_matching > 0:
            # Check for cross-bracket mismatches: ( with } or ] , [ with ) or }
            has_mismatch = (
                ("(" in line and ("}" in line or "]" in line and "[" not in line))
                or ("[" in line and (")" in line and "(" not in line or "}" in line))
            )
            if has_mismatch:
                return True

        # Orphaned fragments: short line that is NOT a number and NOT
        # a recognizable word pattern (catches garbled OCR like "atl", "ncrore)")
        if len(line) <= 8:
            # Allow pure numbers or numbers with units
            if re.match(r"^\d[\d,.]*\s*(%|crore|million|billion|lakh|k|m|b)?$", line.strip(), re.IGNORECASE):
                return False
            # Allow lines that are at least 4 alpha chars (likely real short words)
            alpha_count = sum(c.isalpha() for c in line)
            if alpha_count < 4:
                return True

        # Truncated parenthetical fragments: e.g. "ncrore)", "ncr)"
        if re.match(r"^[a-z]+\)$", line) and len(line) <= 10:
            return True

        return False

    @staticmethod
    def _is_fuzzy_duplicate(key: str, seen_keys: List[str], threshold: float = 0.65) -> bool:
        """
        Check if `key` is a fuzzy duplicate of any already-seen key.
        Uses token-level Jaccard similarity to catch OCR variants like:
          'growth pate: 21.5% 248%' vs 'growth rate: 21.5% 24.8%'
        """
        if not seen_keys:
            return False

        tokens_new = set(key.split())
        if not tokens_new:
            return False

        for existing in seen_keys:
            tokens_old = set(existing.split())
            if not tokens_old:
                continue
            intersection = tokens_new & tokens_old
            union = tokens_new | tokens_old
            jaccard = len(intersection) / len(union)
            if jaccard >= threshold:
                return True

        return False
