# =============================================
# visual_explainer_cpu_v1.py
#
# CPU-only Visual / Chart Explanation provider
#
# Characteristics:
# - CPU safe, no GPU required
# - Pre-cleans OCR noise before analysis
# - Extracts structured data: titles, year-value series, percentages
# - Detects data patterns (trends, comparisons, distributions)
# - Implements VisualExplainer contract
# =============================================

import re
from typing import List, Tuple, Optional, Dict

from app.ai.contracts import VisualExplainer
from app.utils.logger import log_info


# Lines matching these patterns are stripped before analysis
_NOISE_LINE_RE = re.compile(
    r"^\s*("
    r"graphic:|sauroe:|sauro\w*:|diu$|div$|div,|today|group"
    r"|e\s+\d|e\s+india"
    r"|\[extracted\s+data"
    r"|ncrore\)?|ncr\w*\)?"    # truncated "(in crore)" fragments
    r"|atl$|ail$|afl$"         # 3-char OCR ghosts
    r"|explanation\s*:?\s*$"   # empty prompt fragment
    r"|content\s*:\s*the\s+image"  # echoed LLM prompt
    r"|the\s+following\s+content\s+is"  # echoed LLM prompt
    r")",
    re.IGNORECASE,
)

# Metadata prefixes on color-region lines
_COLOR_REGION_RE = re.compile(r"^\[(\w+)\s+region\]\s*(.*)$", re.IGNORECASE)


class VisualExplainerCPU(VisualExplainer):
    """
    CPU-only visual explainer that converts raw chart/table/visual OCR text
    into human-readable semantic explanations using heuristic analysis.
    """

    # -------------------------------------------------
    # Contract method
    # -------------------------------------------------
    async def explain(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        # Step 1: Pre-clean the OCR text
        cleaned = self._pre_clean(text)
        if not cleaned:
            return ""

        # Step 2: Detect visual kind
        visual_kind = self._detect_visual_kind(cleaned)

        # Step 3: Choose explanation strategy
        if visual_kind == "table":
            return self._explain_table(cleaned)
        elif visual_kind in ("bar_chart", "line_chart"):
            return self._explain_chart(cleaned, visual_kind)
        elif visual_kind == "pie_chart":
            return self._explain_pie_chart(cleaned)
        elif visual_kind == "infographic":
            return self._explain_infographic(cleaned)
        else:
            return self._explain_general(cleaned)

    # =================================================
    # PRE-CLEANING
    # =================================================
    def _pre_clean(self, text: str) -> str:
        """
        Strip OCR noise, metadata lines, and color-region wrappers.
        Keeps only lines with meaningful content.
        """
        clean_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Skip known noise patterns
            if _NOISE_LINE_RE.search(stripped):
                continue
            # Skip copyright / watermark lines
            if re.search(r"[©®™]", stripped):
                continue
            # Skip lines starting with pipe/underscore sequences
            if re.match(r"^[\|_\-—–]{2,}", stripped):
                continue
            # Skip lines with high single-char token ratio (garbled OCR)
            tokens = stripped.split()
            if len(tokens) >= 4:
                single_chars = sum(1 for t in tokens if len(t.strip(".,;:!?")) <= 1)
                if single_chars / len(tokens) > 0.30:
                    continue
            # Unwrap [color region] prefix — keep only the value part
            m = _COLOR_REGION_RE.match(stripped)
            if m:
                value_part = m.group(2).strip()
                # Only keep if it contains a real number
                if re.search(r"\d+\.?\d*\s*%?", value_part):
                    clean_lines.append(value_part)
                continue
            # Skip lines that are mostly non-alphanumeric
            alnum = sum(c.isalnum() for c in stripped)
            if alnum < 3 or alnum / max(len(stripped.replace(" ", "")), 1) < 0.4:
                continue
            # Skip very short lines that aren't numbers (catches orphaned OCR fragments)
            if len(stripped) <= 6:
                alpha_count = sum(c.isalpha() for c in stripped)
                if alpha_count > 0 and alpha_count < 4 and not re.match(r"^\d[\d,.]*\s*%?$", stripped):
                    continue
            clean_lines.append(stripped)
        return "\n".join(clean_lines)

    # =================================================
    # VISUAL KIND DETECTION
    # =================================================
    def _detect_visual_kind(self, text: str) -> str:
        lower = text.lower()

        if re.search(r"\t.*\t", text) or re.search(r"\|.*\|", text):
            return "table"

        bar_signals = sum(1 for k in [
            "growth", "comparison", "population", "revenue",
            "sales", "crore", "million", "billion", "lakh",
            "census", "rate", "dropped", "projected",
        ] if k in lower)
        has_years = bool(re.search(r"\b(19|20)\d{2}\b.*\b(19|20)\d{2}\b", text))

        if bar_signals >= 2 or (bar_signals >= 1 and has_years):
            return "bar_chart"

        if any(k in lower for k in ["trend", "over time", "line chart", "forecast"]):
            return "line_chart"

        if "pie" in lower or (lower.count("%") >= 3 and not has_years):
            return "pie_chart"

        if any(k in lower for k in ["infographic", "visual summary", "key facts"]):
            return "infographic"

        return "general"

    # =================================================
    # CHART EXPLANATION (bar + line)
    # =================================================
    def _explain_chart(self, text: str, kind: str) -> str:
        parts: List[str] = []

        # --- Title extraction ---
        title = self._extract_title(text)
        if title:
            parts.append(f"Title: {title}")

        # --- Subtitle / description ---
        subtitle = self._extract_subtitle(text)
        if subtitle:
            parts.append(f"Description: {subtitle}")

        # --- Year-value data series ---
        years = self._extract_sorted_years(text)
        standalone_values = self._extract_standalone_values(text)
        year_value_pairs = self._match_years_to_values(years, standalone_values, text)

        if year_value_pairs:
            # Detect unit from context
            unit = self._detect_unit(text)
            parts.append(f"\nData series ({unit}):" if unit else "\nData series:")
            for year, value in year_value_pairs:
                parts.append(f"  {year}: {value}{' ' + unit if unit else ''}")

            # Trend analysis
            values = [v for _, v in year_value_pairs]
            if len(values) >= 2:
                if all(values[i] <= values[i + 1] for i in range(len(values) - 1)):
                    parts.append(f"\nTrend: Consistently increasing from {values[0]} to {values[-1]}.")
                elif all(values[i] >= values[i + 1] for i in range(len(values) - 1)):
                    parts.append(f"\nTrend: Consistently decreasing from {values[0]} to {values[-1]}.")
                else:
                    parts.append(f"\nTrend: Values range from {min(values)} to {max(values)}.")

                if values[0] > 0:
                    change = ((values[-1] - values[0]) / values[0]) * 100
                    parts.append(f"Overall change: {change:+.1f}% ({values[0]} → {values[-1]}).")

        # --- Percentages ---
        pcts = self._extract_percentages(text)
        if pcts:
            parts.append(f"\nPercentage values found: {', '.join(pcts)}")

        # --- Source ---
        source = self._extract_source(text)
        if source:
            parts.append(f"\nSource: {source}")

        if not parts:
            return self._explain_general(text)

        return "\n".join(parts)

    # =================================================
    # TABLE / PIE / INFOGRAPHIC / GENERAL
    # =================================================
    def _explain_table(self, text: str) -> str:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        numbers = self._extract_numbers(text)
        explanation = f"This table contains {len(lines)} rows of data."
        if numbers:
            explanation += f" Values range from {min(numbers)} to {max(numbers)}."
        return explanation

    def _explain_pie_chart(self, text: str) -> str:
        parts = ["This pie chart shows a proportional distribution."]
        pct_matches = re.findall(r"([A-Za-z\s]+?)[\s:]+(\d+\.?\d*)\s*%", text)
        if pct_matches:
            parts.append("Segments:")
            for label, pct in pct_matches:
                parts.append(f"  - {label.strip()}: {pct}%")
        else:
            pcts = re.findall(r"(\d+\.?\d*)\s*%", text)
            if pcts:
                parts.append(f"Percentage values: {', '.join(p + '%' for p in pcts)}")
        return "\n".join(parts)

    def _explain_infographic(self, text: str) -> str:
        parts = ["This infographic presents key data points and visual highlights."]
        title = self._extract_title(text)
        if title:
            parts.append(f"Title: {title}")
        subtitle = self._extract_subtitle(text)
        if subtitle:
            parts.append(f"Description: {subtitle}")

        years = self._extract_sorted_years(text)
        standalone_values = self._extract_standalone_values(text)
        year_value_pairs = self._match_years_to_values(years, standalone_values, text)
        if year_value_pairs:
            unit = self._detect_unit(text)
            parts.append("\nKey data points:")
            for year, value in year_value_pairs:
                parts.append(f"  {year}: {value}{' ' + unit if unit else ''}")
        else:
            numbers = self._extract_numbers(text)
            if numbers:
                parts.append(f"Numeric values: {', '.join(str(n) for n in numbers[:15])}")
        return "\n".join(parts)

    def _explain_general(self, text: str) -> str:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        numbers = self._extract_numbers(text)
        clean_lines = [l for l in lines if len(l) > 5]
        parts = []
        if clean_lines:
            summary = " | ".join(clean_lines[:5])
            if len(summary) > 300:
                summary = summary[:300] + "..."
            parts.append(f"Content: {summary}")
        if numbers:
            parts.append(f"Values: {', '.join(str(n) for n in numbers[:15])}")
        return "\n".join(parts) if parts else "Visual content detected but no structured data could be extracted."

    # =================================================
    # STRUCTURED DATA EXTRACTION
    # =================================================
    def _extract_title(self, text: str) -> str:
        """First line that looks like a title (mostly alpha, > 10 chars, no numbers-only)."""
        for line in text.splitlines()[:5]:
            line = line.strip()
            if len(line) < 8:
                continue
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in line) / max(len(line), 1)
            if alpha_ratio > 0.6 and not line.startswith("["):
                return line
        return ""

    def _extract_subtitle(self, text: str) -> str:
        """Find a descriptive sentence (contains verbs/prepositions, > 20 chars)."""
        desc_keywords = ["dropped", "increased", "decreased", "projected", "from", "to a", "grew", "fell", "rose"]
        for line in text.splitlines()[:10]:
            line = line.strip()
            if len(line) < 20:
                continue
            if any(k in line.lower() for k in desc_keywords):
                return line
        return ""

    def _extract_sorted_years(self, text: str) -> List[int]:
        """Find all unique 4-digit years and return sorted."""
        years = set()
        for m in re.finditer(r"\b(19\d{2}|20[0-2]\d)\b", text):
            years.add(int(m.group(1)))
        return sorted(years)

    def _extract_standalone_values(self, text: str) -> List[float]:
        """
        Find numbers that appear as standalone values (on their own line or
        separated from years). These are typically bar chart / Y-axis values.
        Filter out years and very small noise values.
        """
        values = []
        years_set = set(self._extract_sorted_years(text))

        # Standalone numbers on their own lines
        for line in text.splitlines():
            stripped = line.strip()
            # Match lines that are just a number (possibly with unit)
            m = re.match(r"^(\d[\d,]*\.?\d*)\s*(?:crore|million|billion|lakh|k|m|b)?$", stripped, re.IGNORECASE)
            if m:
                val = float(m.group(1).replace(",", ""))
                if val not in years_set and val > 1:
                    values.append(val)

        return values

    def _match_years_to_values(
        self,
        years: List[int],
        standalone_values: List[float],
        text: str,
    ) -> List[Tuple[int, float]]:
        """
        Match years to their corresponding values.

        Strategy:
        1. Look for year-value pairs on the same line
        2. If not enough pairs found, match sorted years to sorted standalone values
           (works for bar charts where bars go left-to-right, values top-to-bottom)
        """
        pairs: List[Tuple[int, float]] = []
        years_set = set(years)
        used_years = set()

        # Strategy 1: Same-line year→value pairs
        for line in text.splitlines():
            for m in re.finditer(
                r"\b(19\d{2}|20[0-2]\d)\b[^\d\n]{0,30}?(\d[\d,]*\.?\d*)\s*(%|crore|million|billion|lakh|k|m|b)?",
                line, re.IGNORECASE,
            ):
                year = int(m.group(1))
                val_str = m.group(2).replace(",", "")
                val = float(val_str)
                unit_suffix = (m.group(3) or "").strip()

                # Reject values that look like garbled years (5-digit starting with 19/20)
                if re.match(r"^(19|20)\d{3,}$", val_str):
                    continue
                # Reject percentage values — they are growth rates, NOT data points
                if unit_suffix == "%":
                    continue
                # Reject tiny values (<2) unless they have a unit (crore/million etc.)
                if val <= 1 and unit_suffix.lower() not in ("crore", "million", "billion", "lakh", "k", "m", "b"):
                    continue
                if year in years_set and val not in years_set and year not in used_years:
                    pairs.append((year, val))
                    used_years.add(year)

        # Strategy 2: If we have matching counts, align sorted years ↔ sorted values
        if len(pairs) < len(years) and standalone_values:
            # Remove values already used
            used_vals = {v for _, v in pairs}
            remaining_vals = sorted([v for v in standalone_values if v not in used_vals])
            remaining_years = sorted([y for y in years if y not in used_years])

            if remaining_years and remaining_vals:
                # If counts match or are close, align them
                if abs(len(remaining_years) - len(remaining_vals)) <= 2:
                    # Sort values ascending — matches years ascending for growth charts
                    for y, v in zip(remaining_years, remaining_vals):
                        pairs.append((y, v))

        # Sort by year
        pairs.sort(key=lambda x: x[0])
        return pairs

    def _extract_percentages(self, text: str) -> List[str]:
        """Extract percentage values like '2.2%', '1.2%'."""
        pcts = []
        seen = set()
        for m in re.finditer(r"(\d+\.?\d*)\s*%", text):
            val = m.group(1) + "%"
            if val not in seen:
                seen.add(val)
                pcts.append(val)
        return pcts

    def _extract_source(self, text: str) -> str:
        """Extract source attribution line."""
        for line in text.splitlines():
            if re.match(r"\s*source\s*:", line, re.IGNORECASE):
                return re.sub(r"^\s*source\s*:\s*", "", line, flags=re.IGNORECASE).strip()
        return ""

    def _detect_unit(self, text: str) -> str:
        """Detect the unit of measurement from context."""
        lower = text.lower()
        if "crore" in lower:
            return "crore"
        if "million" in lower:
            return "million"
        if "billion" in lower:
            return "billion"
        if "lakh" in lower:
            return "lakh"
        return ""

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numeric values from text."""
        raw = re.findall(r"\b(\d[\d,]*\.?\d*)\b", text)
        numbers = []
        for r in raw:
            try:
                val = float(r.replace(",", ""))
                numbers.append(val)
            except ValueError:
                pass
        return numbers
