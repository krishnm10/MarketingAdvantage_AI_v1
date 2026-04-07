# =============================================
# text_cleaner_v2.py — Advanced Text Normalizer (Production-Ready)
# Enhanced and optimized for ingestion_v2 pipeline
# =============================================

import re
import unicodedata
from bs4 import BeautifulSoup
from typing import Optional

# -------------------------------------------------------------------
# HTML STRIPPING
# -------------------------------------------------------------------
def strip_html(text: Optional[str]) -> str:
    """Remove HTML tags safely using BeautifulSoup."""
    if not text:
        return ""
    try:
        return BeautifulSoup(text, "html.parser").get_text(separator=" ")
    except Exception:
        return text

# -------------------------------------------------------------------
# UNICODE NORMALIZATION
# -------------------------------------------------------------------
def normalize_unicode(text: str) -> str:
    """Normalize text to NFKC (safe canonical form)."""
    return unicodedata.normalize("NFKC", text)

# -------------------------------------------------------------------
# EMOJI + SYMBOL REMOVAL
# -------------------------------------------------------------------
def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002700-\U000027BF"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)

# -------------------------------------------------------------------
# URL + CONTROL CHARACTER REMOVAL
# -------------------------------------------------------------------
def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(r"https?://\S+|www\.\S+", "", text)

def remove_control_chars(text: str) -> str:
    """Remove invisible or control characters."""
    return re.sub(r"[\x00-\x1F\x7F]", " ", text)

# -------------------------------------------------------------------
# CID CHARACTER RESOLUTION (from PDF font encodings)
# -------------------------------------------------------------------
_CID_UNICODE_MAP = {
    133: "\u2026", 145: "\u2018", 146: "\u2019", 147: "\u201C",
    148: "\u201D", 149: "\u2022", 150: "\u2013", 151: "\u2014",
    160: "\u00A0", 169: "\u00A9", 174: "\u00AE", 176: "\u00B0",
    188: "\u00BC", 189: "\u00BD", 190: "\u00BE",
    210: "\u2013", 211: "\u2014", 212: "\u201C", 213: "\u201D",
}
_CID_PATTERN = re.compile(r"\(cid:(\d+)\)")


def resolve_cid_characters(text: str) -> str:
    """Replace (cid:NNN) placeholders with their Unicode equivalents."""
    if "(cid:" not in text:
        return text
    def _replace(match):
        cid = int(match.group(1))
        return _CID_UNICODE_MAP.get(cid, "\uFFFD")
    return _CID_PATTERN.sub(_replace, text)

# -------------------------------------------------------------------
# SPECIAL SYMBOL CLEANUP
# -------------------------------------------------------------------
def remove_special_symbols(text: str) -> str:
    """Remove stray bullets, arrows, and repeated punctuation."""
    text = re.sub(r"[•→←↔✔✖★☆●■□◆◇✓✗]+", " ", text)
    text = re.sub(r"([.,!?])\1{2,}", r"\1", text)  # limit punctuation repetition
    return text

# -------------------------------------------------------------------
# WHITESPACE NORMALIZATION
# -------------------------------------------------------------------
def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

# -------------------------------------------------------------------
# MASTER CLEAN FUNCTION
# -------------------------------------------------------------------
def clean_text(text: Optional[str]) -> str:
    """
    Advanced hybrid text cleaner.
    Steps:
      1. Strip HTML
      2. Normalize Unicode
      3. Remove emojis
      4. Remove URLs and control chars
      5. Remove special symbols
      6. Collapse whitespace
    """

    if not text or not isinstance(text, str):
        return ""

    text = strip_html(text)
    text = normalize_unicode(text)
    text = resolve_cid_characters(text)
    text = remove_emojis(text)
    text = remove_urls(text)
    text = remove_control_chars(text)
    text = remove_special_symbols(text)
    text = collapse_whitespace(text)

    return text