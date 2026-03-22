import warnings
import locale

# ── Force UTF-8 as default file I/O encoding on Windows ──────────────────────
# Windows uses cp1252 by default which breaks reading UTF-8 files (e.g. .env).
# This makes behaviour consistent with Linux/Mac and with PYTHONUTF8=1 mode.
_orig_getpreferredencoding = locale.getpreferredencoding


def _utf8_preferred_encoding(do_setlocale: bool = True) -> str:
    enc = _orig_getpreferredencoding(do_setlocale)
    if enc.upper().replace("-", "") in ("CP1252", "ASCII", "LATIN1", "ISO88591"):
        return "UTF-8"
    return enc


locale.getpreferredencoding = _utf8_preferred_encoding
_SWIG_WARNING_PATTERNS = (
    r"builtin type SwigPyPacked has no __module__ attribute",
    r"builtin type SwigPyObject has no __module__ attribute",
    r"builtin type swigvarlink has no __module__ attribute",
)

for pattern in _SWIG_WARNING_PATTERNS:
    warnings.filterwarnings(
        "ignore",
        message=pattern,
        category=DeprecationWarning,
    )
