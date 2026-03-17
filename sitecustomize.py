import warnings


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
