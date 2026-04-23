"""
Regression tests for app/utils/logger.py public wrappers.

Covers:
  1. f-string (single-arg) safety – existing call pattern must still work.
  2. printf-style (multi-arg) safety – log_*(msg, *args) must format correctly.
  3. Bad format specifier – must never raise; must emit a logger.warning fallback.
  4. All four wrappers (log_info, log_warning, log_error, log_debug) behave the same.
  5. WebSocket broadcast path is not called when broadcast is None (no side-effects
     in unit tests).
"""
import logging
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def captured_logger(monkeypatch):
    """
    Replace the underlying 'logger' object inside logger.py with a MagicMock
    so we can assert on calls without touching handlers or files.
    """
    import app.utils.logger as log_module

    mock_log = MagicMock(spec=logging.Logger)
    monkeypatch.setattr(log_module, "logger", mock_log)
    # Also stub out _safe_async_run so WS broadcast is never attempted.
    monkeypatch.setattr(log_module, "_safe_async_run", MagicMock())
    return mock_log, log_module


# ---------------------------------------------------------------------------
# _format_log_message unit tests
# ---------------------------------------------------------------------------

class TestFormatLogMessage:
    """Unit tests for the internal helper (not exposed publicly but testable)."""

    def test_no_args_returns_message_unchanged(self, captured_logger):
        _mock, mod = captured_logger
        assert mod._format_log_message("Hello world") == "Hello world"

    def test_fstring_no_args_returns_message_unchanged(self, captured_logger):
        _mock, mod = captured_logger
        msg = f"File {'abc.pdf'} processed"
        assert mod._format_log_message(msg) == msg

    def test_positional_args_formatted(self, captured_logger):
        _mock, mod = captured_logger
        result = mod._format_log_message("model=%s max_tokens=%d", "nomic", 512)
        assert result == "model=nomic max_tokens=512"

    def test_single_positional_arg(self, captured_logger):
        _mock, mod = captured_logger
        result = mod._format_log_message("Error: %s", "timeout")
        assert result == "Error: timeout"

    def test_kwargs_formatted(self, captured_logger):
        _mock, mod = captured_logger
        result = mod._format_log_message("model=%(m)s tokens=%(t)d", m="ada", t=256)
        assert result == "model=ada tokens=256"

    def test_bad_format_specifier_does_not_raise(self, captured_logger):
        mock_log, mod = captured_logger
        # %d expects int but receives a string — must not raise.
        result = mod._format_log_message("count=%d", "not_an_int")
        # Result must be a string (fallback concatenation).
        assert isinstance(result, str)
        assert "count=%d" in result
        assert "not_an_int" in result
        # A direct logger.warning must have been emitted.
        mock_log.warning.assert_called_once()
        warning_msg = str(mock_log.warning.call_args)
        assert "formatting failed" in warning_msg.lower() or "Message formatting failed" in warning_msg


# ---------------------------------------------------------------------------
# Public wrapper tests — log_info
# ---------------------------------------------------------------------------

class TestLogInfo:
    def test_fstring_single_arg(self, captured_logger):
        mock_log, mod = captured_logger
        mod.log_info(f"[IngestionV2] Parsed file {'abc.pdf'}")
        mock_log.info.assert_called_once_with("[IngestionV2] Parsed file abc.pdf")

    def test_printf_style_positional(self, captured_logger):
        mock_log, mod = captured_logger
        mod.log_info("[IngestionV2] model=%s max_tokens=%d", "nomic-embed-text", 512)
        mock_log.info.assert_called_once_with(
            "[IngestionV2] model=nomic-embed-text max_tokens=512"
        )

    def test_printf_style_single_exception_arg(self, captured_logger):
        mock_log, mod = captured_logger
        exc = RuntimeError("disk full")
        mod.log_info("[IngestionV2] Phase 1 token_aware path failed (%s); fallback.", exc)
        expected = "[IngestionV2] Phase 1 token_aware path failed (disk full); fallback."
        mock_log.info.assert_called_once_with(expected)

    def test_bad_format_does_not_raise(self, captured_logger):
        mock_log, mod = captured_logger
        # Must not raise even with a mismatched format specifier.
        mod.log_info("Bad format: %d", "string_not_int")
        mock_log.info.assert_called_once()
        # logger.warning should have fired for the formatting failure.
        mock_log.warning.assert_called_once()

    def test_ws_broadcast_scheduled(self, captured_logger):
        mock_log, mod = captured_logger
        mod.log_info("simple message")
        mod._safe_async_run.assert_called_once()
        # First positional arg after the coroutine fn is the level string.
        level_arg = mod._safe_async_run.call_args[0][1]
        assert level_arg == "INFO"


# ---------------------------------------------------------------------------
# Public wrapper tests — log_warning
# ---------------------------------------------------------------------------

class TestLogWarning:
    def test_fstring_single_arg(self, captured_logger):
        mock_log, mod = captured_logger
        mod.log_warning(f"[Warn] chunk {'x'} skipped")
        mock_log.warning.assert_called_once_with("[Warn] chunk x skipped")

    def test_printf_positional(self, captured_logger):
        mock_log, mod = captured_logger
        mod.log_warning("[Warn] worker=%s failed", "worker-3")
        mock_log.warning.assert_called_once_with("[Warn] worker=worker-3 failed")

    def test_bad_format_does_not_raise(self, captured_logger):
        mock_log, mod = captured_logger
        mod.log_warning("count=%d", "oops")
        mock_log.warning.assert_called()  # at least one call (fallback + the original msg)


# ---------------------------------------------------------------------------
# Public wrapper tests — log_error
# ---------------------------------------------------------------------------

class TestLogError:
    def test_fstring_single_arg(self, captured_logger):
        mock_log, mod = captured_logger
        mod.log_error("[Error] critical failure")
        mock_log.error.assert_called_once_with("[Error] critical failure")

    def test_printf_positional(self, captured_logger):
        mock_log, mod = captured_logger
        mod.log_error("[Error] stage=%s error=%s", "embed", "timeout")
        mock_log.error.assert_called_once_with("[Error] stage=embed error=timeout")

    def test_bad_format_does_not_raise(self, captured_logger):
        mock_log, mod = captured_logger
        mod.log_error("value=%d", "bad")
        mock_log.error.assert_called_once()


# ---------------------------------------------------------------------------
# Public wrapper tests — log_debug
# ---------------------------------------------------------------------------

class TestLogDebug:
    def test_fstring_single_arg(self, captured_logger):
        mock_log, mod = captured_logger
        mod.log_debug("[Debug] step=1")
        mock_log.debug.assert_called_once_with("[Debug] step=1")

    def test_printf_positional(self, captured_logger):
        mock_log, mod = captured_logger
        mod.log_debug("[Debug] key=%s value=%d", "top_k", 10)
        mock_log.debug.assert_called_once_with("[Debug] key=top_k value=10")

    def test_bad_format_does_not_raise(self, captured_logger):
        mock_log, mod = captured_logger
        mod.log_debug("score=%f", "nan_string")
        mock_log.debug.assert_called_once()
