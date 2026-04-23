import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Any
import asyncio

# ----------------------------------
# LOG DIRECTORY SETUP
# ----------------------------------

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ----------------------------------
# LOG FORMAT MODE
# ----------------------------------
# Set LOG_FORMAT=json in production for structured JSON lines consumed by
# Datadog / Splunk / CloudWatch / any log aggregator.
# Default ("text") keeps the existing colorized console format unchanged.

_LOG_FORMAT_MODE = os.getenv("LOG_FORMAT", "text").lower()

# ----------------------------------
# STRUCTLOG JSON CONFIGURATION
# (activated when LOG_FORMAT=json)
# ----------------------------------

if _LOG_FORMAT_MODE == "json":
    import structlog

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.ExceptionRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

# ----------------------------------
# LOG FORMATTER  (text mode)
# ----------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

# ----------------------------------
# FILE HANDLER (Rotating)
# ----------------------------------

log_file_path = os.path.join(LOG_DIR, "app.log")

file_handler = RotatingFileHandler(
    log_file_path,
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5,
    encoding="utf-8"
)
# In JSON mode use plain formatter for file so each line is a JSON object
if _LOG_FORMAT_MODE == "json":
    file_handler.setFormatter(logging.Formatter("%(message)s"))
else:
    file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# ----------------------------------
# CONSOLE HANDLER (Colorized / JSON)
# ----------------------------------

class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[94m",
        "INFO": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "CRITICAL": "\033[95m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

console_handler = logging.StreamHandler()
if _LOG_FORMAT_MODE == "json":
    console_handler.setFormatter(logging.Formatter("%(message)s"))
else:
    console_handler.setFormatter(ColorFormatter(LOG_FORMAT, DATE_FORMAT))
console_handler.setLevel(logging.DEBUG)

# ----------------------------------
# ROOT LOGGER CONFIGURATION
# ----------------------------------

logger = logging.getLogger("MarketingAdvantageAI")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False

# ----------------------------------
# ASYNC WEBSOCKET BROADCAST SUPPORT
# ----------------------------------

try:
    from app.api.v2.ingestion_ws_api import broadcast
except Exception:
    broadcast = None  # Safe fallback if WS route not yet loaded

async def _broadcast_log(level: str, msg: str):
    """
    Sends logs to frontend WebSocket live feed (if available).
    """
    if broadcast is None:
        return
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "stage": "logger",
        "status": level.lower(),
        "message": msg,
    }
    try:
        await broadcast(payload)
    except Exception as e:
        logger.debug(f"Broadcast failed: {e}")

def _safe_async_run(async_fn, *args):
    """Run async function safely from sync context."""
    try:
        coro = async_fn(*args)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(coro)
        else:
            loop.run_until_complete(coro)
    except Exception as e:
        logger.debug(f"Logger async dispatch failed: {e}")
        try:
            if "coro" in locals():
                coro.close()
        except Exception:
            pass

# ----------------------------------
# INTERNAL FORMATTING HELPER
# ----------------------------------

def _format_log_message(msg: str, *args: Any, **kwargs: Any) -> str:
    """
    Safely format a log message that may use printf-style positional args or
    keyword args, or may already be a fully-formatted f-string (no extra args).

    Rules:
    - If neither args nor kwargs are provided the message is returned as-is.
    - If args are provided, attempt ``msg % args`` (printf-style).
    - If only kwargs are provided, attempt ``msg % kwargs``.
    - On any formatting error the raw message plus the arguments are returned
      as a fallback string; a direct logger.warning is emitted so the failure
      is visible without ever raising back to the caller.
    """
    if not args and not kwargs:
        return msg
    try:
        if args:
            return msg % args
        return msg % kwargs
    except Exception as fmt_exc:
        # Emit a direct warning — do NOT call log_warning() to avoid recursion.
        logger.warning(
            "[Logger] Message formatting failed (%s). "
            "raw_msg=%r args=%r kwargs=%r",
            fmt_exc, msg, args, kwargs,
        )
        extras = " | ".join(str(a) for a in args) if args else str(kwargs)
        return f"{msg} | {extras}"


# ----------------------------------
# PUBLIC LOG FUNCTIONS (with WS broadcast)
# ----------------------------------

def log_info(msg: str, *args: Any, **kwargs: Any) -> None:
    formatted = _format_log_message(msg, *args, **kwargs)
    logger.info(formatted)
    _safe_async_run(_broadcast_log, "INFO", formatted)

def log_warning(msg: str, *args: Any, **kwargs: Any) -> None:
    formatted = _format_log_message(msg, *args, **kwargs)
    logger.warning(formatted)
    _safe_async_run(_broadcast_log, "WARNING", formatted)

def log_error(msg: str, *args: Any, **kwargs: Any) -> None:
    formatted = _format_log_message(msg, *args, **kwargs)
    logger.error(formatted)
    _safe_async_run(_broadcast_log, "ERROR", formatted)

def log_debug(msg: str, *args: Any, **kwargs: Any) -> None:
    formatted = _format_log_message(msg, *args, **kwargs)
    logger.debug(formatted)
    _safe_async_run(_broadcast_log, "DEBUG", formatted)
