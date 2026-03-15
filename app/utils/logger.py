import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
import asyncio

# ----------------------------------
# LOG DIRECTORY SETUP
# ----------------------------------

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ----------------------------------
# LOG FORMATTER
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
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# ----------------------------------
# CONSOLE HANDLER (Colorized)
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
# PUBLIC LOG FUNCTIONS (with WS broadcast)
# ----------------------------------

def log_info(msg: str):
    logger.info(msg)
    _safe_async_run(_broadcast_log, "INFO", msg)

def log_warning(msg: str):
    logger.warning(msg)
    _safe_async_run(_broadcast_log, "WARNING", msg)

def log_error(msg: str):
    logger.error(msg)
    _safe_async_run(_broadcast_log, "ERROR", msg)

def log_debug(msg: str):
    logger.debug(msg)
    _safe_async_run(_broadcast_log, "DEBUG", msg)
