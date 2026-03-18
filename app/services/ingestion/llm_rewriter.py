# =============================================
# llm_rewriter.py — Factual/Creative Rewriter (Production-Grade)
# Async-safe queue + retry + adaptive backpressure control
# PATCH C: fixed unsafe Semaphore mutation, loop handling, and worker lifecycle
# PATCH D: HARD LLM OFF GATE (NO OLLAMA WHEN DISABLED)
# =============================================

import aiohttp
import asyncio
import hashlib
import statistics
import time
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from app.utils.logger import log_info, log_warning
from app.config import ingestion_settings

# -------------------------------------------------------------------
# CONFIGURATION (kept as in your original file)
# -------------------------------------------------------------------
OLLAMA_URL = ingestion_settings.OLLAMA_API_URL
MODEL_NAME = ingestion_settings.OLLAMA_MODEL
CACHE_ENABLED = True
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
BATCH_SIZE = 8
TIMEOUT = 180  # Increased from 60 → 180 for long content (e.g. web scraping)
MAX_CONCURRENCY = 2 if MODEL_NAME.startswith("llama3.1:8b") else 4
LATENCY_WINDOW = 20
BACKPRESSURE_THRESHOLD = 5.0
MIN_BACKOFF = 1.0
MAX_BACKOFF = 10.0

# -------------------------------------------------------------------
# PROMPT FACTORY — Dual Mode (Factual | Creative)
# (unchanged)
# -------------------------------------------------------------------
def create_normalization_prompt(text_input: str) -> str:
    mode = getattr(ingestion_settings, "LLM_MODE", "factual")

    if mode == "creative":
        prompt = f"""
        You are a creative yet accurate language rewriter.
        Rewrite the given text in a more natural, engaging, and fluent style
        while preserving all factual details.
        - Do not add new information or hallucinate.
        - Improve readability, coherence, and tone only.
        - Ensure facts, numbers, and names remain unchanged.
        Input: {text_input}
        Output:
        """
    else:
        prompt = f"""
        You are a factual text normalizer.
        Rewrite the text so that it is clear, concise, and human-readable.
        - Keep all data, numbers, names, and factual details exactly as is.
        - Do not summarize, infer, or add any new information.
        - Focus on grammatical and stylistic cleanup only.
        Input: {text_input}
        Output:
        """
    return " ".join(prompt.split())

# -------------------------------------------------------------------
# CACHE
# -------------------------------------------------------------------
_MAX_CACHE_SIZE: int = 4096  # bound in-memory cache to prevent unbounded growth
_cache: Dict[str, str] = {}
_cache_lock = asyncio.Lock()

def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_put(key: str, value: str) -> None:
    """Insert into the bounded cache, evicting oldest entries when full."""
    if len(_cache) >= _MAX_CACHE_SIZE:
        # Evict ~25% oldest entries to amortise eviction cost
        evict_count = _MAX_CACHE_SIZE // 4
        keys_to_evict = list(_cache.keys())[:evict_count]
        for k in keys_to_evict:
            _cache.pop(k, None)
    _cache[key] = value

# -------------------------------------------------------------------
# LATENCY TRACKER
# -------------------------------------------------------------------
class LatencyMonitor:
    def __init__(self, window: int = LATENCY_WINDOW):
        self.samples: List[float] = []
        self.window = window

    def record(self, value: float):
        self.samples.append(value)
        if len(self.samples) > self.window:
            self.samples.pop(0)

    def avg(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0.0

latency_monitor = LatencyMonitor()

# -------------------------------------------------------------------
# QUEUE SYSTEM (Patched)
# -------------------------------------------------------------------
class LLMRewriteQueue:
    """Async-safe rewrite queue with retry + backpressure."""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.active = False
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        self._worker_task: Optional[asyncio.Task] = None
        self.current_backoff = MIN_BACKOFF
        self._max_concurrency = MAX_CONCURRENCY
        self._http_client: Optional[aiohttp.ClientSession] = None

    async def start(self):
        """Start background worker (idempotent)."""
        if self.active:
            return
        self.active = True
        loop = asyncio.get_running_loop()
        self._worker_task = loop.create_task(self._worker_loop())
        log_info("[llm_rewriter] Rewrite queue started with backpressure control.")

    async def stop(self):
        """Graceful stop of the worker."""
        self.active = False
        if self._worker_task:
            await self._worker_task
            self._worker_task = None
        if self._http_client:
            try:
                await self._http_client.close()
            except Exception:
                pass
            self._http_client = None
        log_info("[llm_rewriter] Rewrite queue stopped.")

    async def enqueue(self, text: str, future: asyncio.Future):
        """Enqueue a (text, future) pair for processing."""
        await self.queue.put((text, future))

    async def _apply_backpressure(self):
        avg_latency = latency_monitor.avg()
        qsize = self.queue.qsize()
        if avg_latency > BACKPRESSURE_THRESHOLD:
            self.current_backoff = min(self.current_backoff * 1.5, MAX_BACKOFF)
            log_warning(
                f"[llm_rewriter] ⚠️ Backpressure: latency={avg_latency:.2f}s, "
                f"queue={qsize}, backoff={self.current_backoff:.1f}s"
            )
        else:
            if self.current_backoff > MIN_BACKOFF:
                self.current_backoff = max(MIN_BACKOFF, self.current_backoff * 0.9)
        await asyncio.sleep(self.current_backoff)

    async def _worker_loop(self):
        self._http_client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=TIMEOUT)
        )
        try:
            while self.active:
                try:
                    text, future = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                await self._semaphore.acquire()
                asyncio.create_task(self._process_item(text, future))
        finally:
            if self._http_client and not self._http_client.closed:
                await self._http_client.close()
            self._http_client = None
            self.active = False

    async def _process_item(self, text: str, future: asyncio.Future):
        start_time = time.time()
        try:
            result = await _attempt_rewrite_with_retry(
                text, http_client=self._http_client
            )
            if not future.done():
                future.set_result(result)
        except Exception as e:
            log_warning(f"[llm_rewriter] Failed to process: {e}")
            if not future.done():
                future.set_result(text)
        finally:
            latency_monitor.record(time.time() - start_time)
            self._semaphore.release()
            try:
                await self._apply_backpressure()
            except Exception:
                pass
            try:
                self.queue.task_done()
            except Exception:
                pass

rewrite_queue = LLMRewriteQueue()

# -------------------------------------------------------------------
# REWRITE CORE
# -------------------------------------------------------------------
async def _attempt_rewrite_with_retry(
    text: str, http_client: Optional[aiohttp.ClientSession] = None
) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            rewritten = await _rewrite_call(text, http_client=http_client)
            if rewritten:
                return rewritten
        except Exception as e:
            log_warning(f"[llm_rewriter] Attempt {attempt} failed: {e}")
        await asyncio.sleep(RETRY_DELAY_BASE ** attempt)
    log_warning(
        f"[llm_rewriter] ⚠️ All attempts failed, returning original text (len={len(text)})."
    )
    return text

async def _rewrite_call(
    text: str, http_client: Optional[aiohttp.ClientSession] = None
) -> str:
    key = _cache_key(text)
    async with _cache_lock:
        if CACHE_ENABLED and key in _cache:
            return _cache[key]

    prompt = create_normalization_prompt(text)
    log_info(
        f"[llm_rewriter] Sending prompt to Ollama: model={MODEL_NAME}, len={len(prompt)} chars"
    )

    close_client = False
    client = http_client
    if client is None:
        client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=TIMEOUT)
        )
        close_client = True

    try:
        payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
        async with client.post(OLLAMA_URL, json=payload) as response:
            if response.status != 200:
                raise RuntimeError(f"Ollama returned {response.status}")
            result = await response.json()

            output = (
                result.get("response")
                or result.get("text")
                or result.get("generated")
                or ""
            ).strip()

            if not output:
                raise RuntimeError("Empty Ollama response")

            async with _cache_lock:
                if CACHE_ENABLED:
                    _cache_put(key, output)
            return output
    finally:
        if close_client:
            await client.close()

# -------------------------------------------------------------------
# PUBLIC FUNCTIONS (API-COMPATIBLE)
# -------------------------------------------------------------------
async def rewrite_text(text: str) -> str:
    """Rewrite a single text entry asynchronously."""

    # 🔒 HARD LLM OFF GATE (ADDED)
    if not ingestion_settings.ENABLE_LLM_NORMALIZATION:
        return text

    if not text or not text.strip():
        return text

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    await rewrite_queue.enqueue(text, future)
    await rewrite_queue.start()
    return await future

async def rewrite_batch(texts: List[str]) -> List[str]:
    """Rewrite multiple texts concurrently with batching."""

    # 🔒 HARD LLM OFF GATE (ADDED)
    if not ingestion_settings.ENABLE_LLM_NORMALIZATION:
        return texts

    if not texts:
        return []

    await rewrite_queue.start()
    results: List[str] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        rewritten = await asyncio.gather(*(rewrite_text(t) for t in batch))
        results.extend(rewritten)
        await asyncio.sleep(0)
    return results

# -------------------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------------------
async def is_llm_ready() -> bool:
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            payload = {"model": MODEL_NAME, "prompt": "ping", "stream": False}
            async with session.post(OLLAMA_URL, json=payload) as response:
                return response.status == 200
    except Exception:
        return False
