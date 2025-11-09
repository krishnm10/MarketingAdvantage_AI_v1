import json
import requests
import time
import re
import os
from typing import Any, Dict, Tuple

# Optional: only import if you plan to use semantic caching
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    _CACHE: Dict[str, Dict[str, Any]] = {}
except Exception:
    _encoder = None
    _CACHE = {}

# === Ollama Local API ===
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")

# === Models ===
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "gpt-oss:120b-cloud")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v3.1:671b-cloud")

# =====================================================
# 🔹 Text Cleaner
# =====================================================
def clean_text(text: str) -> str:
    """Cleans up spacing, token artifacts, and punctuation issues."""
    if not text:
        return text
    text = re.sub(r'(\b[a-zA-Z])\s(?=[a-zA-Z]\b)', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s+([,.:;!?])', r'\1', text)
    text = re.sub(r'\*\s+', '* ', text)
    text = re.sub(r'\s+\*', ' *', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# =====================================================
# 🔹 Semantic Cache Utilities
# =====================================================
def _get_cached_result(prompt: str, threshold: float = 0.93) -> Tuple[Dict[str, Any], bool]:
    """Return cached result if semantically similar prompt found."""
    if not _encoder or not _CACHE:
        return {}, False
    emb = _encoder.encode(prompt, convert_to_tensor=True)
    best_sim, best_key = 0.0, None
    for key, val in _CACHE.items():
        sim = util.cos_sim(emb, val["emb"]).item()
        if sim > best_sim:
            best_sim, best_key = sim, key
    if best_key and best_sim >= threshold:
        return _CACHE[best_key]["result"], True
    return {}, False

def _store_cache(prompt: str, result: Dict[str, Any]):
    if not _encoder:
        return
    emb = _encoder.encode(prompt, convert_to_tensor=True)
    _CACHE[prompt] = {"emb": emb, "result": result, "ts": time.time()}

# =====================================================
# 🔹 Streamed Generation Function
# =====================================================
def stream_generate(model: str, prompt: str, temperature: float = 0.7) -> str:
    """Streams LLM output in real time for faster generation."""
    payload = {"model": model, "prompt": prompt, "stream": True}
    # Some Ollama builds support temperature
    if temperature is not None:
        payload["temperature"] = temperature

    response = requests.post(
        OLLAMA_API_URL,
        json=payload,
        stream=True,
        timeout=360
    )
    output_text = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8").replace("data: ", ""))
                if "response" in data:
                    output_text += data["response"]
            except Exception:
                continue
    return clean_text(output_text)

# =====================================================
# 🔹 JSON Safe Parse
# =====================================================
def safe_json_parse(text: str) -> Dict[str, Any]:
    """Try to extract JSON block from text; fall back to raw text."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {"raw_text": text.strip()}

# =====================================================
# 🔹 Universal Dual-Model Chain
# =====================================================
def dual_chain_generate(
    prompt: str,
    context_type: str = "generic",
    temperature: float = 0.7,
    use_cache: bool = True,
    web_context: str = ""
):
    """
    2-Phase process:
      1️⃣ LLaMA creates creative draft
      2️⃣ DeepSeek polishes and optimizes the result
    Supports temperature, semantic cache, optional web_context prepend.
    """
    full_prompt = f"{web_context}\n\n{prompt}" if web_context else prompt

    # --- Check cache first ---
    if use_cache:
        cached, found = _get_cached_result(full_prompt)
        if found:
            cached["from_cache"] = True
            return cached

    start_time = time.time()
    results: Dict[str, Any] = {}
    models_used = {
        "phase_1": LLAMA_MODEL,
        "phase_2": DEEPSEEK_MODEL,
        "chain": "LLaMA → DeepSeek"
    }

    try:
        # --- Phase 1: Draft ---
        draft_output = stream_generate(LLAMA_MODEL, full_prompt, temperature)
        # --- Phase 2: Refinement ---
        refine_prompt = (
            f"Refine this {context_type} for clarity, engagement, and structure.\n"
            f"Output JSON if possible.\n\n{draft_output}"
        )
        refined_output = stream_generate(DEEPSEEK_MODEL, refine_prompt, temperature)

        results["draft_output"] = draft_output
        results["content_raw"] = refined_output
        results["content"] = safe_json_parse(refined_output)

    except Exception as e:
        results["content"] = {"error": f"⚠️ Error during generation: {e}"}

    results["execution_time_sec"] = round(time.time() - start_time, 2)
    results["models_used"] = models_used
    results["temperature"] = temperature
    results["from_cache"] = False

    if use_cache:
        _store_cache(full_prompt, results)

    return results
