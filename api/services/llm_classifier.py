# ==========================================================
# 🚀 llm_classifier.py — Enterprise LLM-based Taxonomy Classifier (v3.2)
# ==========================================================
"""
Uses DeepSeek (via Ollama) to classify text rows into structured JSON
based on enterprise_ingest_prompt_v3_optimized.json stored in /api/services/prompts/.

✅ Features:
  - Loads model + prompt dynamically
  - Enforces JSON-only output with flexible recovery
  - Retries on malformed output
  - Logs reasoning traces for audit
  - Always returns valid normalized dict

Optimized for: DeepSeek-R1, DeepSeek-R1:1.5b, and similar Ollama-hosted models
"""

import os
import json
import re
import logging
import requests
from datetime import datetime

# ==========================================================
# ⚙️ CONFIGURATION
# ==========================================================
PROMPT_PATH = os.path.join(
    os.getcwd(), "api", "services", "prompts", "enterprise_ingest_prompt_v3_optimized.json"
)
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
TRACE_LOG_PATH = os.path.join(LOG_DIR, "llm_trace.log")

OLLAMA_API = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

logging.basicConfig(
    filename=TRACE_LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("llm_classifier")


# ==========================================================
# 🧠 LOAD PROMPT
# ==========================================================
def load_prompt():
    """Load enterprise prompt for DeepSeek."""
    if not os.path.exists(PROMPT_PATH):
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return (
        data.get("content"),
        data.get("model_name", "llama3.1:8b"),
        data.get("version", "3.1"),
    )


PROMPT_TEXT, MODEL_NAME, PROMPT_VERSION = load_prompt()
logger.info(f"[Init] Loaded prompt version {PROMPT_VERSION} with model {MODEL_NAME}")


# ==========================================================
# 🧩 JSON SANITIZATION / RECOVERY
# ==========================================================
def parse_llm_json_output(raw_output: str) -> dict:
    """
    Parses JSON output robustly, even if DeepSeek adds non-JSON content.
    If parsing fails, extract best JSON substring or return default.
    """
    try:
        # Try direct JSON first
        return json.loads(raw_output)
    except Exception:
        # Try to extract JSON from mixed response
        match = re.search(r"\{[\s\S]*\}", raw_output)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

    # If still invalid, return fallback
    logger.warning(f"[ParseFallback] Invalid JSON. Raw snippet: {raw_output[:150]}")
    return {
        "entity_type": "content",
        "category_level_1": "Uncategorized",
        "category_level_2_sub": "General Business",
        "business_concept_name": None,
        "business_specific_name": None,
        "primary_process_type": "Other",
        "title": "Unclassified Entry",
        "description": raw_output[:250],
        "extraction_confidence": 0.4,
    }


def normalize_json_schema(data: dict) -> dict:
    """Ensures all required keys exist and cleans internal reasoning."""
    required_keys = [
        "entity_type", "category_level_1", "category_level_2_sub",
        "business_concept_name", "business_specific_name",
        "primary_process_type", "title", "description", "extraction_confidence"
    ]

    for k in required_keys:
        data.setdefault(k, None)

    if "__reasoning_trace" in data:
        logger.info(f"[ReasoningTrace] {data['__reasoning_trace'][:200]}")
        data.pop("__reasoning_trace")

    return data


# ==========================================================
# 🚀 MAIN CLASSIFIER FUNCTION
# ==========================================================
def classify_text_with_llm(text: str, max_retries: int = 2) -> dict:
    """
    Send text to local Ollama / DeepSeek LLM and get structured classification.
    Retries automatically on malformed JSON.
    """
    if not text or not text.strip():
        return {
            "entity_type": "content",
            "category_level_1": "Uncategorized",
            "category_level_2_sub": "General Business",
            "business_concept_name": None,
            "business_specific_name": None,
            "primary_process_type": "Other",
            "title": "Empty Input",
            "description": "No text provided for classification.",
            "extraction_confidence": 0.0,
        }

    for attempt in range(max_retries):
        try:
            payload = {
                "model": MODEL_NAME,
                "prompt": f"{PROMPT_TEXT}\n\n[INPUT]: {text.strip()}",
                "stream": False,
                "temperature": 0.1,
            }

            response = requests.post(OLLAMA_API, json=payload, timeout=180)
            response.raise_for_status()

            # Ollama returns JSON like {"model": ..., "created_at": ..., "response": "..."}
            raw_output = response.json().get("response", "").strip()
            if not raw_output:
                raise ValueError("Empty model response.")

            parsed = parse_llm_json_output(raw_output)
            normalized = normalize_json_schema(parsed)

            # If high-quality classification found, break retry loop
            if normalized.get("category_level_1") not in ("Uncategorized", None):
                logger.info(f"[LLMCall] ✅ Success | Attempt {attempt+1} | {len(text)} chars processed")
                return normalized

        except Exception as e:
            logger.error(f"[LLMError] Attempt {attempt+1}: {e}")

    # Fallback — if all retries fail
    logger.warning(f"[LLMFallback] Max retries reached for text: {text[:100]}...")
    return {
        "entity_type": "content",
        "category_level_1": "Uncategorized",
        "category_level_2_sub": "General Business",
        "business_concept_name": None,
        "business_specific_name": None,
        "primary_process_type": "Other",
        "title": "Classification Error",
        "description": text[:200],
        "extraction_confidence": 0.4,
    }


# ==========================================================
# 🧪 TEST EXECUTION
# ==========================================================
if __name__ == "__main__":
    sample_texts = [
        "Setting up a commercial manufacturing unit for synthetic rubber gloves for surgical use.",
        "Trends in organic fertilizer production and retail in southern India.",
        "Feasibility report for solar rooftop installation by GreenPower Pvt Ltd."
    ]
    for txt in sample_texts:
        result = classify_text_with_llm(txt)
        print(json.dumps(result, indent=2))
        print("-" * 60)
