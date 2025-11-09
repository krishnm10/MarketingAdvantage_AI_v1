import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load persona definition
PROMPT_PATH = Path(__file__).resolve().parent / "persona_system_prompt.txt"
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

def build_prompt(user_query: str, context: dict | None = None) -> str:
    """Combine persona prompt, context and user message."""
    ctx_text = ""
    if context:
        ctx_text = "\n\nContext:\n" + "\n".join(
            f"{k}: {v}" for k, v in context.items()
        )
    return f"{SYSTEM_PROMPT}\n\nUser request:\n{user_query}{ctx_text}"

def run_gpt(user_query: str, context: dict | None = None) -> str:
    """Send prompt to GPT model and return text output."""
    prompt = build_prompt(user_query, context)
    completion = client.chat.completions.create(
        model=os.getenv("MODEL", "gpt-5"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ],
        max_tokens=900,
        temperature=0.7,
    )
    return completion.choices[0].message.content.strip()
