"""
================================================================================
Marketing Advantage AI — LLM Plugin Registration
File: app/core/llms/register.py

Import once at app startup to register all LLM connectors.
NO default LLM is set — client config must specify type + model.

After import, usage from factory:
  llm_registry.build("ollama",    model="llama3.2", base_url="...")
  llm_registry.build("openai",    model="gpt-4o-mini", api_key="...")
  llm_registry.build("groq",      model="llama3-8b-8192", api_key="...")
  llm_registry.build("anthropic", model="claude-3-5-sonnet-20241022", api_key="...")
  llm_registry.build("gemini",    model="gemini-1.5-flash-latest", api_key="...")
================================================================================
"""

from app.core.plugin_registry import llm_registry

from app.core.llms.ollama_v1    import OllamaLLM
from app.core.llms.openai_v1    import OpenAILLM
from app.core.llms.groq_v1      import GroqLLM
from app.core.llms.anthropic_v1 import AnthropicLLM
from app.core.llms.gemini_v1    import GeminiLLM

llm_registry.register(
    "ollama",
    OllamaLLM,
    description="Ollama local LLM (any pulled model). CPU + GPU.",
)

llm_registry.register(
    "openai",
    OpenAILLM,
    description="OpenAI GPT-4o/GPT-4-turbo. Also works with Azure OpenAI + vLLM.",
)

llm_registry.register(
    "groq",
    GroqLLM,
    description="Groq LPU inference — fastest API available. Free tier generous.",
)

llm_registry.register(
    "anthropic",
    AnthropicLLM,
    description="Anthropic Claude (3.5 Sonnet/Haiku). 200K context. Best reasoning.",
)

llm_registry.register(
    "gemini",
    GeminiLLM,
    description="Google Gemini (1.5 Pro/Flash, 2.0 Flash). 1M context. Multilingual.",
)
