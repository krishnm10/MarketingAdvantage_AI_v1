"""
Embedder plugin registration
File: app/core/embedders/register.py

Import once at startup to register all embedder connectors.
NO default is chosen here — client config must specify embedder type + model.
"""

from app.core.plugin_registry import embedder_registry

from app.core.embedders.huggingface_st_v1 import HuggingFaceSTEmbedder
from app.core.embedders.ollama_v1 import OllamaEmbedder
from app.core.embedders.openai_v1 import OpenAIEmbedder
from app.core.embedders.cohere_v1 import CohereEmbedder

embedder_registry.register(
    "huggingface",
    HuggingFaceSTEmbedder,
    description="SentenceTransformers (HuggingFace) embedder.",
)

embedder_registry.register(
    "ollama",
    OllamaEmbedder,
    description="Local Ollama embeddings (BYO model).",
)

embedder_registry.register(
    "openai",
    OpenAIEmbedder,
    description="OpenAI embeddings (text-embedding-3-*).",
)

embedder_registry.register(
    "cohere",
    CohereEmbedder,
    description="Cohere embeddings (input_type-aware).",
)
