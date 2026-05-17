"""RAG generation module backed by OpenRouter chat completions."""

from .rag import (
    DEFAULT_MODEL,
    RagConfig,
    RagContext,
    RagResult,
    build_rag_prompt,
    find_api_key,
    generate_rag_answer,
)

__all__ = [
    "DEFAULT_MODEL",
    "RagConfig",
    "RagContext",
    "RagResult",
    "build_rag_prompt",
    "find_api_key",
    "generate_rag_answer",
]
