from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-31b-it:free"
API_KEY_NAMES = ("OPENROUTER_API_KEY", "API_KEY")


@dataclass(frozen=True)
class RagContext:
    text: str
    row_id: int | None = None
    source: str | None = None
    url: str | None = None
    score: float | None = None
    rank: int | None = None


@dataclass(frozen=True)
class RagConfig:
    model: str = DEFAULT_MODEL
    temperature: float = 0.1
    max_tokens: int = 700
    max_context_chars: int = 2_000
    timeout_seconds: int = 90
    site_url: str = "http://localhost"
    app_title: str = "Hybrid Search RAG Test"


@dataclass(frozen=True)
class RagResult:
    answer: str
    model: str
    raw_response: dict[str, Any]
    contexts_used: list[RagContext]
    usage: dict[str, Any] | None = None


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def find_api_key(explicit_api_key: str | None = None, env_file: str | None = None) -> tuple[str, str]:
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip(), "--api-key"

    for name in API_KEY_NAMES:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip(), f"environment variable {name}"

    env_paths: list[Path] = []
    if env_file:
        env_paths.append(Path(env_file))
    else:
        env_paths.extend([Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"])

    seen_paths: set[Path] = set()
    for path in env_paths:
        path = path.resolve()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        values = parse_env_file(path)
        for name in API_KEY_NAMES:
            value = values.get(name)
            if value and value.strip():
                return value.strip(), f"{path.name} ({name})"

    names = " or ".join(API_KEY_NAMES)
    raise RuntimeError(
        f"Missing API key. Set {names}, add it to .env, or pass --api-key. "
        'PowerShell example: $env:OPENROUTER_API_KEY="sk-or-v1-..."'
    )


def normalize_contexts(
    contexts: Sequence[str | Mapping[str, Any] | RagContext],
    *,
    max_context_chars: int,
) -> list[RagContext]:
    normalized: list[RagContext] = []
    for index, item in enumerate(contexts, start=1):
        if isinstance(item, RagContext):
            context = item
        elif isinstance(item, str):
            context = RagContext(text=item, rank=index)
        else:
            text = item.get("passage") or item.get("text") or item.get("chunk_text") or item.get("content")
            context = RagContext(
                text="" if text is None else str(text),
                row_id=to_optional_int(item.get("row_id")),
                source=to_optional_str(item.get("source")),
                url=to_optional_str(item.get("url")),
                score=to_optional_float(item.get("score")),
                rank=to_optional_int(item.get("rank")) or index,
            )

        text = context.text.strip()
        if not text:
            continue
        if len(text) > max_context_chars:
            text = text[: max_context_chars - 3].rstrip() + "..."
        normalized.append(
            RagContext(
                text=text,
                row_id=context.row_id,
                source=context.source,
                url=context.url,
                score=context.score,
                rank=context.rank or index,
            )
        )
    return normalized


def to_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def to_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_rag_prompt(query: str, contexts: Sequence[RagContext]) -> list[dict[str, str]]:
    system_prompt = (
        "You are a careful retrieval-augmented QA assistant. "
        "Answer using only the provided passages. "
        "If the passages do not contain enough information, say that the retrieved context is insufficient. "
        "Keep the answer concise, factual, and in English. "
        "Cite passages with bracketed numbers like [1] when you use them."
    )

    context_blocks = []
    for index, context in enumerate(contexts, start=1):
        metadata = []
        if context.row_id is not None:
            metadata.append(f"row_id={context.row_id}")
        if context.score is not None:
            metadata.append(f"score={context.score:.6f}")
        if context.url:
            metadata.append(f"url={context.url}")
        if context.source:
            metadata.append(f"source={context.source}")
        metadata_text = f" ({'; '.join(metadata)})" if metadata else ""
        context_blocks.append(f"[{index}]{metadata_text}\n{context.text}")

    user_prompt = (
        "Question:\n"
        f"{query.strip()}\n\n"
        "Retrieved passages:\n"
        f"{chr(10).join(context_blocks) if context_blocks else 'No passages were provided.'}\n\n"
        "Write the final answer in English. Use citations like [1] or [2]."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_openrouter(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    config: RagConfig,
) -> dict[str, Any]:
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": config.site_url,
            "X-Title": config.app_title,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach OpenRouter: {exc.reason}") from exc


def extract_answer(response: Mapping[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter response has no choices: {json.dumps(response, ensure_ascii=False)}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        raise RuntimeError(f"OpenRouter response has no message content: {json.dumps(response, ensure_ascii=False)}")
    return str(content).strip()


def generate_rag_answer(
    query: str,
    contexts: Sequence[str | Mapping[str, Any] | RagContext],
    *,
    api_key: str | None = None,
    env_file: str | None = None,
    config: RagConfig | None = None,
) -> RagResult:
    if not query.strip():
        raise ValueError("query must not be empty.")

    config = config or RagConfig()
    resolved_api_key, api_key_source = find_api_key(api_key, env_file)
    print(f"Using API key from: {api_key_source}", file=sys.stderr)

    normalized_contexts = normalize_contexts(
        contexts,
        max_context_chars=config.max_context_chars,
    )
    messages = build_rag_prompt(query, normalized_contexts)
    response = call_openrouter(messages, api_key=resolved_api_key, config=config)
    answer = extract_answer(response)

    usage = response.get("usage")
    if usage is not None and not isinstance(usage, dict):
        usage = {"raw_usage": usage}

    return RagResult(
        answer=answer,
        model=str(response.get("model", config.model)),
        raw_response=dict(response),
        contexts_used=normalized_contexts,
        usage=usage,
    )
