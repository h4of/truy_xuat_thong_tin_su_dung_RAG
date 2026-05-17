from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

DEPENDENCY_HINT = "pip install pandas numpy faiss-cpu sentence-transformers pyarrow"

try:
    import pandas as pd

    from rag_engine import DEFAULT_MODEL, RagConfig, generate_rag_answer
    from search_engine import SearchCorpus, hybrid_search, load_bm25_index, load_faiss_index
except ImportError as exc:
    print(f"ERROR: Missing dependency: {exc.name}", file=sys.stderr)
    print(f"Install with: {DEPENDENCY_HINT}", file=sys.stderr)
    raise SystemExit(1) from exc


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def has_expected_artifacts(base_dir: Path) -> bool:
    return (
        (base_dir / "corpus" / "corpus.parquet").is_file()
        and (base_dir / "indexes" / "faiss_index.index").is_file()
        and (base_dir / "indexes" / "bm25" / "vocab.json").is_file()
    )


def resolve_base_dir(base_dir_arg: str | None) -> Path:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()

    if base_dir_arg:
        base_dir = Path(base_dir_arg).expanduser()
        if not base_dir.is_absolute():
            base_dir = (cwd / base_dir).resolve()
        return base_dir

    candidates = []
    for candidate in (cwd / "outputs", script_dir / "outputs", cwd, script_dir):
        resolved = candidate.resolve()
        if resolved not in candidates:
            candidates.append(resolved)

    for candidate in candidates:
        if has_expected_artifacts(candidate):
            preferred = (cwd / "outputs").resolve()
            if candidate != preferred:
                print(
                    f"WARNING: outputs/ was not found with all artifacts; using {candidate} instead.",
                    file=sys.stderr,
                )
            return candidate

    return (cwd / "outputs").resolve()


def load_corpus(base_dir: Path) -> SearchCorpus:
    corpus_path = base_dir / "corpus" / "corpus.parquet"
    require_file(corpus_path)

    try:
        frame = pd.read_parquet(corpus_path)
    except ImportError as exc:
        raise RuntimeError(
            "Unable to read parquet corpus because pandas could not find a parquet engine "
            "(pyarrow or fastparquet). Install dependencies with: "
            f"{DEPENDENCY_HINT}"
        ) from exc

    if "chunk_text" in frame.columns:
        text_col = "chunk_text"
    elif "text" in frame.columns:
        text_col = "text"
    else:
        raise ValueError(
            f"Corpus must contain either 'chunk_text' or 'text'. Available columns: {list(frame.columns)}"
        )

    print(f"Loaded corpus: {len(frame):,} rows from {corpus_path}")
    print(f"Using text column: {text_col}")
    return SearchCorpus(frame=frame, text_col=text_col)


def retrieval_rows_for_rag(results: pd.DataFrame) -> list[dict[str, Any]]:
    records = results.to_dict(orient="records")
    contexts = []
    for rank, record in enumerate(records, start=1):
        context = dict(record)
        context["rank"] = rank
        contexts.append(context)
    return contexts


def truncate_text(value: Any, max_chars: int = 300) -> str:
    text = re.sub(r"\s+", " ", "" if pd.isna(value) else str(value)).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def print_retrieved_contexts(results: pd.DataFrame, *, max_chars: int = 300) -> None:
    if results.empty:
        print("No retrieved contexts.")
        return

    print("\nRetrieved contexts:")
    print("=" * 80)
    for rank, row in enumerate(results.to_dict(orient="records"), start=1):
        score = row.get("score", row.get("hybrid_score"))
        score_text = "NA" if score is None or pd.isna(score) else f"{float(score):.6f}"
        parts = [f"rank={rank}", f"row_id={row.get('row_id')}", f"score={score_text}"]
        if "bm25_rank" in row and not pd.isna(row.get("bm25_rank")):
            parts.append(f"bm25_rank={int(row['bm25_rank'])}")
        if "dense_rank" in row and not pd.isna(row.get("dense_rank")):
            parts.append(f"dense_rank={int(row['dense_rank'])}")
        print(" | ".join(parts))
        if row.get("url") and not pd.isna(row.get("url")):
            print(f"url: {row['url']}")
        if row.get("source") and not pd.isna(row.get("source")):
            print(f"source: {row['source']}")
        print(truncate_text(row.get("passage"), max_chars=max_chars))
        print("-" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid retrieval top-5 + Gemma RAG smoke test.")
    parser.add_argument("--query", required=True, help="Question for retrieval and RAG.")
    parser.add_argument("--base-dir", default=None, help="Artifact base directory. Defaults to outputs/, then repo root.")
    parser.add_argument("--bm25-k", type=int, default=100)
    parser.add_argument("--dense-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--retrieval-top-k", type=int, default=5, help="Number of hybrid results passed to RAG.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouter model id. Default: {DEFAULT_MODEL}")
    parser.add_argument("--api-key", default=None, help="OpenRouter API key. Prefer env/.env for normal use.")
    parser.add_argument("--env-file", default=None, help="Optional .env file path.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--max-context-chars", type=int, default=2_000)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--hide-contexts", action="store_true", help="Do not print retrieved passages before the answer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = resolve_base_dir(args.base_dir)
    print(f"Artifact base directory: {base_dir}")

    corpus = load_corpus(base_dir)
    bm25 = load_bm25_index(base_dir)
    if bm25.num_docs != len(corpus.frame):
        print(
            f"WARNING: BM25 num_docs ({bm25.num_docs:,}) differs from corpus rows ({len(corpus.frame):,}).",
            file=sys.stderr,
        )
    dense = load_faiss_index(base_dir, corpus_size=len(corpus.frame), dependency_hint=DEPENDENCY_HINT)

    retrieved = hybrid_search(
        args.query,
        corpus,
        bm25,
        dense,
        bm25_k=args.bm25_k,
        dense_k=args.dense_k,
        top_k=args.retrieval_top_k,
        rrf_k=args.rrf_k,
    )
    if not args.hide_contexts:
        print_retrieved_contexts(retrieved)

    rag_config = RagConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_context_chars=args.max_context_chars,
        timeout_seconds=args.timeout_seconds,
    )
    rag_result = generate_rag_answer(
        args.query,
        retrieval_rows_for_rag(retrieved),
        api_key=args.api_key,
        env_file=args.env_file,
        config=rag_config,
    )

    print("\nRAG answer:")
    print("=" * 80)
    print(rag_result.answer)
    if rag_result.usage:
        print("\nusage:")
        print(rag_result.usage)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Dependency hint: {DEPENDENCY_HINT}", file=sys.stderr)
        raise SystemExit(1) from exc
