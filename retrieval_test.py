from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEPENDENCY_HINT = "pip install pandas numpy faiss-cpu sentence-transformers pyarrow"

try:
    import pandas as pd

    from bm25_retrieval import BM25Index, bm25_search, load_bm25_index
    from faiss_semantic_search import DenseIndex, dense_search, load_faiss_index
except ImportError as exc:
    print(f"ERROR: Missing dependency: {exc.name}", file=sys.stderr)
    print(f"Install with: {DEPENDENCY_HINT}", file=sys.stderr)
    raise SystemExit(1) from exc


@dataclass
class Corpus:
    frame: pd.DataFrame
    text_col: str


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


def load_corpus(base_dir: Path) -> Corpus:
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

    preview_cols = [col for col in ("id", "doc_id", "url", "source", text_col) if col in frame.columns]
    if not preview_cols:
        preview_cols = [text_col]
    preview = frame.loc[:, preview_cols].head(3).copy()
    if text_col in preview.columns:
        preview[text_col] = preview[text_col].map(lambda value: truncate_text(value, max_chars=160))
    print("Corpus preview:")
    print(preview.to_string(index=True))
    print()

    return Corpus(frame=frame, text_col=text_col)


def row_to_record(
    corpus: Corpus,
    row_id: int,
    *,
    bm25_score: float | None = None,
    dense_score: float | None = None,
    hybrid_score: float | None = None,
    bm25_rank: int | None = None,
    dense_rank: int | None = None,
) -> dict[str, Any]:
    row = corpus.frame.iloc[row_id]
    record: dict[str, Any] = {
        "row_id": int(row_id),
        "passage": row[corpus.text_col],
    }
    if hybrid_score is not None:
        record["score"] = float(hybrid_score)
        record["hybrid_score"] = float(hybrid_score)
    if bm25_score is not None:
        record["bm25_score"] = float(bm25_score)
    if dense_score is not None:
        record["dense_score"] = float(dense_score)
    if bm25_rank is not None:
        record["bm25_rank"] = int(bm25_rank)
    if dense_rank is not None:
        record["dense_rank"] = int(dense_rank)
    for col in ("url", "source"):
        if col in corpus.frame.columns:
            record[col] = row[col]
    return record


def hybrid_search(
    query: str,
    corpus: Corpus,
    bm25: BM25Index,
    dense: DenseIndex,
    *,
    bm25_k: int = 100,
    dense_k: int = 100,
    top_k: int = 10,
    rrf_k: int = 60,
) -> pd.DataFrame:
    bm25_results = bm25_search(
        query,
        bm25_k,
        bm25,
        corpus=corpus.frame,
        text_col=corpus.text_col,
    )
    dense_results = dense_search(
        query,
        dense_k,
        dense,
        corpus=corpus.frame,
        text_col=corpus.text_col,
    )

    fused: dict[int, dict[str, Any]] = {}

    for zero_rank, row in enumerate(bm25_results.itertuples(index=False), start=0):
        row_id = int(row.row_id)
        entry = fused.setdefault(row_id, {"row_id": row_id, "hybrid_score": 0.0})
        entry["hybrid_score"] += 1.0 / (rrf_k + zero_rank + 1)
        entry["bm25_rank"] = zero_rank + 1
        entry["bm25_score"] = float(row.bm25_score)

    for zero_rank, row in enumerate(dense_results.itertuples(index=False), start=0):
        row_id = int(row.row_id)
        entry = fused.setdefault(row_id, {"row_id": row_id, "hybrid_score": 0.0})
        entry["hybrid_score"] += 1.0 / (rrf_k + zero_rank + 1)
        entry["dense_rank"] = zero_rank + 1
        entry["dense_score"] = float(row.dense_score)

    ranked = sorted(fused.values(), key=lambda item: item["hybrid_score"], reverse=True)[:top_k]
    records = []
    for item in ranked:
        records.append(
            row_to_record(
                corpus,
                int(item["row_id"]),
                hybrid_score=float(item["hybrid_score"]),
                bm25_score=item.get("bm25_score"),
                dense_score=item.get("dense_score"),
                bm25_rank=item.get("bm25_rank"),
                dense_rank=item.get("dense_rank"),
            )
        )
    return pd.DataFrame.from_records(records)


def truncate_text(value: Any, max_chars: int = 500) -> str:
    text = re.sub(r"\s+", " ", "" if pd.isna(value) else str(value)).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def format_float(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.6f}"


def print_results(results: pd.DataFrame, *, mode: str, max_chars: int = 500) -> None:
    if results.empty:
        print("No results.")
        return

    if "hybrid_score" in results.columns:
        score_col = "hybrid_score"
    elif "bm25_score" in results.columns:
        score_col = "bm25_score"
    elif "dense_score" in results.columns:
        score_col = "dense_score"
    else:
        score_col = "score" if "score" in results.columns else None

    print(f"Results ({mode}):")
    print("=" * 80)
    for rank, row in enumerate(results.to_dict(orient="records"), start=1):
        score = format_float(row.get(score_col)) if score_col else "NA"
        parts = [f"rank={rank}", f"row_id={row.get('row_id')}", f"score={score}"]

        if "bm25_score" in row and not pd.isna(row.get("bm25_score")):
            parts.append(f"bm25_score={format_float(row.get('bm25_score'))}")
        if "dense_score" in row and not pd.isna(row.get("dense_score")):
            parts.append(f"dense_score={format_float(row.get('dense_score'))}")
        if "bm25_rank" in row and not pd.isna(row.get("bm25_rank")):
            parts.append(f"bm25_rank={int(row.get('bm25_rank'))}")
        if "dense_rank" in row and not pd.isna(row.get("dense_rank")):
            parts.append(f"dense_rank={int(row.get('dense_rank'))}")

        print(" | ".join(parts))
        for col in ("url", "source"):
            if col in row and not pd.isna(row.get(col)):
                print(f"{col}: {row.get(col)}")
        print(truncate_text(row.get("passage"), max_chars=max_chars))
        print("-" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test retrieval from prebuilt MS MARCO indexes.")
    parser.add_argument("--query", required=True, help="Query text.")
    parser.add_argument(
        "--mode",
        choices=("bm25", "dense", "hybrid"),
        default="hybrid",
        help="Retrieval mode.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of final results.")
    parser.add_argument("--bm25-k", type=int, default=100, help="BM25 candidates for hybrid.")
    parser.add_argument("--dense-k", type=int, default=100, help="Dense candidates for hybrid.")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF constant for hybrid fusion.")
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Artifact base directory. Defaults to outputs/; falls back to repo root if artifacts are there.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = resolve_base_dir(args.base_dir)
    print(f"Artifact base directory: {base_dir}")

    corpus = load_corpus(base_dir)

    if args.mode == "bm25":
        bm25 = load_bm25_index(base_dir)
        if bm25.num_docs != len(corpus.frame):
            print(
                f"WARNING: BM25 num_docs ({bm25.num_docs:,}) differs from corpus rows ({len(corpus.frame):,}).",
                file=sys.stderr,
            )
        results = bm25_search(
            args.query,
            args.top_k,
            bm25,
            corpus=corpus.frame,
            text_col=corpus.text_col,
        )
    elif args.mode == "dense":
        dense = load_faiss_index(base_dir, corpus_size=len(corpus.frame), dependency_hint=DEPENDENCY_HINT)
        results = dense_search(
            args.query,
            args.top_k,
            dense,
            corpus=corpus.frame,
            text_col=corpus.text_col,
        )
    else:
        bm25 = load_bm25_index(base_dir)
        if bm25.num_docs != len(corpus.frame):
            print(
                f"WARNING: BM25 num_docs ({bm25.num_docs:,}) differs from corpus rows ({len(corpus.frame):,}).",
                file=sys.stderr,
            )
        dense = load_faiss_index(base_dir, corpus_size=len(corpus.frame), dependency_hint=DEPENDENCY_HINT)
        results = hybrid_search(
            args.query,
            corpus,
            bm25,
            dense,
            bm25_k=args.bm25_k,
            dense_k=args.dense_k,
            top_k=args.top_k,
            rrf_k=args.rrf_k,
        )

    print_results(results, mode=args.mode)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Dependency hint: {DEPENDENCY_HINT}", file=sys.stderr)
        raise SystemExit(1) from exc
