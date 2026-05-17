from __future__ import annotations

"""Production-style hybrid BM25 + FAISS retrieval with CrossEncoder reranking."""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

DEPENDENCY_HINT = "pip install pandas numpy faiss-cpu sentence-transformers pyarrow"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"

try:
    import pandas as pd

    from .bm25_retrieval import BM25Index, bm25_search, load_bm25_index
    from .faiss_semantic_search import DenseIndex, dense_search, load_faiss_index
except ImportError as exc:
    print(f"ERROR: Missing dependency: {exc.name}", file=sys.stderr)
    print(f"Install with: {DEPENDENCY_HINT}", file=sys.stderr)
    raise SystemExit(1) from exc


@dataclass(frozen=True)
class Corpus:
    frame: pd.DataFrame
    text_col: str


@dataclass(frozen=True)
class HybridRerankConfig:
    alpha: float = 0.5
    bm25_k: int = 100
    dense_k: int = 100
    candidate_k: int = 50
    top_k: int = 10
    rrf_k: int = 60
    rerank_batch_size: int = 32


@dataclass
class HybridRerankRetriever:
    corpus: Corpus
    bm25: BM25Index
    dense: DenseIndex
    reranker: Any

    @classmethod
    def from_artifacts(
        cls,
        base_dir: str | Path,
        *,
        embedding_model_name: str | None = None,
        reranker_model_name: str = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
    ) -> "HybridRerankRetriever":
        base_path = Path(base_dir)
        corpus = load_corpus(base_path)
        bm25 = load_bm25_index(base_path)
        if bm25.num_docs != len(corpus.frame):
            print(
                f"WARNING: BM25 num_docs ({bm25.num_docs:,}) differs from corpus rows ({len(corpus.frame):,}).",
                file=sys.stderr,
            )

        dense_kwargs: dict[str, Any] = {
            "corpus_size": len(corpus.frame),
            "dependency_hint": DEPENDENCY_HINT,
        }
        if embedding_model_name:
            dense_kwargs["model_name"] = embedding_model_name
        dense = load_faiss_index(base_path, **dense_kwargs)

        reranker = load_reranker(reranker_model_name, device=device)
        return cls(corpus=corpus, bm25=bm25, dense=dense, reranker=reranker)

    def search(
        self,
        query: str,
        *,
        alpha: float = 0.5,
        bm25_k: int = 100,
        dense_k: int = 100,
        candidate_k: int = 50,
        top_k: int = 10,
        rrf_k: int = 60,
        rerank_batch_size: int = 32,
    ) -> pd.DataFrame:
        config = HybridRerankConfig(
            alpha=alpha,
            bm25_k=bm25_k,
            dense_k=dense_k,
            candidate_k=candidate_k,
            top_k=top_k,
            rrf_k=rrf_k,
            rerank_batch_size=rerank_batch_size,
        )
        return hybrid_rerank_search(
            query,
            self.corpus,
            self.bm25,
            self.dense,
            self.reranker,
            config=config,
        )


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


def load_corpus(base_dir: str | Path) -> Corpus:
    base_path = Path(base_dir)
    corpus_path = base_path / "corpus" / "corpus.parquet"
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
    return Corpus(frame=frame, text_col=text_col)


def load_reranker(model_name: str = DEFAULT_RERANKER_MODEL, *, device: str | None = None) -> Any:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            f"Missing dependency: sentence-transformers. Install with: {DEPENDENCY_HINT}"
        ) from exc

    kwargs: dict[str, Any] = {}
    if device:
        kwargs["device"] = device

    print(f"Loading reranker model: {model_name}")
    try:
        return CrossEncoder(model_name, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load reranker model '{model_name}'. "
            "Ensure it is installed/cached or that network access is available. "
            f"Original error: {exc}"
        ) from exc


def validate_config(config: HybridRerankConfig) -> None:
    if not 0.0 <= config.alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {config.alpha}")
    if config.bm25_k <= 0:
        raise ValueError(f"bm25_k must be positive, got {config.bm25_k}")
    if config.dense_k <= 0:
        raise ValueError(f"dense_k must be positive, got {config.dense_k}")
    if config.candidate_k <= 0:
        raise ValueError(f"candidate_k must be positive, got {config.candidate_k}")
    if config.top_k <= 0:
        raise ValueError(f"top_k must be positive, got {config.top_k}")
    if config.rrf_k < 0:
        raise ValueError(f"rrf_k must be non-negative, got {config.rrf_k}")
    if config.rerank_batch_size <= 0:
        raise ValueError(
            f"rerank_batch_size must be positive, got {config.rerank_batch_size}"
        )


def row_to_record(
    corpus: Corpus,
    row_id: int,
    *,
    hybrid_score: float,
    rerank_score: float,
    bm25_score: float | None = None,
    dense_score: float | None = None,
    bm25_rank: int | None = None,
    dense_rank: int | None = None,
    extra_cols: Sequence[str] = ("url", "source"),
) -> dict[str, Any]:
    row = corpus.frame.iloc[row_id]
    record: dict[str, Any] = {
        "row_id": int(row_id),
        "score": float(rerank_score),
        "rerank_score": float(rerank_score),
        "hybrid_score": float(hybrid_score),
        "passage": row[corpus.text_col],
    }
    if bm25_score is not None:
        record["bm25_score"] = float(bm25_score)
    if dense_score is not None:
        record["dense_score"] = float(dense_score)
    if bm25_rank is not None:
        record["bm25_rank"] = int(bm25_rank)
    if dense_rank is not None:
        record["dense_rank"] = int(dense_rank)
    for col in extra_cols:
        if col in corpus.frame.columns:
            record[col] = row[col]
    return record


def fuse_candidates(
    bm25_results: pd.DataFrame,
    dense_results: pd.DataFrame,
    *,
    alpha: float,
    rrf_k: int,
) -> list[dict[str, Any]]:
    fused: dict[int, dict[str, Any]] = {}

    for zero_rank, row in enumerate(bm25_results.itertuples(index=False), start=0):
        row_id = int(row.row_id)
        entry = fused.setdefault(row_id, {"row_id": row_id, "hybrid_score": 0.0})
        entry["hybrid_score"] += (1.0 / (rrf_k + zero_rank + 1)) * (1.0 - alpha)
        entry["bm25_rank"] = zero_rank + 1
        entry["bm25_score"] = float(row.bm25_score)

    for zero_rank, row in enumerate(dense_results.itertuples(index=False), start=0):
        row_id = int(row.row_id)
        entry = fused.setdefault(row_id, {"row_id": row_id, "hybrid_score": 0.0})
        entry["hybrid_score"] += (1.0 / (rrf_k + zero_rank + 1)) * alpha
        entry["dense_rank"] = zero_rank + 1
        entry["dense_score"] = float(row.dense_score)

    return sorted(fused.values(), key=lambda item: item["hybrid_score"], reverse=True)


def hybrid_rerank_search(
    query: str,
    corpus: Corpus,
    bm25: BM25Index,
    dense: DenseIndex,
    reranker: Any,
    *,
    config: HybridRerankConfig | None = None,
) -> pd.DataFrame:
    config = config or HybridRerankConfig()
    validate_config(config)

    bm25_results = bm25_search(
        query,
        config.bm25_k,
        bm25,
        corpus=corpus.frame,
        text_col=corpus.text_col,
    )
    dense_results = dense_search(
        query,
        config.dense_k,
        dense,
        corpus=corpus.frame,
        text_col=corpus.text_col,
    )

    candidates = fuse_candidates(
        bm25_results,
        dense_results,
        alpha=config.alpha,
        rrf_k=config.rrf_k,
    )[: config.candidate_k]
    if not candidates:
        return empty_results()

    pairs = []
    valid_candidates = []
    for item in candidates:
        row_id = int(item["row_id"])
        if row_id < 0 or row_id >= len(corpus.frame):
            print(
                f"WARNING: skipped candidate with row_id outside corpus: {row_id}",
                file=sys.stderr,
            )
            continue
        text = corpus.frame.iloc[row_id][corpus.text_col]
        pairs.append((query, "" if pd.isna(text) else str(text)))
        valid_candidates.append(item)

    if not valid_candidates:
        return empty_results()

    rerank_scores = reranker.predict(pairs, batch_size=config.rerank_batch_size)
    for item, rerank_score in zip(valid_candidates, rerank_scores):
        item["rerank_score"] = float(rerank_score)

    final_ranked = sorted(
        valid_candidates,
        key=lambda item: item["rerank_score"],
        reverse=True,
    )[: config.top_k]

    records = [
        row_to_record(
            corpus,
            int(item["row_id"]),
            hybrid_score=float(item["hybrid_score"]),
            rerank_score=float(item["rerank_score"]),
            bm25_score=item.get("bm25_score"),
            dense_score=item.get("dense_score"),
            bm25_rank=item.get("bm25_rank"),
            dense_rank=item.get("dense_rank"),
        )
        for item in final_ranked
    ]
    return pd.DataFrame.from_records(records)


def empty_results() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "row_id",
            "score",
            "rerank_score",
            "hybrid_score",
            "bm25_score",
            "dense_score",
            "bm25_rank",
            "dense_rank",
            "passage",
        ]
    )


def truncate_text(value: Any, max_chars: int = 500) -> str:
    text = re.sub(r"\s+", " ", "" if pd.isna(value) else str(value)).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def format_float(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.6f}"


def print_results(results: pd.DataFrame, *, max_chars: int = 500) -> None:
    if results.empty:
        print("No results.")
        return

    print("Results (hybrid + rerank):")
    print("=" * 80)
    for rank, row in enumerate(results.to_dict(orient="records"), start=1):
        parts = [
            f"rank={rank}",
            f"row_id={row.get('row_id')}",
            f"score={format_float(row.get('score'))}",
            f"rerank_score={format_float(row.get('rerank_score'))}",
            f"hybrid_score={format_float(row.get('hybrid_score'))}",
        ]
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
    parser = argparse.ArgumentParser(
        description="Production-style hybrid BM25 + FAISS + CrossEncoder rerank retrieval."
    )
    parser.add_argument("--query", required=True, help="Query text.")
    parser.add_argument("--top-k", type=int, default=10, help="Final reranked results.")
    parser.add_argument("--candidate-k", type=int, default=50, help="Hybrid candidates to rerank.")
    parser.add_argument("--bm25-k", type=int, default=100, help="BM25 candidates before fusion.")
    parser.add_argument("--dense-k", type=int, default=100, help="Dense candidates before fusion.")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF constant.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dense RRF weight; BM25 weight is 1-alpha.")
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument("--base-dir", default=None, help="Artifact base dir. Defaults to outputs/, then repo root.")
    parser.add_argument("--embedding-model", default=None, help="Override dense embedding model.")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--device", default=None, help="Optional CrossEncoder device, e.g. cpu, cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = resolve_base_dir(args.base_dir)
    print(f"Artifact base directory: {base_dir}")

    retriever = HybridRerankRetriever.from_artifacts(
        base_dir,
        embedding_model_name=args.embedding_model,
        reranker_model_name=args.reranker_model,
        device=args.device,
    )
    results = retriever.search(
        args.query,
        alpha=args.alpha,
        bm25_k=args.bm25_k,
        dense_k=args.dense_k,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        rerank_batch_size=args.rerank_batch_size,
    )
    print_results(results)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Dependency hint: {DEPENDENCY_HINT}", file=sys.stderr)
        raise SystemExit(1) from exc
