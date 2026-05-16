from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class DenseIndex:
    index: Any
    model: Any


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_faiss_index(
    base_dir: str | Path,
    *,
    corpus_size: int | None = None,
    model_name: str = MODEL_NAME,
    dependency_hint: str = "pip install pandas numpy faiss-cpu sentence-transformers pyarrow",
) -> DenseIndex:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency: faiss. Install with: {dependency_hint}") from exc

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            f"Missing dependency: sentence-transformers. Install with: {dependency_hint}"
        ) from exc

    base_path = Path(base_dir)
    index_path = base_path / "indexes" / "faiss_index.index"
    require_file(index_path)

    index = faiss.read_index(str(index_path))
    print(f"Loaded FAISS index: {index.ntotal:,} vectors from {index_path}")
    if corpus_size is not None and index.ntotal != corpus_size:
        print(
            f"WARNING: FAISS index size ({index.ntotal:,}) differs from corpus rows ({corpus_size:,}).",
            file=sys.stderr,
        )

    print(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embedding model '{model_name}'. "
            "Ensure it is installed/cached or that network access is available. "
            f"Original error: {exc}"
        ) from exc

    return DenseIndex(index=index, model=model)


def add_corpus_fields(
    records: list[dict[str, Any]],
    corpus: pd.DataFrame | None,
    text_col: str | None,
    extra_cols: Sequence[str],
) -> list[dict[str, Any]]:
    if corpus is None:
        return records
    if text_col is None:
        raise ValueError("text_col is required when corpus is provided.")
    if text_col not in corpus.columns:
        raise ValueError(f"text_col '{text_col}' not found in corpus.")

    enriched = []
    skipped = 0
    for record in records:
        row_id = int(record["row_id"])
        if row_id >= len(corpus):
            skipped += 1
            continue
        row = corpus.iloc[row_id]
        enriched_record = dict(record)
        enriched_record["passage"] = row[text_col]
        for col in extra_cols:
            if col in corpus.columns:
                enriched_record[col] = row[col]
        enriched.append(enriched_record)
    if skipped:
        print(
            f"WARNING: skipped {skipped} dense results whose row ids exceed corpus size.",
            file=sys.stderr,
        )
    return enriched


def dense_search(
    query: str,
    top_k: int,
    dense: DenseIndex,
    *,
    corpus: pd.DataFrame | None = None,
    text_col: str | None = None,
    extra_cols: Sequence[str] = ("url", "source"),
) -> pd.DataFrame:
    if top_k <= 0:
        return pd.DataFrame(columns=["row_id", "score", "dense_score"])

    k = min(top_k, int(dense.index.ntotal))
    if k <= 0:
        return pd.DataFrame(columns=["row_id", "score", "dense_score", "passage"])

    query_embedding = dense.model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, indices = dense.index.search(query_embedding, k)
    records = []
    skipped = 0
    corpus_size = len(corpus) if corpus is not None else None

    for score, row_id in zip(scores[0], indices[0]):
        row_id = int(row_id)
        if row_id < 0:
            continue
        if corpus_size is not None and row_id >= corpus_size:
            skipped += 1
            continue
        records.append(
            {
                "row_id": row_id,
                "score": float(score),
                "dense_score": float(score),
            }
        )

    if skipped:
        print(
            f"WARNING: skipped {skipped} dense results whose FAISS positions exceed corpus size.",
            file=sys.stderr,
        )

    records = add_corpus_fields(records, corpus, text_col, extra_cols)
    return pd.DataFrame.from_records(records)
