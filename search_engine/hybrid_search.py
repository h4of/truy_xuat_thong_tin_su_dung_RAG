from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .bm25_retrieval import BM25Index, bm25_search
from .faiss_semantic_search import DenseIndex, dense_search


@dataclass(frozen=True)
class SearchCorpus:
    frame: pd.DataFrame
    text_col: str


def row_to_record(
    corpus: SearchCorpus,
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
    corpus: SearchCorpus,
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
    records = [
        row_to_record(
            corpus,
            int(item["row_id"]),
            hybrid_score=float(item["hybrid_score"]),
            bm25_score=item.get("bm25_score"),
            dense_score=item.get("dense_score"),
            bm25_rank=item.get("bm25_rank"),
            dense_rank=item.get("dense_rank"),
        )
        for item in ranked
    ]
    return pd.DataFrame.from_records(records)
