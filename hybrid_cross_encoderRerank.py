from __future__ import annotations

import sys
import math
from sentence_transformers import CrossEncoder
from dataclasses import dataclass
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

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

def row_to_record(
    corpus: Corpus,
    row_id: int,
    *,
    bm25_score: float | None = None,
    rerank_score: float | None = None,
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
    if rerank_score is not None:
        record["score"] = float(rerank_score)
        record["rerank_score"] = float(rerank_score)
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

def hybrid_rerank_search(
    query: str,
    corpus: Corpus,
    bm25: BM25Index,
    dense: DenseIndex,
    *,
    alpha: float = 0.5,
    bm25_k: int = 100,
    dense_k: int = 100,
    top_k: int = 10,
    candidate_k: int = 50,
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
        entry["hybrid_score"] += (1.0 / (rrf_k + zero_rank + 1)) * (1 - alpha)
        entry["bm25_rank"] = zero_rank + 1
        entry["bm25_score"] = float(row.bm25_score)

    for zero_rank, row in enumerate(dense_results.itertuples(index=False), start=0):
        row_id = int(row.row_id)
        entry = fused.setdefault(row_id, {"row_id": row_id, "hybrid_score": 0.0})
        entry["hybrid_score"] += (1.0 / (rrf_k + zero_rank + 1)) * alpha
        entry["dense_rank"] = zero_rank + 1
        entry["dense_score"] = float(row.dense_score)

    #Rerank
    #Select Canđiate
    candidates = sorted(
        fused.values(),
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )[:candidate_k]
    #prepare input
    pairs = []
    for item in candidates:
        row_id = int(item["row_id"])
        text = corpus.frame.iloc[row_id][corpus.text_col]
        pairs.append((query, text))
    #Cross Encoder rerank
    rerank_scores = reranker.predict(pairs)
    for item, rerank_score in zip(candidates, rerank_scores):
        item["rerank_score"] = float(rerank_score)
    #Rerank_score
    final_ranked = sorted(
        candidates,
        key=lambda item: item["rerank_score"],
        reverse=True,
    )[:top_k]

    records = []
    for item in final_ranked:
        records.append(
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
        )
    return pd.DataFrame.from_records(records)

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / k

def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / len(relevant)

def reciprocal_rank(retrieved, relevant):
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved, relevant, k):
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(
        1.0 / math.log2(i + 1)
        for i in range(1, ideal_hits + 1)
    )
    if idcg == 0:
        return 0.0
    return dcg / idcg

#build query relevance judgments
def build_qrels(df):
    qrels = {}
    for row in df.itertuples(index=False):
        qrels.setdefault(row.query, {0})
        if row.is_selected == 1:
            del qrels[row.query]
            qrels.setdefault(row.query, set()).add(row.chunk_text)
    return qrels

def evaluate(
    validation_df,
    mode="hybrid",
    top_k=10,
):
    precisions = []
    recalls = []
    mrrs = []
    ndcgs = []

    qrels =  build_qrels(validation_df)
    print("Complete build relevant")
    valid_query = validation_df['query'].unique()
    for row in range(len(valid_query)):
        query = valid_query[row]
        relevant = qrels[query]
        if mode == "bm25":
            retrieved = bm25_results[query]
        elif mode == "dense":
            retrieved = dense_results[query]
        else:
            retrieved = hybrid_results[query]
        precisions.append(
            precision_at_k(retrieved, relevant, top_k)
        )

        recalls.append(
            recall_at_k(retrieved, relevant, top_k)
        )

        mrrs.append(
            reciprocal_rank(retrieved, relevant)
        )

        ndcgs.append(
            ndcg_at_k(retrieved, relevant, top_k)
        )
        print(f"Complete query {row}")
    return {
        f"Precision@{top_k}": sum(precisions) / len(precisions),
        f"Recall@{top_k}": sum(recalls) / len(recalls),
        f"MRR": sum(mrrs) / len(mrrs),
        f"nDCG@{top_k}": sum(ndcgs) / len(ndcgs),
    }