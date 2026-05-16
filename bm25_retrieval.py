from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

TOKEN_RE = re.compile(r"\b\w+\b")


@dataclass
class BM25Index:
    vocab: dict[str, int]
    metadata: dict[str, Any]
    doc_freq: np.ndarray
    doc_len: np.ndarray
    avgdl: float
    indptr: np.ndarray
    posting_doc_ids: np.ndarray
    posting_tf: np.ndarray
    num_docs: int


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_bm25_index(base_dir: str | Path) -> BM25Index:
    base_path = Path(base_dir)
    bm25_dir = base_path / "indexes" / "bm25"
    paths = {
        "vocab": bm25_dir / "vocab.json",
        "metadata": bm25_dir / "metadata.json",
        "df": bm25_dir / "df.npy",
        "doc_len": bm25_dir / "doc_len.npy",
        "avgdl": bm25_dir / "avgdl.npy",
        "indptr": bm25_dir / "indptr.npy",
        "posting_doc_ids": bm25_dir / "posting_doc_ids.npy",
        "posting_tf": bm25_dir / "posting_tf.npy",
    }
    for path in paths.values():
        require_file(path)

    with paths["vocab"].open("r", encoding="utf-8") as handle:
        raw_vocab = json.load(handle)
    vocab = {str(token): int(term_id) for token, term_id in raw_vocab.items()}

    with paths["metadata"].open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    doc_freq = np.load(paths["df"], mmap_mode="r")
    doc_len = np.load(paths["doc_len"], mmap_mode="r")
    avgdl_arr = np.load(paths["avgdl"])
    avgdl = float(np.asarray(avgdl_arr).reshape(-1)[0])
    indptr = np.load(paths["indptr"], mmap_mode="r")
    posting_doc_ids = np.load(paths["posting_doc_ids"], mmap_mode="r")
    posting_tf = np.load(paths["posting_tf"], mmap_mode="r")

    if len(posting_doc_ids) != len(posting_tf):
        raise ValueError(
            "BM25 posting arrays have different lengths: "
            f"posting_doc_ids={len(posting_doc_ids):,}, posting_tf={len(posting_tf):,}"
        )
    if len(indptr) != len(doc_freq) + 1:
        raise ValueError(
            f"BM25 indptr length must be len(df)+1, got indptr={len(indptr):,}, df={len(doc_freq):,}"
        )
    if avgdl <= 0:
        raise ValueError(f"BM25 avgdl must be positive, got {avgdl}")

    num_docs = int(metadata.get("num_docs", len(doc_len)))
    if num_docs != len(doc_len):
        print(
            f"WARNING: BM25 metadata num_docs ({num_docs:,}) differs from doc_len size ({len(doc_len):,}).",
            file=sys.stderr,
        )

    print(
        "Loaded BM25 index: "
        f"{num_docs:,} docs, {len(vocab):,} terms, {len(posting_doc_ids):,} postings"
    )

    return BM25Index(
        vocab=vocab,
        metadata=metadata,
        doc_freq=doc_freq,
        doc_len=doc_len,
        avgdl=avgdl,
        indptr=indptr,
        posting_doc_ids=posting_doc_ids,
        posting_tf=posting_tf,
        num_docs=num_docs,
    )


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
            f"WARNING: skipped {skipped} BM25 results whose row ids exceed corpus size.",
            file=sys.stderr,
        )
    return enriched


def bm25_search(
    query: str,
    top_k: int,
    bm25: BM25Index,
    *,
    corpus: pd.DataFrame | None = None,
    text_col: str | None = None,
    extra_cols: Sequence[str] = ("url", "source"),
    k1: float = 1.5,
    b: float = 0.75,
) -> pd.DataFrame:
    if top_k <= 0:
        return pd.DataFrame(columns=["row_id", "score", "bm25_score"])

    tokens = tokenize(query)
    term_ids = [bm25.vocab[token] for token in tokens if token in bm25.vocab]
    if not term_ids:
        print(
            f"BM25: no query tokens found in vocabulary. Query tokens: {tokens}",
            file=sys.stderr,
        )
        return pd.DataFrame(columns=["row_id", "score", "bm25_score", "passage"])

    scores = np.zeros(bm25.num_docs, dtype=np.float32)
    for term_id in term_ids:
        if term_id < 0 or term_id >= len(bm25.doc_freq):
            print(f"WARNING: term_id out of range in vocab: {term_id}", file=sys.stderr)
            continue

        df = int(bm25.doc_freq[term_id])
        if df <= 0:
            continue

        start = int(bm25.indptr[term_id])
        end = int(bm25.indptr[term_id + 1])
        if start == end:
            continue

        doc_ids = bm25.posting_doc_ids[start:end].astype(np.int64, copy=False)
        valid_mask = (doc_ids >= 0) & (doc_ids < bm25.num_docs) & (doc_ids < len(bm25.doc_len))
        if not np.all(valid_mask):
            print(
                f"WARNING: posting list for term_id={term_id} contains doc ids outside BM25 doc range.",
                file=sys.stderr,
            )
            doc_ids = doc_ids[valid_mask]
            tf_values = bm25.posting_tf[start:end][valid_mask].astype(np.float32, copy=False)
        else:
            tf_values = bm25.posting_tf[start:end].astype(np.float32, copy=False)

        doc_lengths = bm25.doc_len[doc_ids].astype(np.float32, copy=False)
        idf = math.log(1.0 + (bm25.num_docs - df + 0.5) / (df + 0.5))
        denom = tf_values + k1 * (1.0 - b + b * doc_lengths / bm25.avgdl)
        scores[doc_ids] += idf * tf_values * (k1 + 1.0) / denom

    positive_ids = np.flatnonzero(scores > 0)
    if len(positive_ids) == 0:
        return pd.DataFrame(columns=["row_id", "score", "bm25_score", "passage"])

    limit = min(top_k, len(positive_ids))
    positive_scores = scores[positive_ids]
    if limit < len(positive_ids):
        top_local = np.argpartition(-positive_scores, limit - 1)[:limit]
    else:
        top_local = np.arange(len(positive_ids))
    top_local = top_local[np.argsort(-positive_scores[top_local])]
    top_doc_ids = positive_ids[top_local]

    records = [
        {
            "row_id": int(row_id),
            "score": float(scores[row_id]),
            "bm25_score": float(scores[row_id]),
        }
        for row_id in top_doc_ids
    ]
    records = add_corpus_fields(records, corpus, text_col, extra_cols)
    return pd.DataFrame.from_records(records)
