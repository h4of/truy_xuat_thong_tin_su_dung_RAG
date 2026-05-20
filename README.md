# Information Retrieval with RAG

This repository contains a small MS MARCO v1.1 retrieval stack with:

- BM25 retrieval over prebuilt raw statistics
- Dense semantic search with FAISS
- Hybrid retrieval with Reciprocal Rank Fusion
- Cross-encoder reranking
- A lightweight RAG layer backed by OpenRouter
- A Streamlit demo for search plus answer generation

## Project Layout

```text
corpus/
  corpus.parquet

embeddings/
  corpus_embeddings.npy

indexes/
  faiss_index.index
  bm25/
    vocab.json
    metadata.json
    df.npy
    doc_len.npy
    avgdl.npy
    indptr.npy
    posting_doc_ids.npy
    posting_tf.npy

search_engine/
  bm25_retrieval.py
  faiss_semantic_search.py
  hybrid_search.py
  hybrid_rerank_search.py

rag_engine/
  rag.py

retrieval_test.py
rag_test.py
streamlit_app.py
```

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The demo and retrieval scripts expect these local artifacts to already exist.
If your artifacts are stored under `outputs/`, the scripts will use that path.
If they are stored at the repo root, the scripts will fall back to the root
layout used in this workspace.

## Retrieval Scripts

Test BM25, dense, and hybrid retrieval:

```bash
python retrieval_test.py --query "what is a corporation" --mode bm25 --top-k 5
python retrieval_test.py --query "what is a corporation" --mode dense --top-k 5
python retrieval_test.py --query "what is a corporation" --mode hybrid --top-k 5
```

The reusable retrieval modules live in `search_engine/`:

- `search_engine.bm25_retrieval`
- `search_engine.faiss_semantic_search`
- `search_engine.hybrid_search`
- `search_engine.hybrid_rerank_search`

## RAG Test

Run hybrid retrieval with top 5 contexts and send them to the RAG layer:

```bash
python rag_test.py --query "what is a corporation"
```

The RAG module uses OpenRouter and defaults to:

```text
google/gemma-4-31b-it:free
```

API key lookup order:

1. `--api-key`
2. `OPENROUTER_API_KEY`
3. `API_KEY`
4. `.env`

## Streamlit Demo

Launch the demo app:

```bash
python -m streamlit run streamlit_app.py
```

The app:

- runs hybrid search with `top_k=100`
- sends only the top 5 results into RAG
- shows an answer panel at the top
- renders the results underneath with URL and score details

## Notes

- The repository does not rebuild indexes or embeddings at runtime.
- The retrieval code assumes the corpus and index files already exist.
- If Parquet loading fails, install `pyarrow`:

```bash
pip install pyarrow
```

## Index Download

If you need the packaged index artifacts, the original shared archive is here:

[Index archive](https://drive.google.com/file/d/17rtLlQI6K-kcaLBTAR-TT4eyVS5u-ewX/view?usp=sharing)
