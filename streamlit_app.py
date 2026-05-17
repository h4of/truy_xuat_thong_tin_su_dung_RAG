from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEPENDENCY_HINT = "pip install streamlit pandas numpy faiss-cpu sentence-transformers pyarrow"
DEFAULT_RETRIEVAL_TOP_K = 100
DEFAULT_RAG_TOP_K = 5
FREE_RAG_MODELS = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "qwen/qwen3-32b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.3-8b-instruct:free",
    "deepseek/deepseek-r1:free",
    "openai/gpt-oss-20b:free",
    "openrouter/free",
]

try:
    import pandas as pd
    import streamlit as st

    from rag_engine import DEFAULT_MODEL, RagConfig, generate_rag_answer
    from search_engine import SearchCorpus, hybrid_search, load_bm25_index, load_faiss_index
except ImportError as exc:
    print(f"ERROR: Missing dependency: {exc.name}", file=sys.stderr)
    print(f"Install with: {DEPENDENCY_HINT}", file=sys.stderr)
    raise


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
    cwd = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent

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
            return candidate

    return (cwd / "outputs").resolve()


@st.cache_resource(show_spinner=False)
def load_search_stack(base_dir_text: str) -> tuple[SearchCorpus, Any, Any]:
    base_dir = Path(base_dir_text)
    corpus_path = base_dir / "corpus" / "corpus.parquet"
    require_file(corpus_path)

    try:
        frame = pd.read_parquet(corpus_path)
    except ImportError as exc:
        raise RuntimeError(
            "Unable to read parquet corpus because pandas could not find a parquet engine "
            f"(pyarrow or fastparquet). Install dependencies with: {DEPENDENCY_HINT}"
        ) from exc

    if "chunk_text" in frame.columns:
        text_col = "chunk_text"
    elif "text" in frame.columns:
        text_col = "text"
    else:
        raise ValueError(
            f"Corpus must contain either 'chunk_text' or 'text'. Available columns: {list(frame.columns)}"
        )

    corpus = SearchCorpus(frame=frame, text_col=text_col)
    bm25 = load_bm25_index(base_dir)
    dense = load_faiss_index(
        base_dir,
        corpus_size=len(frame),
        dependency_hint=DEPENDENCY_HINT,
    )
    return corpus, bm25, dense


def run_hybrid_search(
    query: str,
    corpus: SearchCorpus,
    bm25: Any,
    dense: Any,
    *,
    bm25_k: int,
    dense_k: int,
    rrf_k: int,
) -> pd.DataFrame:
    return hybrid_search(
        query,
        corpus,
        bm25,
        dense,
        bm25_k=bm25_k,
        dense_k=dense_k,
        top_k=DEFAULT_RETRIEVAL_TOP_K,
        rrf_k=rrf_k,
    )


def rows_for_rag(results: pd.DataFrame) -> list[dict[str, Any]]:
    contexts = []
    for rank, record in enumerate(results.head(DEFAULT_RAG_TOP_K).to_dict(orient="records"), start=1):
        item = dict(record)
        item["rank"] = rank
        contexts.append(item)
    return contexts


def generate_ai_overview(
    query: str,
    results: pd.DataFrame,
    *,
    models: list[str],
    temperature: float,
    max_tokens: int,
    max_context_chars: int,
) -> tuple[str, str, list[str]]:
    errors = []
    contexts = rows_for_rag(results)

    for model in unique_models(models):
        config = RagConfig(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_context_chars=max_context_chars,
            app_title="Streamlit Hybrid Search Demo",
        )
        try:
            rag_result = generate_rag_answer(query, contexts, config=config)
            return rag_result.answer, rag_result.model, errors
        except Exception as exc:
            errors.append(f"{model}: {exc}")

    raise RuntimeError("All RAG models failed. " + " | ".join(errors))


def unique_models(models: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for model in models:
        model = model.strip()
        if model and model not in seen:
            seen.add(model)
            ordered.append(model)
    return ordered


def fallback_models(selected_model: str, enabled_models: list[str], *, use_auto_fallback: bool) -> list[str]:
    if not use_auto_fallback:
        return [selected_model]
    return unique_models([selected_model, *enabled_models, "openrouter/free"])


def truncate_text(value: Any, max_chars: int = 260) -> str:
    text = re.sub(r"\s+", " ", "" if pd.isna(value) else str(value)).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def display_url(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    url = str(value).strip()
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.netloc:
        path = parsed.path.strip("/")
        if path:
            path = " > " + " > ".join(part for part in path.split("/")[:3] if part)
        return f"{parsed.netloc}{path}"
    return url


def result_title(row: dict[str, Any]) -> str:
    for col in ("title", "heading", "name"):
        value = row.get(col)
        if value is not None and not pd.isna(value) and str(value).strip():
            return truncate_text(value, max_chars=85)

    passage = truncate_text(row.get("passage"), max_chars=85)
    if passage:
        return passage

    url_text = display_url(row.get("url"))
    if url_text:
        return url_text
    return f"Result {row.get('row_id', '')}".strip()


def score_line(row: dict[str, Any]) -> str:
    pieces = []
    for label, key in (
        ("hybrid", "hybrid_score"),
        ("bm25", "bm25_score"),
        ("dense", "dense_score"),
    ):
        value = row.get(key)
        if value is not None and not pd.isna(value):
            pieces.append(f"{label}: {float(value):.6f}")
    for label, key in (("bm25 rank", "bm25_rank"), ("dense rank", "dense_rank")):
        value = row.get(key)
        if value is not None and not pd.isna(value):
            pieces.append(f"{label}: {int(value)}")
    return " | ".join(pieces)


def render_result(row: dict[str, Any], rank: int) -> None:
    url = row.get("url")
    source = row.get("source")
    url_text = display_url(url) or ("" if source is None or pd.isna(source) else str(source))
    title = html.escape(result_title(row))
    snippet = html.escape(truncate_text(row.get("passage"), max_chars=320))
    score_text = html.escape(score_line(row))
    safe_url_text = html.escape(url_text)
    href = html.escape(str(url)) if url is not None and not pd.isna(url) and str(url).strip() else "#"

    st.markdown(
        f"""
        <div class="search-result">
          <div class="result-rank">{rank}</div>
          <div class="result-body">
            <div class="result-url">{safe_url_text}</div>
            <a class="result-title" href="{href}" target="_blank">{title}</a>
            <div class="result-snippet">{snippet}</div>
            <div class="result-score">{score_text}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_overview(
    answer: str | None,
    *,
    error: str | None = None,
    model_used: str | None = None,
    fallback_notes: list[str] | None = None,
) -> None:
    if error:
        body = f'<div class="ai-error">{html.escape(error)}</div>'
    elif answer:
        model_badge = f'<div class="ai-model">Model: {html.escape(model_used)}</div>' if model_used else ""
        fallback_text = ""
        if fallback_notes:
            fallback_text = (
                '<details class="fallback-details"><summary>Fallback attempts</summary>'
                + "".join(f"<div>{html.escape(note)}</div>" for note in fallback_notes)
                + "</details>"
            )
        body = (
            f'<div class="ai-answer">{html.escape(answer).replace(chr(10), "<br>")}</div>'
            f"{model_badge}{fallback_text}"
        )
    else:
        body = '<div class="ai-muted">Run a search to generate an answer from the top 5 retrieved passages.</div>'

    st.markdown(
        f"""
        <section class="ai-card">
          <div class="ai-label">AI overview</div>
          {body}
        </section>
        """,
        unsafe_allow_html=True,
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>
          :root {
            --google-blue: #1a73e8;
            --text-primary: #202124;
            --text-secondary: #5f6368;
            --border-soft: #dadce0;
          }
          .block-container {
            max-width: 920px;
            padding-top: 1.05rem;
          }
          header[data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(8px);
          }
          section[data-testid="stSidebar"] {
            background: #f7f9fc;
            border-right: 1px solid #e8ebf0;
          }
          section[data-testid="stSidebar"] .block-container {
            padding: 1.15rem 1rem 1.4rem;
          }
          section[data-testid="stSidebar"] h3 {
            color: var(--text-primary);
            font-size: 17px;
            margin-bottom: 0.35rem;
          }
          section[data-testid="stSidebar"] label,
          section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--text-primary);
            font-size: 13px;
          }
          section[data-testid="stSidebar"] div[data-testid="stTextInput"] input,
          section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input,
          section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background: #fff;
            border: 1px solid #e3e7ee;
            border-radius: 10px;
            box-shadow: none;
            font-size: 14px;
            min-height: 38px;
            height: 38px;
            padding-left: 12px;
          }
          section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button {
            border-radius: 8px;
            height: 30px;
            width: 30px;
          }
          section[data-testid="stSidebar"] div[data-testid="stSlider"] {
            padding-top: 0.15rem;
            padding-bottom: 0.35rem;
          }
          section[data-testid="stSidebar"] div[data-testid="stSlider"] [role="slider"] {
            background: var(--google-blue);
          }
          section[data-testid="stSidebar"] div[data-testid="stExpander"] {
            background: #fff;
            border: 1px solid #edf0f5;
            border-radius: 12px;
            box-shadow: 0 1px 2px rgba(60, 64, 67, 0.04);
            margin-bottom: 10px;
          }
          section[data-testid="stSidebar"] div[data-testid="stExpander"] details {
            border-radius: 12px;
          }
          div[data-testid="stForm"] {
            border: 0;
            padding: 0;
          }
          div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
            align-items: center;
            gap: 12px;
          }
          div[data-testid="stForm"] div[data-testid="column"] {
            display: flex;
            flex-direction: column;
            justify-content: center;
          }
          div[data-testid="stForm"] div[data-testid="stTextInput"],
          div[data-testid="stForm"] div[data-testid="stButton"] {
            margin-bottom: 0;
          }
          div[data-testid="stForm"] div[data-testid="stTextInput"] > div {
            margin-bottom: 0;
          }
          div[data-testid="stForm"] div[data-testid="stTextInput"] div[data-baseweb="input"] {
            background: #fff;
            border-radius: 28px;
            border: 1px solid #dfe1e5;
            box-shadow: 0 1px 6px rgba(32, 33, 36, 0.16);
            display: flex;
            align-items: center;
            height: 52px;
            min-height: 52px;
            overflow: hidden;
          }
          div[data-testid="stForm"] div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
            border-color: #dfe1e5;
            box-shadow: 0 1px 8px rgba(32, 33, 36, 0.22);
          }
          div[data-testid="stForm"] div[data-testid="stTextInput"] div[data-baseweb="base-input"] {
            background: transparent;
            border: 0;
            height: 100%;
          }
          div[data-testid="stForm"] div[data-testid="stTextInput"] input {
            background: transparent;
            border: 0;
            box-shadow: none;
            font-size: 19px;
            height: 50px;
            line-height: 50px;
            min-height: 50px;
            padding-left: 22px;
            padding-right: 18px;
          }
          div[data-testid="stForm"] div[data-testid="stTextInput"] input:focus {
            border: 0;
            box-shadow: none;
            outline: none;
          }
          div[data-testid="stForm"] .stButton button {
            background: var(--google-blue);
            border: 1px solid var(--google-blue);
            border-radius: 27px;
            color: #fff;
            font-size: 15px;
            font-weight: 600;
            height: 52px;
            min-height: 52px;
            margin-top: 0;
            padding: 0 18px;
            width: 100%;
          }
          div[data-testid="stForm"] .stButton button:hover {
            background: #1558b0;
            border-color: #1558b0;
            color: #fff;
          }
          .app-brand {
            color: var(--text-secondary);
            font-size: 14px;
            margin-bottom: 10px;
          }
          .ai-card {
            border-bottom: 1px solid var(--border-soft);
            margin: 20px 0 22px;
            padding: 0 0 24px;
          }
          .ai-label {
            color: #0b57d0;
            font-size: 15px;
            font-weight: 650;
            margin-bottom: 14px;
          }
          .ai-answer {
            color: var(--text-primary);
            font-size: 21px;
            line-height: 1.48;
          }
          .ai-model {
            color: #70757a;
            font-size: 12px;
            margin-top: 10px;
          }
          .fallback-details {
            color: #70757a;
            font-size: 12px;
            margin-top: 8px;
          }
          .fallback-details summary {
            cursor: pointer;
          }
          .ai-muted {
            color: var(--text-secondary);
            font-size: 16px;
            line-height: 1.5;
          }
          .ai-error {
            background: #fce8e6;
            border: 1px solid #fad2cf;
            border-radius: 10px;
            color: #a50e0e;
            font-size: 15px;
            line-height: 1.45;
            padding: 12px 14px;
          }
          .result-count {
            color: #70757a;
            font-size: 14px;
            margin: 4px 0 18px;
          }
          .search-result {
            display: flex;
            gap: 14px;
            margin: 0 0 28px;
          }
          .result-rank {
            align-items: center;
            background: #f1f3f4;
            border-radius: 50%;
            color: var(--text-secondary);
            display: flex;
            flex: 0 0 34px;
            font-size: 13px;
            height: 34px;
            justify-content: center;
            margin-top: 2px;
            width: 34px;
          }
          .result-url {
            color: var(--text-primary);
            font-size: 14px;
            line-height: 1.35;
            margin-bottom: 2px;
          }
          .result-title {
            color: #1a0dab !important;
            display: block;
            font-size: 21px;
            line-height: 1.3;
            margin-bottom: 4px;
            text-decoration: none;
          }
          .result-title:hover {
            text-decoration: underline;
          }
          .result-snippet {
            color: #4d5156;
            font-size: 15px;
            line-height: 1.55;
          }
          .result-score {
            color: #70757a;
            font-size: 12px;
            line-height: 1.4;
            margin-top: 6px;
          }
          .stButton button {
            border-radius: 22px;
          }
          @media (max-width: 640px) {
            .block-container {
              padding-left: 1rem;
              padding-right: 1rem;
            }
            div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
              gap: 8px;
            }
            div[data-testid="stForm"] div[data-testid="stTextInput"] div[data-baseweb="input"],
            div[data-testid="stForm"] .stButton button {
              height: 48px;
              min-height: 48px;
            }
            div[data-testid="stForm"] div[data-testid="stTextInput"] input {
              height: 46px;
              line-height: 46px;
              min-height: 46px;
            }
            .ai-answer {
              font-size: 18px;
            }
            .result-title {
              font-size: 18px;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Hybrid Search Demo",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    inject_css()

    with st.sidebar:
        st.subheader("Settings")
        with st.expander("Retrieval", expanded=True):
            base_dir_input = st.text_input("Artifact base directory", value="", key="artifact_base_dir")
            bm25_k = st.number_input("BM25 candidates", min_value=1, max_value=1000, value=100, step=10)
            dense_k = st.number_input("Dense candidates", min_value=1, max_value=1000, value=100, step=10)
            rrf_k = st.number_input("RRF k", min_value=0, max_value=500, value=60, step=5)
        with st.expander("RAG", expanded=True):
            default_model_index = FREE_RAG_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in FREE_RAG_MODELS else 0
            selected_model = st.selectbox(
                "Primary model",
                options=FREE_RAG_MODELS,
                index=default_model_index,
                help="The app tries this model first for the AI overview.",
            )
            use_auto_fallback = st.checkbox(
                "Try fallback models if the primary model fails",
                value=True,
            )
            fallback_pool = st.multiselect(
                "Fallback order",
                options=FREE_RAG_MODELS,
                default=[
                    model_id
                    for model_id in FREE_RAG_MODELS
                    if model_id not in {selected_model, "openrouter/free"}
                ][:4],
                disabled=not use_auto_fallback,
            )
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
            max_tokens = st.number_input("Max tokens", min_value=64, max_value=4096, value=700, step=64)
            max_context_chars = st.number_input(
                "Chars per context",
                min_value=200,
                max_value=5000,
                value=2000,
                step=100,
            )

    st.markdown('<div class="app-brand">Hybrid BM25 + FAISS search demo</div>', unsafe_allow_html=True)

    with st.form("search-form"):
        default_query = st.session_state.get("query", "")
        search_col, button_col = st.columns([0.86, 0.14], gap="small")
        with search_col:
            query = st.text_input(
                "Search query",
                value=default_query,
                label_visibility="collapsed",
                placeholder="Search MS MARCO passages",
                key="main_search_query_input",
            )
        with button_col:
            submitted = st.form_submit_button("Search", use_container_width=True)

    if submitted:
        st.session_state["query"] = query.strip()

    active_query = st.session_state.get("query", "").strip()
    if not active_query:
        render_ai_overview(None)
        return

    base_dir = resolve_base_dir(base_dir_input.strip() or None)

    try:
        with st.spinner("Loading corpus and indexes..."):
            corpus, bm25, dense = load_search_stack(str(base_dir))
    except Exception as exc:
        st.error(str(exc))
        st.caption(f"Dependency hint: {DEPENDENCY_HINT}")
        return

    try:
        with st.spinner("Searching hybrid index..."):
            results = run_hybrid_search(
                active_query,
                corpus,
                bm25,
                dense,
                bm25_k=int(bm25_k),
                dense_k=int(dense_k),
                rrf_k=int(rrf_k),
            )
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        return

    rag_answer = None
    rag_error = None
    rag_model_used = None
    fallback_notes: list[str] = []
    if results.empty:
        rag_error = "No retrieved passages were found, so the RAG answer was not generated."
    else:
        try:
            with st.spinner("Generating AI overview from top 5 passages..."):
                model_candidates = fallback_models(
                    selected_model,
                    fallback_pool,
                    use_auto_fallback=use_auto_fallback,
                )
                rag_answer, rag_model_used, fallback_notes = generate_ai_overview(
                    active_query,
                    results,
                    models=model_candidates,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    max_context_chars=int(max_context_chars),
                )
        except Exception as exc:
            rag_error = str(exc)

    render_ai_overview(
        rag_answer,
        error=rag_error,
        model_used=rag_model_used,
        fallback_notes=fallback_notes,
    )
    st.markdown(
        f'<div class="result-count">About {len(results):,} results from hybrid search | RAG used top {min(DEFAULT_RAG_TOP_K, len(results))}</div>',
        unsafe_allow_html=True,
    )

    for rank, row in enumerate(results.to_dict(orient="records"), start=1):
        render_result(row, rank)


if __name__ == "__main__":
    main()
