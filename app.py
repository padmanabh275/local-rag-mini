"""
Streamlit app for the local RAG pipeline.
Shows top-k retrieved chunks (with source filename) and the grounded answer.
"""

import streamlit as st
from pathlib import Path

from query import (
    SNIPPET_LEN,
    load_index_and_meta,
    retrieve,
    generate_answer,
)

DATA_DIR = "data"
DEFAULT_K = 3


@st.cache_resource
def get_rag_components():
    """Load index, meta, and model once and cache."""
    return load_index_and_meta(DATA_DIR)


def main():
    st.set_page_config(page_title="Local RAG", page_icon="📄", layout="centered")
    st.title("Local RAG")
    st.caption("Ask a question and get top chunks + a grounded answer from your documents.")

    data_dir = Path(DATA_DIR)
    if not (data_dir / "meta.json").exists() or not (data_dir / "embeddings.npy").exists():
        st.warning(
            "Index not found. Run `python ingest.py` first to build the index from the `docs/` folder."
        )
        st.stop()

    question = st.text_input("Question", placeholder="e.g. How do I build the index?")
    k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=DEFAULT_K)

    if not question or not question.strip():
        st.info("Enter a question above to see retrieved chunks and the answer.")
        return

    with st.spinner("Loading index and model..."):
        try:
            embeddings, meta, model = get_rag_components()
        except Exception as e:
            st.error(f"Failed to load index: {e}")
            return

    with st.spinner("Retrieving..."):
        chunks = retrieve(question, embeddings, meta, model, k=k)
        answer = generate_answer(question, chunks)

    st.subheader("Top retrieved chunks")
    for c in chunks:
        snippet = c["text"]
        if len(snippet) > SNIPPET_LEN:
            snippet = snippet[:SNIPPET_LEN].rsplit(" ", 1)[0] + "..."
        with st.expander(f"**[{c['rank']}] {c['source']}**"):
            st.text(snippet)
            st.caption(f"Distance: {c['distance']:.4f}")

    st.subheader("Answer")
    st.info(answer)
    st.caption("Answers are built only from the retrieved chunks (no hallucination).")


if __name__ == "__main__":
    main()
