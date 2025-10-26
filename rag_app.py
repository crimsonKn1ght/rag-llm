
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_app.py — Streamlit UI for the RAG pipeline (Full-featured)
==============================================================
Features
--------
- Sidebar config: API key, backend (OpenAI/SBERT), model, k, temperature, chunking, persist dir.
- Source selection: URL input and/or multiple file uploads (.txt, .md, .pdf).
- GPU awareness for SBERT (uses CUDA if available).
- Displays retrieved chunk previews and the final grounded answer.

Run
---
    streamlit run rag_app.py

Dependencies
------------
    pip install streamlit langchain langchain-community langchain-openai chromadb tiktoken python-dotenv
    # Optional (for SBERT + GPU):
    pip install sentence-transformers torch --extra-index-url https://download.pytorch.org/whl/cu121
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import streamlit as st
from rag_core import (
    load_documents_from_url,
    load_documents_from_paths,
    run_rag_with_docs,
)

# -----------------------------
# [UI] Page & Sidebar Settings
# -----------------------------
st.set_page_config(page_title="RAG From Scratch (Streamlit)", page_icon="📚", layout="wide")
st.title("📚 RAG From Scratch — Full App")

with st.sidebar:
    st.header("⚙️ Configuration")

    # API key input (only needed for OpenAI backend)
    openai_key = st.text_input("OpenAI API Key", type="password", help="Required if you choose OpenAI embeddings/LLM.")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    backend = st.selectbox("Embedding Backend", ["openai", "sbert"], index=0,
                           help="Use OpenAI API or local SentenceTransformers (GPU-capable).")

    sbert_model = None
    if backend == "sbert":
        sbert_model = st.text_input("SBERT Model ID", value="sentence-transformers/all-MiniLM-L6-v2")

    model_name = st.text_input("Chat Model", value="gpt-4o-mini")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    top_k = st.slider("Top-k Retrieved Chunks", 1, 10, 4, 1)

    st.subheader("Chunking")
    chunk_size = st.number_input("Chunk Size", min_value=200, max_value=4000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1000, value=200, step=50)

    st.subheader("Persistence")
    persist_dir = st.text_input("Chroma Persist Directory", value="./rag_db", help="Directory to store/reuse vectors.")

# -----------------------------
# [UI] Data Source Selection
# -----------------------------
st.subheader("📥 Source Documents")
source_mode = st.radio("Choose source:", ["URL", "Files", "URL + Files"], horizontal=True)

url = ""
if source_mode in ["URL", "URL + Files"]:
    url = st.text_input("Web URL", placeholder="https://example.com/article")

uploaded_files = []
if source_mode in ["Files", "URL + Files"]:
    uploaded_files = st.file_uploader(
        "Upload .txt, .md, .pdf files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True
    )

# -----------------------------
# [UI] Question Input
# -----------------------------
st.subheader("❓ Ask a Question")
question = st.text_input("Your question", placeholder="What are the key ideas?")

# Run button
run_clicked = st.button("🚀 Run RAG", type="primary")

# -----------------------------
# [Controller] Orchestrate a Run
# -----------------------------
def save_uploaded_files(files: List, dest_dir: Path) -> List[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for f in files:
        # Sanitize filename and save
        safe_name = f.name.replace("/", "_")
        out_path = dest_dir / safe_name
        with open(out_path, "wb") as w:
            w.write(f.getbuffer())
        saved_paths.append(out_path)
    return saved_paths


def validate_inputs():
    if backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not provided. Enter it in the sidebar or switch to SBERT backend.")
        return False
    if source_mode in ["URL", "URL + Files"] and not url:
        st.error("Please provide a URL or switch source mode.")
        return False
    if source_mode in ["Files", "URL + Files"] and not uploaded_files:
        st.error("Please upload at least one file or switch source mode.")
        return False
    if not question.strip():
        st.error("Please type a question.")
        return False
    return True


if run_clicked:
    if not validate_inputs():
        st.stop()

    # Assemble documents
    docs = []

    with st.spinner("Loading documents..."):
        if source_mode in ["URL", "URL + Files"] and url:
            try:
                docs.extend(load_documents_from_url(url))
            except Exception as e:
                st.error(f"Failed to load URL: {e}")
                st.stop()

        if source_mode in ["Files", "URL + Files"] and uploaded_files:
            tmp_dir = Path("./_uploads")
            saved_paths = save_uploaded_files(uploaded_files, tmp_dir)
            try:
                docs.extend(load_documents_from_paths(saved_paths))
            except Exception as e:
                st.error(f"Failed to load uploaded files: {e}")
                st.stop()

    if not docs:
        st.error("No documents loaded. Please check your inputs.")
        st.stop()

    # Run RAG
    with st.spinner("Building vector store and generating answer..."):
        try:
            answer, chunks = run_rag_with_docs(
                docs=docs,
                question=question,
                embeddings_backend=backend,
                sbert_model=sbert_model,
                persist_dir=persist_dir,
                k=top_k,
                model_name=model_name,
                temperature=temperature,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception as e:
            st.error(f"Error during RAG run: {e}")
            st.stop()

    # -----------------------------
    # [UI] Results
    # -----------------------------
    st.success("Done!")

    st.markdown("### 🧠 Answer")
    st.markdown(f"> {answer}")

    with st.expander("📄 Retrieved Chunk Previews"):
        # Show up to top_k chunks (if available) to keep UI tidy
        shown = 0
        for i, d in enumerate(chunks):
            if shown >= top_k:
                break
            # A crude heuristic: show the first 'top_k' chunks; in production, you'd display *the actual retrieved* docs.
            st.markdown(f"**Chunk {i+1}:**")
            st.code((d.page_content[:1200] + ("..." if len(d.page_content) > 1200 else "")))
            shown += 1

    with st.expander("ℹ️ Run Metadata"):
        import platform
        cuda = False
        try:
            import torch  # noqa
            cuda = torch.cuda.is_available()
        except Exception:
            pass

        st.write({
            "Embedding backend": backend,
            "SBERT model": sbert_model if backend == "sbert" else None,
            "Chat model": model_name,
            "Top-k": top_k,
            "Chunk size": chunk_size,
            "Chunk overlap": chunk_overlap,
            "Persist dir": persist_dir,
            "CUDA available": cuda,
            "Python": platform.python_version(),
        })
