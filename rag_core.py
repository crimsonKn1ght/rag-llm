
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_core.py — Core RAG logic (GPU-aware, Streamlit-agnostic)
============================================================
Provides reusable functions to build a RAG pipeline with LangChain + Chroma.

Sections
--------
[0] Optional .env loading
[1] Imports
[2] Prompt
[3] Embeddings factory (OpenAI or SBERT w/ GPU)
[4] Document loading (URL, local file paths)
[5] Vector store & retriever (Chroma)
[6] RAG chain assembly (LCEL)
[7] Orchestrator helpers
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# [0] Optional: .env loading
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# [1] Imports for the RAG system
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

# Optional local-embedding backend (GPU-capable)
try:
    import torch  # for device detection
except Exception:
    torch = None

try:
    from sentence_transformers import SentenceTransformer  # optional
except Exception:
    SentenceTransformer = None


# [2] Prompt used by the chain (concise; grounded answers only)
DEFAULT_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Use ONLY the provided context to answer.\n"
    "If the answer is not in the context, say you don't know.\n\n"
    "Question:\n{question}\n\n"
    "Context:\n{context}\n\n"
    "Answer:"
)


# [3] Embedding backend factory (GPU-aware)
def build_embedding_fn(
    backend: str = "openai",
    sbert_model: Optional[str] = None,
):
    """
    Returns an object implementing LangChain's Embeddings interface.

    backend:
        - 'openai': OpenAIEmbeddings (requires OPENAI_API_KEY)
        - 'sbert' : SentenceTransformers (can run on GPU if available)
    """
    backend = backend.lower().strip()

    if backend == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set. Provide it or use backend='sbert'.")
        return OpenAIEmbeddings()

    if backend == "sbert":
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
        model_name = sbert_model or "sentence-transformers/all-MiniLM-L6-v2"
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        st_model = SentenceTransformer(model_name, device=device)

        class SBERTEmbeddings:
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return st_model.encode(texts, convert_to_numpy=False, batch_size=32, show_progress_bar=False).tolist()

            def embed_query(self, text: str) -> List[float]:
                return st_model.encode([text], convert_to_numpy=False, show_progress_bar=False)[0].tolist()
        return SBERTEmbeddings()

    raise ValueError(f"Unknown embedding backend: {backend!r}")


# [4] Document loading (URL, local file paths)
def load_documents_from_url(url: str):
    """Load a single web page as LangChain Documents."""
    loader = WebBaseLoader(url)
    return loader.load()


def load_documents_from_paths(paths: Sequence[Path]):
    """
    Load documents from file paths.
    - .txt / .md handled by TextLoader
    - .pdf handled by PyPDFLoader
    """
    docs = []
    for p in paths:
        suffix = p.suffix.lower()
        if suffix in [".txt", ".md"]:
            docs.extend(TextLoader(str(p), autodetect_encoding=True).load())
        elif suffix == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
        else:
            # Skip unsupported types silently; caller can filter beforehand.
            continue
    return docs


# [5] Build vector store & retriever (Chroma)
def build_vectorstore_and_retriever(
    docs,
    embeddings,
    persist_dir: str,
    k: int = 4,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Create a Chroma store and return a retriever.
    Persists to `persist_dir` so you can reuse across runs.
    """
    os.makedirs(persist_dir, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return vectorstore, retriever, chunks


# [6] Compose the RAG chain (LCEL style)
def build_rag_chain(retriever, prompt=DEFAULT_PROMPT, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    """
    RAG = retrieve context -> prompt LLM -> parse text output
    """
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {
            "context": retriever | (lambda docs: format_docs(docs)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# [7] Orchestrator helpers
def run_rag_with_docs(
    docs,
    question: str,
    embeddings_backend: str = "openai",
    sbert_model: Optional[str] = None,
    persist_dir: str = "./rag_db",
    k: int = 4,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    High-level convenience wrapper: build embeddings, store, retriever, chain; return answer + retrieved chunks.
    """
    embeddings = build_embedding_fn(backend=embeddings_backend, sbert_model=sbert_model)
    _, retriever, chunks = build_vectorstore_and_retriever(
        docs=docs,
        embeddings=embeddings,
        persist_dir=persist_dir,
        k=k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chain = build_rag_chain(retriever, model_name=model_name, temperature=temperature)
    answer = chain.invoke(question)
    return answer, chunks
