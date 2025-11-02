#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

# === [0] Environment setup ===
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Enable LangSmith tracing if available
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# === [1] Core Imports ===
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.load import dumps, loads

# Optional GPU + SBERT support
try:
    import torch
except Exception:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


# === [2] Default Prompt ===
DEFAULT_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Use ONLY the provided context to answer.\n"
    "If the answer is not in the context, say you don't know.\n\n"
    "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:"
)


# === [3] Embeddings Factory (OpenAI or SBERT) ===
def build_embedding_fn(backend: str = "openai", sbert_model: Optional[str] = None):
    backend = backend.lower().strip()

    if backend == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set.")
        return OpenAIEmbeddings()

    if backend == "sbert":
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed.")
        model_name = sbert_model or "sentence-transformers/all-MiniLM-L6-v2"
        device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        st_model = SentenceTransformer(model_name, device=device)

        class SBERTEmbeddings:
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return st_model.encode(texts, convert_to_numpy=False, batch_size=32, show_progress_bar=False).tolist()

            def embed_query(self, text: str) -> List[float]:
                return st_model.encode([text], convert_to_numpy=False, show_progress_bar=False)[0].tolist()

        return SBERTEmbeddings()

    raise ValueError(f"Unknown embedding backend: {backend!r}")


# === [4] Document Loading ===
def load_docs_from_url(url: str):
    loader = WebBaseLoader(url)
    return loader.load()


def load_docs_from_paths(paths: Sequence[Path]):
    docs = []
    for p in paths:
        suffix = p.suffix.lower()
        if suffix in [".txt", ".md"]:
            docs.extend(TextLoader(str(p), autodetect_encoding=True).load())
        elif suffix == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
    return docs


# === [5] Vector Store & Retriever ===
def build_vectorstore_and_retriever(
    docs,
    embeddings,
    persist_dir: str = "./rag_db",
    k: int = 4,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    os.makedirs(persist_dir, exist_ok=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return vectordb, retriever, chunks


# === [6] Utility Functions ===
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def get_unique_union(document_lists: list[list]):
    """Flatten + deduplicate documents."""
    flat = [dumps(doc) for sublist in document_lists for doc in sublist]
    return [loads(doc) for doc in set(flat)]


def reciprocal_rank_fusion(results: list[list], k=60):
    """Fuse multiple ranked lists."""
    fused = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            d_str = dumps(doc)
            fused[d_str] = fused.get(d_str, 0) + 1 / (rank + k)
    reranked = [(loads(doc), score) for doc, score in sorted(fused.items(), key=lambda x: x[1], reverse=True)]
    return [d for d, _ in reranked]


# === [7] Retrieval Modes ===
def build_multiquery_chain(retriever):
    prompt = ChatPromptTemplate.from_template(
        """Generate 5 alternative formulations of this user question to help retrieve more relevant documents.
        Original question: {question}"""
    )
    llm = ChatOpenAI(temperature=0)
    generate_queries = prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    return retrieval_chain


def build_ragfusion_chain(retriever):
    prompt = ChatPromptTemplate.from_template("Generate 4 related search queries for: {question}")
    llm = ChatOpenAI(temperature=0)
    generate_queries = prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion
    return retrieval_chain


def build_hyde_chain(retriever):
    prompt = ChatPromptTemplate.from_template(
        "Please write a passage that answers the question.\nQuestion: {question}\nPassage:"
    )
    llm = ChatOpenAI(temperature=0)
    generate_docs = prompt | llm | StrOutputParser()
    retrieval_chain = generate_docs | retriever
    return retrieval_chain


# === [8] Build RAG Chain (LCEL Style) ===
def build_rag_chain(retrieval_chain, model_name: str = "gpt-4o-mini", temperature: float = 0.0, prompt=DEFAULT_PROMPT):
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    chain = (
        {"context": retrieval_chain | (lambda docs: format_docs(docs)), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# === [9] Orchestrator ===
def run_rag(
    docs,
    question: str,
    mode: str = "basic",  # 'basic', 'multiquery', 'fusion', 'hyde'
    embeddings_backend: str = "openai",
    sbert_model: Optional[str] = None,
    persist_dir: str = "./rag_db",
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    k: int = 4,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    High-level unified entrypoint for RAG and Advanced RAG.
    """

    # Build embeddings + retriever
    embeddings = build_embedding_fn(backend=embeddings_backend, sbert_model=sbert_model)
    _, retriever, _ = build_vectorstore_and_retriever(
        docs, embeddings, persist_dir, k=k, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Choose retrieval strategy
    if mode == "basic":
        retrieval_chain = retriever
    elif mode == "multiquery":
        retrieval_chain = build_multiquery_chain(retriever)
    elif mode == "fusion":
        retrieval_chain = build_ragfusion_chain(retriever)
    elif mode == "hyde":
        retrieval_chain = build_hyde_chain(retriever)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Compose full RAG chain
    chain = build_rag_chain(retrieval_chain, model_name=model_name, temperature=temperature)
    answer = chain.invoke({"question": question})
    return answer


# === [10] Example Usage ===
if __name__ == "__main__":
    # Example (replace with your own docs)
    docs = load_docs_from_url("https://en.wikipedia.org/wiki/Artificial_intelligence")
    q = "What are the main goals of AI research?"
    print(run_rag(docs, q, mode="fusion"))  
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
    Backward-compatible wrapper for Streamlit app.
    Returns (answer, chunks) just like the old core version.
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
    answer = chain.invoke({"question": question})
    return answer, chunks
