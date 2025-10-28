#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_core_advanced.py — Advanced RAG Logic (LangSmith + MultiQuery + Fusion + HyDE)
==================================================================================
Extends base RAG to include multi-query retrieval, RAG-fusion, decomposition, 
step-back prompting, and hypothetical document embeddings (HyDE).
"""

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

# Enable LangSmith tracing if API key exists
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# === [1] Core Imports ===
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.load import dumps, loads

# === [2] Embedding Backend ===
def build_embeddings():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing.")
    return OpenAIEmbeddings()

# === [3] Document Loading ===
def load_docs_from_url(url: str):
    loader = WebBaseLoader(url)
    return loader.load()

def load_docs_from_paths(paths: Sequence[Path]):
    docs = []
    for p in paths:
        if p.suffix in [".txt", ".md"]:
            docs.extend(TextLoader(str(p)).load())
        elif p.suffix == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
    return docs

# === [4] Vector Store & Retriever ===
def build_vectorstore(docs, persist_dir="./rag_db", chunk_size=1000, chunk_overlap=200):
    os.makedirs(persist_dir, exist_ok=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    embeddings = build_embeddings()
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    return retriever

# === [5] Utility Functions ===
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def get_unique_union(documents: list[list]):
    """Flatten + deduplicate documents."""
    flat = [dumps(doc) for sublist in documents for doc in sublist]
    return [loads(doc) for doc in set(flat)]

def reciprocal_rank_fusion(results: list[list], k=60):
    """Fuse multiple ranked lists."""
    fused = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            d_str = dumps(doc)
            fused[d_str] = fused.get(d_str, 0) + 1 / (rank + k)
    reranked = [
        (loads(doc), score) for doc, score in sorted(fused.items(), key=lambda x: x[1], reverse=True)
    ]
    return [d for d, _ in reranked]

# === [6] Advanced Retrieval Chains ===

## (a) Multi-Query Retrieval
def build_multiquery_chain(retriever):
    prompt = ChatPromptTemplate.from_template(
        """You are an AI assistant. Generate 5 alternative formulations of the user question
        to help retrieve more relevant documents.
        Original question: {question}"""
    )
    llm = ChatOpenAI(temperature=0)
    generate_queries = prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    return retrieval_chain

## (b) RAG-Fusion Retrieval
def build_ragfusion_chain(retriever):
    prompt = ChatPromptTemplate.from_template(
        "Generate 4 related search queries for: {question}"
    )
    llm = ChatOpenAI(temperature=0)
    generate_queries = prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion
    return retrieval_chain

## (c) HyDE Retrieval
def build_hyde_chain(retriever):
    prompt = ChatPromptTemplate.from_template(
        "Please write a passage that answers the question.\nQuestion: {question}\nPassage:"
    )
    llm = ChatOpenAI(temperature=0)
    generate_docs = prompt | llm | StrOutputParser()
    retrieval_chain = generate_docs | retriever
    return retrieval_chain

# === [7] RAG Chain Assembly ===
def build_rag_chain(retrieval_chain, model_name="gpt-4o-mini", temperature=0.0):
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question based only on this context:\n\n{context}\n\nQuestion: {question}"
    )

    chain = (
        {"context": retrieval_chain, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# === [8] Orchestrator ===
def run_advanced_rag(
    docs,
    question: str,
    mode: str = "multiquery",  # options: 'multiquery', 'fusion', 'hyde'
    persist_dir: str = "./rag_db",
):
    retriever = build_vectorstore(docs, persist_dir)
    if mode == "multiquery":
        retrieval_chain = build_multiquery_chain(retriever)
    elif mode == "fusion":
        retrieval_chain = build_ragfusion_chain(retriever)
    elif mode == "hyde":
        retrieval_chain = build_hyde_chain(retriever)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    chain = build_rag_chain(retrieval_chain)
    answer = chain.invoke({"question": question})
    return answer
