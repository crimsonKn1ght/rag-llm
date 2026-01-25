#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main RAG orchestrator module.
Provides high-level entrypoints for running RAG pipelines.
"""

from __future__ import annotations
from typing import List, Optional, Callable, Any
import logging

from rag_system.core.config import DEFAULT_PROMPT
from rag_system.core import _chat_prompt_from_template, logger
from rag_system.embeddings.factory import build_embedding_fn
from rag_system.loaders.documents import split_documents
from rag_system.indexing.vector_store import build_multi_index
from rag_system.retrieval.reranker import Reranker
from rag_system.chains.multiquery import build_multiquery_chain
from rag_system.chains.fusion import build_ragfusion_chain
from rag_system.chains.hyde import build_hyde_chain
from rag_system.chains.rag_chain import build_rag_chain_simple
from rag_system.chains.llm import make_openai_llm
from rag_system.routing.semantic import semantic_router, build_simple_routers
from rag_system.routing.crag import crag_retrieve
from rag_system.utils.context import merge_chunks_for_long_context


def run_rag(
    docs,
    question: str,
    mode: str = "basic",  # 'basic', 'multiquery', 'fusion', 'hyde', 'crag', 'self-rag', 'router'
    embeddings_backend: str = "openai",
    sbert_model: Optional[str] = None,
    persist_dir: str = "./rag_db",
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    k: int = 4,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    multi_index_names: Optional[List[str]] = None,
    rerank: bool = True,
    reranker_model: Optional[str] = None,
    router_enabled: bool = False,
    llm_callable: Optional[Callable[[str], str]] = None,
    long_context_merge_tokens: Optional[int] = None,
):
    """
    High-level unified entrypoint for advanced RAG flows.
    
    Args:
        docs: Input documents to index
        question: User query
        mode: Retrieval mode - one of 'basic', 'multiquery', 'fusion', 'hyde', 'crag', 'self-rag'
        embeddings_backend: 'openai' or 'sbert'
        sbert_model: Model name for sentence-transformers (if using sbert backend)
        persist_dir: Directory to persist vector store
        model_name: LLM model name for generation
        temperature: LLM temperature
        k: Number of documents to retrieve
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        multi_index_names: Names of indices to build (e.g., ['dense', 'colbert'])
        rerank: Whether to rerank retrieved documents
        reranker_model: Model for reranking
        router_enabled: Whether to use semantic routing
        llm_callable: Custom LLM callable function
        long_context_merge_tokens: If set, merge chunks up to this token limit
    
    Returns:
        Tuple of (answer string, list of context chunks used)
    """
    # Build embeddings
    embeddings = build_embedding_fn(backend=embeddings_backend, sbert_model=sbert_model)

    # Chunk documents
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if long_context_merge_tokens:
        chunks = merge_chunks_for_long_context(chunks, max_tokens=long_context_merge_tokens)

    # Build multi-index
    names = multi_index_names or ["dense"]
    indexes = build_multi_index(chunks, embeddings, base_persist_dir=persist_dir, names=names)

    # Create retriever (default: dense)
    base_retriever = indexes.get("dense")
    retriever = (
        base_retriever.as_retriever(search_kwargs={"k": k})
        if hasattr(base_retriever, "as_retriever")
        else base_retriever
    )

    # Prepare reranker
    reranker_obj = Reranker(model_name=reranker_model) if rerank else None

    # Prepare LLM
    llm_call = llm_callable or make_openai_llm(model_name=model_name, temperature=temperature)

    # Router (if enabled)
    if router_enabled:
        routers = build_simple_routers(indexes, embeddings)
        chosen_index_name = semantic_router(question, routers)
        logger.info("Router chose index: %s", chosen_index_name)
        chosen_index = indexes[chosen_index_name]
        retriever = (
            chosen_index.as_retriever(search_kwargs={"k": k})
            if hasattr(chosen_index, "as_retriever")
            else chosen_index
        )

    # Select retrieval chain based on mode
    if mode == "basic":
        docs_ret = (
            retriever.get_relevant_documents(question)
            if hasattr(retriever, "get_relevant_documents")
            else retriever(question)
        )
    elif mode == "multiquery":
        chain = build_multiquery_chain(retriever, llm=llm_call, count=5)
        docs_ret = chain(question)
    elif mode == "fusion":
        chain = build_ragfusion_chain(retriever, llm=llm_call, count=4)
        docs_ret = chain(question)
    elif mode == "hyde":
        chain = build_hyde_chain(retriever, llm=llm_call)
        docs_ret = chain(question)
    elif mode == "crag":
        docs_ret = crag_retrieve(retriever, question, reranker=reranker_obj, steps=2, k=k)
    elif mode == "self-rag":
        # HyDE-like: generate passage then retrieve
        hyde_prompt = _chat_prompt_from_template(
            "Please write a passage that answers the question.\nQuestion: {question}\nPassage:"
        )
        hyde_text = None
        if llm_call:
            hyde_text = llm_call(hyde_prompt.format(question=question))
            if hasattr(hyde_text, "content"):
                hyde_text = hyde_text.content
        hyde_text = hyde_text or question
        docs_ret = (
            retriever.get_relevant_documents(hyde_text)
            if hasattr(retriever, "get_relevant_documents")
            else retriever(hyde_text)
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Optional rerank
    if reranker_obj:
        scored = reranker_obj.score(question, docs_ret)
        docs_sorted = [d for s, d in scored]
    else:
        docs_sorted = docs_ret

    # Build final chain & answer
    rag_chain = build_rag_chain_simple(llm_call, prompt_template=DEFAULT_PROMPT)
    answer = rag_chain(question, docs_sorted)

    return answer, docs_sorted


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
    Backward-compatible wrapper for run_rag with sensible defaults.
    Uses fusion mode with reranking and routing enabled.
    """
    answer, chunks = run_rag(
        docs=docs,
        question=question,
        mode="fusion",
        embeddings_backend=embeddings_backend,
        sbert_model=sbert_model,
        persist_dir=persist_dir,
        model_name=model_name,
        temperature=temperature,
        k=k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        multi_index_names=["dense", "colbert"],
        rerank=True,
        router_enabled=True,
        llm_callable=None,
        long_context_merge_tokens=None,
    )
    return answer, chunks
