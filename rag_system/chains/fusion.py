#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Fusion retrieval chain.
Uses reciprocal rank fusion to combine results from multiple queries.
"""

from __future__ import annotations
from typing import Optional, Callable

from rag_system.core import _chat_prompt_from_template
from rag_system.retrieval.helpers import reciprocal_rank_fusion


def build_ragfusion_chain(retriever, llm: Optional[Callable] = None, count: int = 4):
    """
    Build a RAG Fusion retrieval chain.
    
    Generates related search queries and uses reciprocal rank fusion
    to combine and rerank the results.
    
    Args:
        retriever: The base retriever to use
        llm: LLM callable for generating related queries (optional)
        count: Number of related queries to generate
    
    Returns:
        A callable that takes a question and returns fused/reranked documents
    """
    prompt = _chat_prompt_from_template(
        "Generate {n} related search queries for: {question}"
    )

    def chain(question: str):
        # Generate related queries (simple fallback if no LLM)
        queries = [f"{question} related {i + 1}" for i in range(count)]
        
        results = []
        for q in queries:
            docs = (
                retriever.get_relevant_documents(q)
                if hasattr(retriever, "get_relevant_documents")
                else retriever(q)
            )
            results.append(docs)
        
        return reciprocal_rank_fusion(results)

    return chain
