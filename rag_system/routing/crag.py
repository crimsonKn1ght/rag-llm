#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRAG (Contextual Retrieval Augmented Generation) implementation.
Iterative retrieval with query expansion based on retrieved context.
"""

from __future__ import annotations
from typing import Optional, List, Any

from rag_system.retrieval.reranker import Reranker
from rag_system.retrieval.helpers import get_unique_union


def crag_retrieve(
    retriever,
    question: str,
    reranker: Optional[Reranker] = None,
    steps: int = 2,
    k: int = 4,
) -> List[Any]:
    """
    CRAG-like iterative retrieval.
    
    Performs multiple retrieval steps where each subsequent step
    expands the query using the top retrieved document from the
    previous step.
    
    Args:
        retriever: The base retriever to use
        question: The original user question
        reranker: Optional reranker for scoring documents
        steps: Number of retrieval iterations
        k: Number of documents to retrieve per step
    
    Returns:
        Deduplicated list of all retrieved documents
    """
    all_retrieved = []
    current_query = question
    
    for step in range(steps):
        # Retrieve documents
        docs = (
            retriever.get_relevant_documents(current_query)
            if hasattr(retriever, "get_relevant_documents")
            else retriever(current_query)
        )
        
        # Optionally rerank
        if reranker:
            scored = reranker.score(current_query, docs)
            docs = [d for s, d in scored]
        
        all_retrieved.append(docs)
        
        # Expand query with the highest-scoring document
        if docs:
            top = getattr(docs[0], "page_content", str(docs[0]))
            # Create a new "contextualized" query by concatenation
            current_query = question + " " + top[:400]
    
    # Flatten and deduplicate
    return get_unique_union(all_retrieved)
