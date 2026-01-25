#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrieval helper functions for document processing and ranking.
"""

from __future__ import annotations
import json
from typing import List, Any

from rag_system.core import dumps, loads


def format_docs(docs: List[Any]) -> str:
    """
    Format a list of documents into a single string.
    
    Args:
        docs: List of documents with page_content attribute
    
    Returns:
        Concatenated string of document contents
    """
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)


def get_unique_union(document_lists: List[List[Any]]) -> List[Any]:
    """
    Flatten and deduplicate documents from multiple lists.
    
    Args:
        document_lists: List of document lists to merge
    
    Returns:
        Deduplicated list of documents
    """
    try:
        flat = [dumps(doc) for sublist in document_lists for doc in sublist]
        unique = list(dict.fromkeys(flat))  # Keep order
        return [loads(doc) for doc in unique]
    except Exception:
        # Fallback: deduplicate by page_content & metadata
        seen = set()
        out = []
        for sublist in document_lists:
            for d in sublist:
                key = (
                    getattr(d, "page_content", None),
                    json.dumps(getattr(d, "metadata", {}), sort_keys=True),
                )
                if key not in seen:
                    seen.add(key)
                    out.append(d)
        return out


def reciprocal_rank_fusion(results: List[List[Any]], k: int = 60) -> List[Any]:
    """
    Perform reciprocal rank fusion on multiple result lists.
    
    Args:
        results: List of ranked document lists
        k: Fusion constant (default 60)
    
    Returns:
        Fused and reranked list of documents
    """
    fused = {}
    
    for docs in results:
        for rank, doc in enumerate(docs):
            try:
                d_str = dumps(doc)
            except Exception:
                d_str = getattr(doc, "page_content", str(doc))
            fused[d_str] = fused.get(d_str, 0) + 1 / (rank + k)
    
    reranked = [
        (
            loads(doc) if isinstance(doc, str) and doc.strip().startswith("{") else doc,
            score,
        )
        for doc, score in sorted(fused.items(), key=lambda x: x[1], reverse=True)
    ]
    
    return [d for d, _ in reranked]
