#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic and logical routing for multi-index RAG systems.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Callable

from rag_system.core import logger


def semantic_router(
    question: str,
    routers: Dict[str, Callable[[str], float]],
) -> str:
    """
    Route a question to the best index based on semantic similarity.
    
    Args:
        question: The user's question
        routers: Dictionary mapping index names to scoring functions
                 Each function takes a question and returns a relevance score
    
    Returns:
        Name of the best-matching index
    """
    scores = {k: f(question) for k, f in routers.items()}
    best = max(scores.items(), key=lambda x: x[1])[0]
    logger.debug("Router scores: %s", scores)
    return best


def build_simple_routers(
    indexes: Dict[str, Any],
    embeddings,
) -> Dict[str, Callable[[str], float]]:
    """
    Build simple routing functions for each index.
    
    Uses query-to-centroid embedding similarity for routing decisions.
    
    Args:
        indexes: Dictionary of index name to index object
        embeddings: Embedding function with embed_documents() and embed_query()
    
    Returns:
        Dictionary mapping index names to scoring functions
    """
    routers = {}
    
    for name, idx in indexes.items():
        # Compute centroid of index docs if possible
        try:
            texts = [
                getattr(d, "page_content", "")
                for d in (idx._docs if hasattr(idx, "_docs") else [])
            ]
            if not texts and hasattr(idx, "get_all_documents"):
                texts = [
                    getattr(d, "page_content", "")
                    for d in idx.get_all_documents()
                ]
        except Exception:
            texts = []
        
        # Generate a scoring function
        def make_router(texts_sample):
            if not texts_sample:
                return lambda q: 0.5
            
            # Compute centroid embedding
            try:
                import numpy as np
                
                emb_list = embeddings.embed_documents(texts_sample)
                centroid = np.mean(emb_list, axis=0)
                
                def score(q):
                    q_emb = embeddings.embed_query(q)
                    # Cosine similarity
                    dot = sum(a * b for a, b in zip(q_emb, centroid))
                    na = math.sqrt(sum(a * a for a in q_emb))
                    nb = math.sqrt(sum(b * b for b in centroid))
                    if na == 0 or nb == 0:
                        return 0.0
                    return dot / (na * nb)
                
                return score
            except Exception:
                # Fallback: keyword-matching scoring
                keywords = " ".join(texts_sample[:5]).lower()
                
                def score(q):
                    return sum(1 for w in q.lower().split() if w in keywords) / max(
                        1, len(q.split())
                    )
                
                return score
        
        # Use up to 50 docs for centroid calculation
        routers[name] = make_router(texts[:50])
    
    return routers
