#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document reranking utilities using cross-encoders or embedding similarity.
"""

from __future__ import annotations
from typing import List, Optional, Any, Tuple

from rag_system.core import SentenceTransformer


class Reranker:
    """
    Reranker for scoring and reordering retrieved documents.
    Uses sentence-transformers if available, otherwise falls back to heuristics.
    """

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        """
        Initialize the reranker.
        
        Args:
            model_name: Model name for sentence-transformers
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.device = device
        
        if SentenceTransformer is not None:
            # Use a bi-encoder model for similarity scoring
            self.scorer = SentenceTransformer(
                model_name or "sentence-transformers/all-MiniLM-L6-v2",
                device=device,
            )
        else:
            self.scorer = None

    def score(self, query: str, docs: List[Any]) -> List[Tuple[float, Any]]:
        """
        Score documents against a query.
        
        Args:
            query: The query string
            docs: List of documents to score
        
        Returns:
            List of (score, document) tuples sorted by score descending
        """
        texts = [getattr(d, "page_content", str(d)) for d in docs]
        
        if self.scorer is not None:
            # Compute embeddings and use dot product
            q_emb = self.scorer.encode([query], convert_to_numpy=True)[0]
            d_embs = self.scorer.encode(texts, convert_to_numpy=True)
            
            def cosine_similarity(a, b):
                na = (a * a).sum() ** 0.5
                nb = (b * b).sum() ** 0.5
                if na == 0 or nb == 0:
                    return 0.0
                return float((a * b).sum() / (na * nb))
            
            scored = [
                (cosine_similarity(q_emb, d_embs[i]), docs[i])
                for i in range(len(docs))
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored
        else:
            # Fallback: length-based heuristic
            scored = [(len(getattr(d, "page_content", "")), d) for d in docs]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored
