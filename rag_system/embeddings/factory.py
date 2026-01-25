#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embeddings factory for creating embedding functions.
Supports OpenAI and Sentence-Transformers backends.
"""

from __future__ import annotations
import os
from typing import List, Optional

from rag_system.core import (
    OpenAIEmbeddings,
    SentenceTransformer,
    torch,
)


def build_embedding_fn(backend: str = "openai", sbert_model: Optional[str] = None):
    """
    Build an embedding function based on the specified backend.
    
    Args:
        backend: Either 'openai' or 'sbert'
        sbert_model: Model name for sentence-transformers (only used if backend='sbert')
    
    Returns:
        An embedding object with embed_documents() and embed_query() methods
    
    Raises:
        RuntimeError: If required dependencies are not available
        ValueError: If unknown backend is specified
    """
    backend = backend.lower().strip()
    
    if backend == "openai":
        if OpenAIEmbeddings is None:
            raise RuntimeError(
                "OpenAIEmbeddings not available; install langchain_openai or set backend to 'sbert'."
            )
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
            """Wrapper class for sentence-transformers embeddings."""
            
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                """Embed a list of documents."""
                emb = st_model.encode(
                    texts,
                    convert_to_numpy=False,
                    batch_size=32,
                    show_progress_bar=False,
                )
                return [list(x) for x in emb]

            def embed_query(self, text: str) -> List[float]:
                """Embed a single query."""
                emb = st_model.encode(
                    [text],
                    convert_to_numpy=False,
                    show_progress_bar=False,
                )[0]
                return list(emb)

        return SBERTEmbeddings()
    
    raise ValueError(f"Unknown embedding backend: {backend!r}")
