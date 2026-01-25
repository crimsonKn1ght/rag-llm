#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector store and multi-index building utilities.
"""

from __future__ import annotations
import os
from typing import List, Optional, Dict, Any

from rag_system.core import Chroma, logger


def build_chroma_index(chunks, embeddings, persist_dir: str):
    """
    Build a Chroma vector index from document chunks.
    
    Args:
        chunks: List of document chunks to index
        embeddings: Embedding function with embed_documents() and embed_query() methods
        persist_dir: Directory to persist the index
    
    Returns:
        A vector database object with as_retriever() method
    """
    if Chroma is not None:
        os.makedirs(persist_dir, exist_ok=True)
        vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
        return vectordb
    
    # Fallback: a simple in-memory index (brute-force) storing embeddings
    class InMemoryIndex:
        """Simple in-memory vector index using brute-force cosine similarity."""
        
        def __init__(self, docs, embeddings):
            self.docs = docs
            self._docs = docs  # For router compatibility
            self.embeddings = embeddings
            self._emb_matrix = embeddings.embed_documents(
                [d.page_content for d in docs]
            )

        def as_retriever(self, search_kwargs=None):
            return SimpleRetriever(self, search_kwargs or {})

    class SimpleRetriever:
        """Simple retriever using cosine similarity."""
        
        def __init__(self, index, kwargs):
            self.index = index
            self.k = kwargs.get("k", 4)

        def get_relevant_documents(self, query):
            qemb = self.index.embeddings.embed_query(query)
            
            def cosine_similarity(a, b):
                da = sum(x * x for x in a) ** 0.5
                db = sum(x * x for x in b) ** 0.5
                if da == 0 or db == 0:
                    return 0.0
                return sum(x * y for x, y in zip(a, b)) / (da * db)
            
            scored = []
            for d, emb in zip(self.index.docs, self.index._emb_matrix):
                scored.append((cosine_similarity(qemb, emb), d))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [d for s, d in scored[: self.k]]

    return InMemoryIndex(chunks, embeddings)


def build_multi_index(
    chunks,
    embeddings,
    base_persist_dir: str = "./rag_db",
    names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build one or more vector indices.
    
    Args:
        chunks: List of document chunks to index
        embeddings: Embedding function
        base_persist_dir: Base directory for persisting indices
        names: List of index names to build (e.g., ['dense', 'colbert'])
    
    Returns:
        Dictionary mapping index names to vector database objects
    """
    names = names or ["dense"]
    stores = {}
    
    for name in names:
        pdir = os.path.join(base_persist_dir, name)
        
        if name == "dense":
            vectordb = build_chroma_index(chunks, embeddings, persist_dir=pdir)
            stores["dense"] = vectordb
        elif name == "colbert":
            # Placeholder: in practice you'd create a ColBERT index
            logger.info("Creating placeholder 'colbert' index (not a true ColBERT).")
            vectordb = build_chroma_index(chunks, embeddings, persist_dir=pdir)
            stores["colbert"] = vectordb
        else:
            vectordb = build_chroma_index(chunks, embeddings, persist_dir=pdir)
            stores[name] = vectordb
    
    return stores
