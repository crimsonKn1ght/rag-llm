#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced RAG System

A modular RAG system integrating:
- Routing (semantic / logical)
- Query structuring & metadata-aware filters
- Multi-representation indexing (dense + placeholder for ColBERT/RAPTOR)
- Reranking (cross-encoder fallback)
- CRAG (contextual retrieval + re-retrieve)
- Self-RAG / HyDE generation flow
- Long-context/chunk merging helpers
"""

from rag_system.core.orchestrator import run_rag, run_rag_with_docs
from rag_system.core.config import DEFAULT_PROMPT
from rag_system.embeddings.factory import build_embedding_fn
from rag_system.loaders.documents import (
    load_docs_from_url,
    load_docs_from_paths,
    split_documents,
)
from rag_system.indexing.vector_store import build_chroma_index, build_multi_index
from rag_system.retrieval.reranker import Reranker
from rag_system.retrieval.helpers import format_docs, get_unique_union, reciprocal_rank_fusion
from rag_system.chains.multiquery import build_multiquery_chain
from rag_system.chains.fusion import build_ragfusion_chain
from rag_system.chains.hyde import build_hyde_chain
from rag_system.chains.rag_chain import build_rag_chain_simple
from rag_system.chains.llm import make_openai_llm
from rag_system.routing.semantic import semantic_router, build_simple_routers
from rag_system.routing.crag import crag_retrieve
from rag_system.utils.context import merge_chunks_for_long_context

__all__ = [
    # Orchestrator
    "run_rag",
    "run_rag_with_docs",
    # Config
    "DEFAULT_PROMPT",
    # Embeddings
    "build_embedding_fn",
    # Loaders
    "load_docs_from_url",
    "load_docs_from_paths",
    "split_documents",
    # Indexing
    "build_chroma_index",
    "build_multi_index",
    # Retrieval
    "Reranker",
    "format_docs",
    "get_unique_union",
    "reciprocal_rank_fusion",
    # Chains
    "build_multiquery_chain",
    "build_ragfusion_chain",
    "build_hyde_chain",
    "build_rag_chain_simple",
    "make_openai_llm",
    # Routing
    "semantic_router",
    "build_simple_routers",
    "crag_retrieve",
    # Utils
    "merge_chunks_for_long_context",
]

__version__ = "1.0.0"
