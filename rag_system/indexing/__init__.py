#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indexing module for the RAG system.
"""

from rag_system.indexing.vector_store import build_chroma_index, build_multi_index

__all__ = [
    "build_chroma_index",
    "build_multi_index",
]
