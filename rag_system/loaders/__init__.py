#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document loaders module for the RAG system.
"""

from rag_system.loaders.documents import (
    load_docs_from_url,
    load_docs_from_paths,
    split_documents,
)

__all__ = [
    "load_docs_from_url",
    "load_docs_from_paths",
    "split_documents",
]
