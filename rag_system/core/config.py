#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration constants for the RAG system.
"""

# Default prompt template
DEFAULT_PROMPT = (
    "You are a helpful assistant. Use ONLY the provided context to answer.\n"
    "If the answer is not in the context, say you don't know.\n\n"
    "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:"
)
