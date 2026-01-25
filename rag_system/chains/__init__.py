#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chains module for the RAG system.
Contains various retrieval and generation chains.
"""

from rag_system.chains.multiquery import build_multiquery_chain
from rag_system.chains.fusion import build_ragfusion_chain
from rag_system.chains.hyde import build_hyde_chain
from rag_system.chains.rag_chain import build_rag_chain_simple
from rag_system.chains.llm import make_openai_llm

__all__ = [
    "build_multiquery_chain",
    "build_ragfusion_chain",
    "build_hyde_chain",
    "build_rag_chain_simple",
    "make_openai_llm",
]
