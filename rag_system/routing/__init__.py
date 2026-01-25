#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Routing module for the RAG system.
Contains semantic routing and CRAG implementations.
"""

from rag_system.routing.semantic import semantic_router, build_simple_routers
from rag_system.routing.crag import crag_retrieve

__all__ = [
    "semantic_router",
    "build_simple_routers",
    "crag_retrieve",
]
