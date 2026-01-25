#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-query retrieval chain.
Generates multiple query variants and retrieves from each.
"""

from __future__ import annotations
from typing import Optional, Callable

from rag_system.core import _chat_prompt_from_template
from rag_system.retrieval.helpers import get_unique_union


def build_multiquery_chain(retriever, llm: Optional[Callable] = None, count: int = 5):
    """
    Build a multi-query retrieval chain.
    
    Generates multiple variants of the input question using an LLM,
    then retrieves documents for each variant and merges the results.
    
    Args:
        retriever: The base retriever to use
        llm: LLM callable for generating query variants
        count: Number of query variants to generate
    
    Returns:
        A callable that takes a question and returns retrieved documents
    """
    prompt = _chat_prompt_from_template(
        "Generate {n} alternative formulations of this user question.\n"
        "Original question: {question}"
    )

    class QueryGenerator:
        """Generates multiple variants of a query."""
        
        def __init__(self, prompt, llm, n):
            self.prompt = prompt
            self.llm = llm
            self.n = n

        def __call__(self, question):
            if self.llm is None:
                # Fallback: add trivial variants
                return [question + f" (variant {i + 1})" for i in range(self.n)]
            
            # Use LLM to generate variants
            formatted = self.prompt.format(question=question, n=self.n)
            res = self.llm(formatted) if callable(self.llm) else formatted
            text = res if isinstance(res, str) else getattr(res, "content", str(res))
            items = [line.strip() for line in text.splitlines() if line.strip()]
            return items[: self.n]

    generator = QueryGenerator(prompt, llm, count)

    def chain(question: str):
        queries = generator(question)
        results = []
        
        for q in queries:
            try:
                docs = (
                    retriever.get_relevant_documents(q)
                    if hasattr(retriever, "get_relevant_documents")
                    else retriever(q)
                )
            except Exception:
                # Some retrievers use different interfaces
                docs = retriever(q)
            results.append(docs)
        
        # Dedupe and flatten
        return get_unique_union(results)

    return chain
