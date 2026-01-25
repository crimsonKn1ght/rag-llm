#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HyDE (Hypothetical Document Embeddings) retrieval chain.
Generates a hypothetical answer and uses it for retrieval.
"""

from __future__ import annotations
from typing import Optional, Callable

from rag_system.core import _chat_prompt_from_template


def build_hyde_chain(retriever, llm: Optional[Callable] = None):
    """
    Build a HyDE retrieval chain.
    
    HyDE generates a hypothetical answer passage using the LLM,
    then uses that passage as the query for retrieval instead of
    the original question.
    
    Args:
        retriever: The base retriever to use
        llm: LLM callable for generating the hypothetical answer
    
    Returns:
        A callable that takes a question and returns retrieved documents
    """
    prompt = _chat_prompt_from_template(
        "Please write a passage that answers the question.\n"
        "Question: {question}\n"
        "Passage:"
    )

    def chain(question: str):
        generated = None
        
        if llm is not None and callable(llm):
            generated = llm(prompt.format(question=question))
            if hasattr(generated, "content"):
                generated = generated.content
        else:
            # Fallback: use original question (HyDE less effective)
            generated = question
        
        docs = (
            retriever.get_relevant_documents(generated)
            if hasattr(retriever, "get_relevant_documents")
            else retriever(generated)
        )
        return docs

    return chain
