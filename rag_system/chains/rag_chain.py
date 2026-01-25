#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG chain builder for combining retrieval with generation.
"""

from __future__ import annotations
from typing import List, Optional, Callable, Any

from rag_system.core import _chat_prompt_from_template
from rag_system.core.config import DEFAULT_PROMPT
from rag_system.retrieval.helpers import format_docs


def build_rag_chain_simple(
    llm_callable: Optional[Callable[[str], str]],
    prompt_template: str = DEFAULT_PROMPT,
):
    """
    Build a simple RAG chain that combines context with a question.
    
    Args:
        llm_callable: Function that takes a prompt string and returns a response
        prompt_template: Template string with {question} and {context} placeholders
    
    Returns:
        A callable that takes a question and context docs and returns an answer
    """
    prompt_obj = _chat_prompt_from_template(prompt_template)

    def chain(question: str, context_docs: List[Any]) -> str:
        context = format_docs(context_docs)
        
        # Format the prompt
        if hasattr(prompt_obj, "format"):
            full_prompt = prompt_obj.format(question=question, context=context)
        else:
            full_prompt = prompt_template.format(question=question, context=context)
        
        if llm_callable is None:
            # Fallback: return context (debug mode)
            return "LLM missing: would run model on prompt:\n\n" + full_prompt[:2000]
        
        return llm_callable(full_prompt)

    return chain
