#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM wrappers and utilities.
"""

from __future__ import annotations
from typing import Optional, Callable

from rag_system.core import ChatOpenAI, logger


def make_openai_llm(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> Optional[Callable[[str], str]]:
    """
    Create a callable wrapper for ChatOpenAI.
    
    Args:
        model_name: OpenAI model name
        temperature: Sampling temperature
    
    Returns:
        A callable that takes a prompt string and returns a response,
        or None if ChatOpenAI is not available
    """
    if ChatOpenAI is None:
        logger.warning(
            "ChatOpenAI not available; llm operations will be no-ops. "
            "Provide your own llm_callable."
        )
        return None

    def call(prompt_text: str) -> Optional[str]:
        instance = ChatOpenAI(model=model_name, temperature=temperature)
        
        try:
            res = instance.invoke({"prompt": prompt_text})
            return getattr(res, "content", str(res))
        except Exception:
            # Fallback: try calling directly
            try:
                return instance(prompt_text)
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                return None

    return call
