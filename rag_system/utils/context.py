#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context handling utilities for long-context scenarios.
"""

from __future__ import annotations
from typing import List, Any


def merge_chunks_for_long_context(
    chunks: List[Any],
    max_tokens: int = 3000,
    approx_chars_per_token: int = 4,
) -> List[Any]:
    """
    Merge neighboring chunks into larger contexts.
    
    Useful when working with models that have large context windows
    and you want to provide more cohesive context blocks.
    
    Warning: Larger contexts might exceed LLM token limits.
    
    Args:
        chunks: List of document chunks to merge
        max_tokens: Maximum tokens per merged chunk
        approx_chars_per_token: Approximate characters per token (default 4)
    
    Returns:
        List of merged document chunks
    """
    merged = []
    current_text = ""
    current_meta = None
    max_chars = max_tokens * approx_chars_per_token
    
    for c in chunks:
        text = getattr(c, "page_content", str(c))
        meta = getattr(c, "metadata", {})
        
        if len(current_text) + len(text) <= max_chars:
            current_text += "\n\n" + text
            current_meta = current_meta or meta
        else:
            # Save current merged chunk and start new one
            merged.append(
                type(
                    "Doc",
                    (),
                    {"page_content": current_text.strip(), "metadata": current_meta or {}},
                )
            )
            current_text = text
            current_meta = meta
    
    # Don't forget the last chunk
    if current_text:
        merged.append(
            type(
                "Doc",
                (),
                {"page_content": current_text.strip(), "metadata": current_meta or {}},
            )
        )
    
    return merged
