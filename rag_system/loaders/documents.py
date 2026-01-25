#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document loaders and text splitting utilities.
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence, List, Any

from rag_system.core import (
    WebBaseLoader,
    TextLoader,
    PyPDFLoader,
    RecursiveCharacterTextSplitter,
)


def load_docs_from_url(url: str):
    """
    Load documents from a URL using WebBaseLoader.
    
    Args:
        url: The URL to load documents from
    
    Returns:
        List of loaded documents
    
    Raises:
        RuntimeError: If WebBaseLoader is not available
    """
    if WebBaseLoader is None:
        raise RuntimeError("WebBaseLoader not available in this environment.")
    loader = WebBaseLoader(url)
    return loader.load()


def load_docs_from_paths(paths: Sequence[Path]):
    """
    Load documents from local file paths.
    Supports .txt, .md, and .pdf files.
    
    Args:
        paths: Sequence of Path objects to load
    
    Returns:
        List of loaded documents
    
    Raises:
        RuntimeError: If required loaders are not available
    """
    docs = []
    for p in paths:
        suffix = p.suffix.lower()
        if suffix in [".txt", ".md"]:
            if TextLoader is None:
                raise RuntimeError("TextLoader not available.")
            docs.extend(TextLoader(str(p), autodetect_encoding=True).load())
        elif suffix == ".pdf":
            if PyPDFLoader is None:
                raise RuntimeError("PyPDFLoader not available.")
            docs.extend(PyPDFLoader(str(p)).load())
    return docs


def split_documents(
    docs: List[Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    """
    Split documents into smaller chunks.
    
    Args:
        docs: List of documents to split
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Overlap between consecutive chunks
    
    Returns:
        List of document chunks
    """
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.split_documents(docs)
    
    # Fallback naive splitter (by characters)
    chunks = []
    for d in docs:
        text = getattr(d, "page_content", str(d))
        metadata = getattr(d, "metadata", {})
        i = 0
        while i < len(text):
            chunk_text = text[i : i + chunk_size]
            chunks.append(
                type("Doc", (), {"page_content": chunk_text, "metadata": metadata})
            )
            i += chunk_size - chunk_overlap
    return chunks
