#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core configuration and shared dependencies for the RAG system.
"""

from __future__ import annotations
import os
import json
import logging

# Environment helpers
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# LangChain / Lang-like imports (keep compatibility)
try:
    from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain.load import dumps, loads
except Exception:
    # Placeholders if LangChain not available
    ChatPromptTemplate = None
    StrOutputParser = None
    RunnablePassthrough = None
    ChatOpenAI = None
    OpenAIEmbeddings = None
    WebBaseLoader = None
    TextLoader = None
    PyPDFLoader = None
    RecursiveCharacterTextSplitter = None
    Chroma = None
    dumps = lambda x: json.dumps(x)
    loads = lambda x: json.loads(x)

# Optional ML/cross-encoder libs
try:
    import torch
except Exception:
    torch = None

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except Exception:
    SentenceTransformer = None
    st_util = None

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default prompt template
DEFAULT_PROMPT = (
    "You are a helpful assistant. Use ONLY the provided context to answer.\n"
    "If the answer is not in the context, say you don't know.\n\n"
    "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:"
)


def _chat_prompt_from_template(template: str):
    """
    Create a ChatPromptTemplate-like object from a template string.
    Falls back to a simple formatter if LangChain is not available.
    """
    if ChatPromptTemplate is not None:
        return ChatPromptTemplate.from_template(template)
    
    # Fallback: return a callable that formats
    class _FakePrompt:
        def format(self, **kwargs):
            return template.format(**kwargs)
    return _FakePrompt()
