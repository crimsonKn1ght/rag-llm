#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage when running the RAG system as a module.
"""

from rag_system import run_rag
from rag_system.loaders import load_docs_from_url
from rag_system.core import WebBaseLoader, SentenceTransformer


def main():
    """Run an example RAG query."""
    # Example: load documents (replace with your own docs)
    if WebBaseLoader is not None:
        docs = load_docs_from_url("https://en.wikipedia.org/wiki/Artificial_intelligence")
    else:
        # Fallback minimal document
        docs = [
            type(
                "D",
                (),
                {
                    "page_content": "Artificial intelligence aims to create agents that act intelligently.",
                    "metadata": {},
                },
            )
        ]

    question = "What are the main goals of AI research?"
    
    # Determine embedding backend based on availability
    if SentenceTransformer is not None:
        embeddings_backend = "sbert"
        sbert_model = "sentence-transformers/all-MiniLM-L6-v2"
    else:
        embeddings_backend = "openai"
        sbert_model = None

    # Run RAG
    answer, context_chunks = run_rag(
        docs,
        question,
        mode="fusion",
        embeddings_backend=embeddings_backend,
        sbert_model=sbert_model,
    )

    print("ANSWER:\n", answer)
    print("\n--- CONTEXT CHUNKS USED ---")
    for c in context_chunks[:5]:
        print(getattr(c, "page_content", "")[:400].replace("\n", " "), "\n---")


if __name__ == "__main__":
    main()