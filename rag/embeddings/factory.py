import os
from typing import Optional
import torch

from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings

def build_embedding_fn(backend="openai", sbert_model: Optional[str] = None):
    backend = backend.lower()

    if backend == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        return OpenAIEmbeddings()

    if backend == "sbert":
        model = SentenceTransformer(
            sbert_model or "sentence-transformers/all-MiniLM-L6-v2",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        class SBERTEmbeddings:
            def embed_documents(self, texts):
                return model.encode(texts, convert_to_numpy=False).tolist()

            def embed_query(self, text):
                return model.encode([text], convert_to_numpy=False)[0].tolist()

        return SBERTEmbeddings()

    raise ValueError(f"Unknown backend: {backend}")
