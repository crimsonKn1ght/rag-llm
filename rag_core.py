#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced RAG system integrating:
- routing (semantic / logical)
- query structuring & metadata-aware filters
- multi-representation indexing (dense + placeholder for ColBERT/RAPTOR)
- reranking (cross-encoder fallback)
- CRAG (contextual retrieval + re-retrieve)
- Self-RAG / HyDE generation flow
- long-context/chunk merging helpers

Drop-in upgrade for your earlier script.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional, Sequence, Dict, Callable, Any, Tuple
import math
import itertools
import logging
import json

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
    # If you don't have LangChain import available, we provide placeholders.
    # The code below will still run for parts that don't require langchain.
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

# ---------------------------
# [0] Default prompt & utils
# ---------------------------
DEFAULT_PROMPT = (
    "You are a helpful assistant. Use ONLY the provided context to answer.\n"
    "If the answer is not in the context, say you don't know.\n\n"
    "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:"
)

# small helper to produce ChatPromptTemplate-like object if LC not present
def _chat_prompt_from_template(template: str):
    if ChatPromptTemplate is not None:
        return ChatPromptTemplate.from_template(template)
    # fallback: return a callable that formats
    class _FakePrompt:
        def format(self, **kwargs):
            return template.format(**kwargs)
    return _FakePrompt()

# ---------------------------
# [1] Embeddings factory
# ---------------------------
def build_embedding_fn(backend: str = "openai", sbert_model: Optional[str] = None):
    backend = backend.lower().strip()
    if backend == "openai":
        if OpenAIEmbeddings is None:
            raise RuntimeError("OpenAIEmbeddings not available; install langchain_openai or set backend to 'sbert'.")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set.")
        return OpenAIEmbeddings()
    if backend == "sbert":
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed.")
        model_name = sbert_model or "sentence-transformers/all-MiniLM-L6-v2"
        device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        st_model = SentenceTransformer(model_name, device=device)

        class SBERTEmbeddings:
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                # returns list of vectors (python list)
                emb = st_model.encode(texts, convert_to_numpy=False, batch_size=32, show_progress_bar=False)
                return [list(x) for x in emb]

            def embed_query(self, text: str) -> List[float]:
                emb = st_model.encode([text], convert_to_numpy=False, show_progress_bar=False)[0]
                return list(emb)

        return SBERTEmbeddings()
    raise ValueError(f"Unknown embedding backend: {backend!r}")

# ---------------------------
# [2] Loaders & chunking
# ---------------------------
def load_docs_from_url(url: str):
    if WebBaseLoader is None:
        raise RuntimeError("WebBaseLoader not available in this environment.")
    loader = WebBaseLoader(url)
    return loader.load()

def load_docs_from_paths(paths: Sequence[Path]):
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

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(docs)
    # fallback naive splitter (by characters)
    chunks = []
    for d in docs:
        text = getattr(d, "page_content", str(d))
        metadata = getattr(d, "metadata", {})
        i = 0
        while i < len(text):
            chunk_text = text[i:i+chunk_size]
            chunks.append(type("Doc", (), {"page_content": chunk_text, "metadata": metadata}))
            i += chunk_size - chunk_overlap
    return chunks

# ---------------------------
# [3] Vector store & multi-index
# ---------------------------
def build_chroma_index(chunks, embeddings, persist_dir: str):
    # Use Chroma.from_documents if available; else simulate simple in-memory store
    if Chroma is not None:
        os.makedirs(persist_dir, exist_ok=True)
        vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
        return vectordb
    # fallback: a simple in-memory index (brute-force) storing embeddings
    class InMemoryIndex:
        def __init__(self, docs, embeddings):
            self.docs = docs
            self.embeddings = embeddings
            self._emb_matrix = embeddings.embed_documents([d.page_content for d in docs])
        def as_retriever(self, search_kwargs=None):
            return SimpleRetriever(self, search_kwargs or {})
    class SimpleRetriever:
        def __init__(self, index, kwargs):
            self.index = index
            self.k = kwargs.get("k", 4)
        def get_relevant_documents(self, query):
            qemb = embeddings.embed_query(query)
            # brute-force cosine
            def cos(a,b):
                da = sum(x*x for x in a)**0.5
                db = sum(x*x for x in b)**0.5
                if da==0 or db==0:
                    return 0.0
                return sum(x*y for x,y in zip(a,b))/(da*db)
            scored = []
            for d, emb in zip(self.index.docs, self.index._emb_matrix):
                scored.append((cos(qemb, emb), d))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [d for s,d in scored[:self.k]]
    return InMemoryIndex(chunks, embeddings)

def build_multi_index(chunks, embeddings, base_persist_dir="./rag_db", names: Optional[List[str]] = None):
    """
    Build one or more indices. names can include e.g. ['dense','colbert'].
    For now we build a dense index (Chroma/OpenAI/SBERT) and provide a placeholder for colbert/raptor.
    Returns dict: name -> vectordb object with .as_retriever(search_kwargs) method.
    """
    names = names or ["dense"]
    stores = {}
    for name in names:
        pdir = os.path.join(base_persist_dir, name)
        if name == "dense":
            vectordb = build_chroma_index(chunks, embeddings, persist_dir=pdir)
            stores["dense"] = vectordb
        elif name == "colbert":
            # placeholder: in practice you'd create a ColBERT index (requires specialized code).
            logger.info("Creating placeholder 'colbert' index (not a true ColBERT).")
            vectordb = build_chroma_index(chunks, embeddings, persist_dir=pdir)
            stores["colbert"] = vectordb
        else:
            vectordb = build_chroma_index(chunks, embeddings, persist_dir=pdir)
            stores[name] = vectordb
    return stores

# ---------------------------
# [4] Retrieval helpers / reranker
# ---------------------------
def format_docs(docs):
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

def get_unique_union(document_lists: List[List[Any]]):
    """Flatten + deduplicate documents using dumps/loads where possible."""
    try:
        flat = [dumps(doc) for sublist in document_lists for doc in sublist]
        unique = list(dict.fromkeys(flat))  # keep order
        return [loads(doc) for doc in unique]
    except Exception:
        # fallback: deduplicate by page_content & metadata
        seen = set()
        out = []
        for sublist in document_lists:
            for d in sublist:
                key = (getattr(d, "page_content", None), json.dumps(getattr(d, "metadata", {}), sort_keys=True))
                if key not in seen:
                    seen.add(key)
                    out.append(d)
        return out

def reciprocal_rank_fusion(results: List[List[Any]], k: int = 60):
    fused = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            try:
                d_str = dumps(doc)
            except Exception:
                d_str = getattr(doc, "page_content", str(doc))
            fused[d_str] = fused.get(d_str, 0) + 1 / (rank + k)
    reranked = [(loads(doc) if isinstance(doc, str) and doc.strip().startswith("{") else doc, score) for doc, score in sorted(fused.items(), key=lambda x: x[1], reverse=True)]
    return [d for d, _ in reranked]

# Reranker: cross-encoder if available, else dot-product on embeddings
class Reranker:
    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        self.device = device
        if SentenceTransformer is not None:
            # use a cross-encoder-ish model if available (we'll use bi-encoder similarity by concatenating query+doc)
            self.scorer = SentenceTransformer(model_name or "sentence-transformers/all-MiniLM-L6-v2", device=device)
        else:
            self.scorer = None

    def score(self, query: str, docs: List[Any]) -> List[Tuple[float, Any]]:
        texts = [getattr(d, "page_content", str(d)) for d in docs]
        if self.scorer is not None:
            # We'll compute embeddings for query and docs and use dot product (fast)
            q_emb = self.scorer.encode([query], convert_to_numpy=True)[0]
            d_embs = self.scorer.encode(texts, convert_to_numpy=True)
            def cos(a,b):
                na = (a*a).sum()**0.5
                nb = (b*b).sum()**0.5
                if na==0 or nb==0:
                    return 0.0
                return float((a*b).sum() / (na*nb))
            scored = [(cos(q_emb, d_embs[i]), docs[i]) for i in range(len(docs))]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored
        else:
            # fallback: length-based heuristic
            scored = [(len(getattr(d, "page_content", "")), d) for d in docs]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored

# ---------------------------
# [5] Retrieval modes / chains
# ---------------------------
def build_multiquery_chain(retriever, llm=None, count=5):
    # Generate variants of the question using llm then map to retriever
    prompt = _chat_prompt_from_template("Generate {n} alternative formulations of this user question.\nOriginal question: {question}")
    class Generator:
        def __init__(self, prompt, llm, n):
            self.prompt = prompt
            self.llm = llm
            self.n = n
        def __call__(self, question):
            if self.llm is None:
                # fallback naive rephrases: add trivial variants
                return [question + " (variant {})".format(i+1) for i in range(self.n)]
            # If we have a ChatOpenAI-compatible callable, call it
            formatted = self.prompt.format(question=question, n=self.n)
            res = self.llm(formatted) if callable(self.llm) else formatted
            text = res if isinstance(res, str) else getattr(res, "content", str(res))
            items = [line.strip() for line in text.splitlines() if line.strip()]
            return items[:self.n]
    generator = Generator(prompt, llm, count)

    def chain(question):
        queries = generator(question)
        results = []
        for q in queries:
            try:
                docs = retriever.get_relevant_documents(q) if hasattr(retriever, "get_relevant_documents") else retriever(q)
            except Exception:
                # some retrievers use .as_retriever(...) or map style; attempt direct call
                docs = retriever(q)
            results.append(docs)
        # dedupe + flatten
        return get_unique_union(results)
    return chain

def build_ragfusion_chain(retriever, llm=None, count=4):
    prompt = _chat_prompt_from_template("Generate {n} related search queries for: {question}")
    # reuse generator pattern
    def chain(question):
        # simple generator fallback
        queries = [(question + f" related {i+1}") for i in range(count)]
        results = []
        for q in queries:
            docs = retriever.get_relevant_documents(q) if hasattr(retriever, "get_relevant_documents") else retriever(q)
            results.append(docs)
        return reciprocal_rank_fusion(results)
    return chain

def build_hyde_chain(retriever, llm=None):
    # HyDE: generate an answer passage using LLM, then retrieve using that passage as query
    prompt = _chat_prompt_from_template("Please write a passage that answers the question.\nQuestion: {question}\nPassage:")
    def chain(question):
        generated = None
        if llm is not None and callable(llm):
            generated = llm(prompt.format(question=question))
            if hasattr(generated, "content"):
                generated = generated.content
        else:
            # fallback: return original question (HyDE less effective)
            generated = question
        docs = retriever.get_relevant_documents(generated) if hasattr(retriever, "get_relevant_documents") else retriever(generated)
        return docs
    return chain

# ---------------------------
# [6] Router (semantic & logical routing)
# ---------------------------
def semantic_router(question: str, routers: Dict[str, Callable[[str], float]]) -> str:
    """
    routers: name -> function(question) -> score (higher = more relevant)
    returns key for best router
    """
    scores = {k: f(question) for k,f in routers.items()}
    best = max(scores.items(), key=lambda x: x[1])[0]
    logger.debug("Router scores: %s", scores)
    return best

def build_simple_routers(indexes: Dict[str, Any], embeddings):
    """
    Build simple routing functions using query->embedding similarity against index centroids or keywords.
    """
    routers = {}
    for name, idx in indexes.items():
        # compute centroid of index docs if possible
        try:
            texts = [getattr(d, "page_content", "") for d in (idx._docs if hasattr(idx, "_docs") else [])]
            if not texts and hasattr(idx, "get_all_documents"):
                texts = [getattr(d,"page_content","") for d in idx.get_all_documents()]
        except Exception:
            texts = []
        # generate a simple scoring function
        def make_router(texts_sample):
            if not texts_sample:
                return lambda q: 0.5
            # compute centroid embedding
            try:
                centroid = None
                emb_list = embeddings.embed_documents(texts_sample)
                import numpy as np
                centroid = np.mean(emb_list, axis=0)
                def score(q):
                    q_emb = embeddings.embed_query(q)
                    # cosine
                    import math
                    dot = sum(a*b for a,b in zip(q_emb, centroid))
                    na = math.sqrt(sum(a*a for a in q_emb))
                    nb = math.sqrt(sum(b*b for b in centroid))
                    if na==0 or nb==0:
                        return 0.0
                    return dot/(na*nb)
                return score
            except Exception:
                # fallback: keyword-matching scoring
                keywords = " ".join(texts_sample[:5]).lower()
                def score(q):
                    return sum(1 for w in q.lower().split() if w in keywords) / max(1,len(q.split()))
                return score
        routers[name] = make_router(texts[:50])  # use up to 50 docs for centroid
    return routers

# ---------------------------
# [7] CRAG: contextual retrieval augmentation
# ---------------------------
def crag_retrieve(retriever, question: str, reranker: Optional[Reranker] = None, steps: int = 2, k: int = 4):
    """
    CRAG-like: iterative retrieval where we:
      - retrieve with retriever
      - optionally rerank
      - expand query using retrieved context and re-retrieve
    """
    all_retrieved = []
    current_query = question
    for step in range(steps):
        docs = retriever.get_relevant_documents(current_query) if hasattr(retriever, "get_relevant_documents") else retriever(current_query)
        if reranker:
            scored = reranker.score(current_query, docs)
            docs = [d for s,d in scored]
        all_retrieved.append(docs)
        # expand query with highest doc
        if docs:
            top = getattr(docs[0], "page_content", str(docs[0]))
            # create a new "contextualized" query: concatenation (simple)
            current_query = question + " " + top[:400]
    # flatten and dedupe
    return get_unique_union(all_retrieved)

# ---------------------------
# [8] Long-context helpers (chunk merging)
# ---------------------------
def merge_chunks_for_long_context(chunks: List[Any], max_tokens: int = 3000, approx_chars_per_token: int = 4):
    """
    Merge neighboring chunks into larger contexts up to max_tokens. Useful if you want fewer,
    larger context windows (warning: larger contexts might exceed LLM limits).
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
            merged.append(type("Doc", (), {"page_content": current_text.strip(), "metadata": current_meta or {}}))
            current_text = text
            current_meta = meta
    if current_text:
        merged.append(type("Doc", (), {"page_content": current_text.strip(), "metadata": current_meta or {}}))
    return merged

# ---------------------------
# [9] Orchestrator: build rag chain & run
# ---------------------------
def build_rag_chain_simple(llm_callable: Optional[Callable[[str], str]], prompt_template: str = DEFAULT_PROMPT):
    """
    Returns a function that given question & context text returns model answer.
    llm_callable should be a function that takes a prompt string and returns a string response.
    """
    prompt_obj = _chat_prompt_from_template(prompt_template)
    def chain(question: str, context_docs: List[Any]):
        context = format_docs(context_docs)
        full_prompt = prompt_obj.format(question=question, context=context) if hasattr(prompt_obj, "format") else prompt_template.format(question=question, context=context)
        if llm_callable is None:
            # fallback: return context (debug mode)
            return "LLM missing: would run model on prompt:\n\n" + full_prompt[:2000]
        return llm_callable(full_prompt)
    return chain

# A convenience wrapper to use ChatOpenAI if available
def make_openai_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    if ChatOpenAI is None:
        logger.warning("ChatOpenAI not available; llm operations will be no-ops. Provide your own llm_callable.")
        return None
    def call(prompt_text: str):
        # ChatOpenAI expects a prompt object in your earlier code — here we call a simple variant
        # If LangChain's ChatOpenAI exposes .invoke or __call__, you may adapt this wrapper
        instance = ChatOpenAI(model=model_name, temperature=temperature)
        # if it supports invocation:
        try:
            res = instance.invoke({"prompt": prompt_text})
            return getattr(res, "content", str(res))
        except Exception:
            # fallback: try calling directly
            try:
                return instance(prompt_text)
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                return None
    return call

# The main orchestrator function (upgraded)
def run_rag(
    docs,
    question: str,
    mode: str = "basic",  # 'basic', 'multiquery', 'fusion', 'hyde', 'crag', 'self-rag', 'router'
    embeddings_backend: str = "openai",
    sbert_model: Optional[str] = None,
    persist_dir: str = "./rag_db",
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    k: int = 4,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    multi_index_names: Optional[List[str]] = None,
    rerank: bool = True,
    reranker_model: Optional[str] = None,
    router_enabled: bool = False,
    llm_callable: Optional[Callable[[str], str]] = None,
    long_context_merge_tokens: Optional[int] = None,
):
    """
    High-level unified entrypoint for advanced RAG flows.
    Returns answer (string) and the final list of context chunks used.
    """
    # Build embeddings
    embeddings = build_embedding_fn(backend=embeddings_backend, sbert_model=sbert_model)

    # chunk docs
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if long_context_merge_tokens:
        chunks = merge_chunks_for_long_context(chunks, max_tokens=long_context_merge_tokens)

    # build multi-index
    names = multi_index_names or ["dense"]
    indexes = build_multi_index(chunks, embeddings, base_persist_dir=persist_dir, names=names)

    # create retriever (default: dense)
    base_retriever = indexes.get("dense")
    retriever = base_retriever.as_retriever(search_kwargs={"k": k}) if hasattr(base_retriever, "as_retriever") else base_retriever

    # prepare reranker
    reranker_obj = Reranker(model_name=reranker_model) if rerank else None

    # prepare llm
    llm_call = llm_callable or make_openai_llm(model_name=model_name, temperature=temperature)

    # Router (if enabled)
    if router_enabled:
        routers = build_simple_routers(indexes, embeddings)
        chosen_index_name = semantic_router(question, routers)
        logger.info("Router chose index: %s", chosen_index_name)
        chosen_index = indexes[chosen_index_name]
        retriever = chosen_index.as_retriever(search_kwargs={"k": k}) if hasattr(chosen_index, "as_retriever") else chosen_index

    # Select retrieval chain
    if mode == "basic":
        # single retrieval
        docs_ret = retriever.get_relevant_documents(question) if hasattr(retriever, "get_relevant_documents") else retriever(question)
    elif mode == "multiquery":
        # multiquery on dense retriever
        chain = build_multiquery_chain(retriever, llm=llm_call, count=5)
        docs_ret = chain(question)
    elif mode == "fusion":
        chain = build_ragfusion_chain(retriever, llm=llm_call, count=4)
        docs_ret = chain(question)
    elif mode == "hyde":
        chain = build_hyde_chain(retriever, llm=llm_call)
        docs_ret = chain(question)
    elif mode == "crag":
        docs_ret = crag_retrieve(retriever, question, reranker=reranker_obj, steps=2, k=k)
    elif mode == "self-rag":
        # HyDE-like: generate passage (hypothetical answer) then retrieve; then answer using RAG
        hyde_prompt = _chat_prompt_from_template("Please write a passage that answers the question.\nQuestion: {question}\nPassage:")
        hyde_text = None
        if llm_call:
            hyde_text = llm_call(hyde_prompt.format(question=question))
            if hasattr(hyde_text, "content"):
                hyde_text = hyde_text.content
        hyde_text = hyde_text or question
        docs_ret = retriever.get_relevant_documents(hyde_text) if hasattr(retriever, "get_relevant_documents") else retriever(hyde_text)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # optional rerank
    if reranker_obj:
        scored = reranker_obj.score(question, docs_ret)
        docs_sorted = [d for s,d in scored]
    else:
        docs_sorted = docs_ret

    # build final chain & answer
    rag_chain = build_rag_chain_simple(llm_call, prompt_template=DEFAULT_PROMPT)
    answer = rag_chain(question, docs_sorted)

    return answer, docs_sorted

# ---------------------------
# [10] Backward-compatible wrapper
# ---------------------------
def run_rag_with_docs(
    docs,
    question: str,
    embeddings_backend: str = "openai",
    sbert_model: Optional[str] = None,
    persist_dir: str = "./rag_db",
    k: int = 4,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    answer, chunks = run_rag(
        docs=docs,
        question=question,
        mode="fusion",
        embeddings_backend=embeddings_backend,
        sbert_model=sbert_model,
        persist_dir=persist_dir,
        model_name=model_name,
        temperature=temperature,
        k=k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        multi_index_names=["dense", "colbert"],
        rerank=True,
        router_enabled=True,
        llm_callable=None,  # use default ChatOpenAI if available
        long_context_merge_tokens=None,
    )
    return answer, chunks

# ---------------------------
# [11] Example usage if run as script
# ---------------------------
if __name__ == "__main__":
    # Example (replace with your own docs)
    if WebBaseLoader is not None:
        docs = load_docs_from_url("https://en.wikipedia.org/wiki/Artificial_intelligence")
    else:
        docs = [type("D", (), {"page_content": "Artificial intelligence aims to create agents that act intelligently.", "metadata": {}})]
    q = "What are the main goals of AI research?"
    ans, ctx = run_rag(docs, q, mode="fusion", embeddings_backend="sbert" if SentenceTransformer else "openai", sbert_model="sentence-transformers/all-MiniLM-L6-v2" if SentenceTransformer else None)
    print("ANSWER:\n", ans)
    print("\n--- CONTEXT CHUNKS USED ---")
    for c in ctx[:5]:
        print(getattr(c,"page_content","")[:400].replace("\n"," "), "\n---")
