# 📚 RAG From Scratch — Streamlit App


<p align="center">
  <a href="https://github.com/crimsonKn1ght/rag-llm/stargazers">
    <img src="https://img.shields.io/github/stars/crimsonKn1ght/rag-llm?style=for-the-badge" alt="GitHub stars">
  </a>
  <a href="https://github.com/crimsonKn1ght/rag-llm/network/members">
    <img src="https://img.shields.io/github/forks/crimsonKn1ght/rag-llm?style=for-the-badge" alt="GitHub forks">
  </a>
  <a href="https://github.com/crimsonKn1ght/rag-llm/graphs/commit-activity">
    <img src="https://img.shields.io/maintenance/yes/2026?style=for-the-badge" alt="Maintained">
  </a>
  <a href="https://github.com/crimsonKn1ght/rag-llm">
    <img src="https://img.shields.io/github/languages/top/crimsonKn1ght/rag-llm?style=for-the-badge" alt="Language">
  </a>
</p>

A modular Retrieval-Augmented Generation (RAG) system with advanced features.

## Features

- Multiple retrieval modes: basic, multiquery, fusion, HyDE, CRAG, self-RAG
- Semantic routing for multi-index systems
- Reranking with cross-encoders
- Support for OpenAI and Sentence-Transformers embeddings
- Long-context chunk merging
- Streamlit web interface

## Installation

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up your environment variables:

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env
```

Your `.env` file should contain:

```
OPENAI_API_KEY=your-actual-api-key-here
```

## Usage

### As a Python Package

```python
from rag_system import run_rag, load_docs_from_url

# Load documents
docs = load_docs_from_url("https://example.com/article")

# Run RAG query
answer, context = run_rag(
    docs=docs,
    question="What is this article about?",
    mode="fusion",  # or: basic, multiquery, hyde, crag, self-rag
    embeddings_backend="openai",  # or: sbert
)

print(answer)
```

### Streamlit App

```bash
streamlit run rag_app.py
```

### As a Module

```bash
python -m rag_system
```

## Configuration

### Embedding Backends

- **openai**: Uses OpenAI embeddings (requires `OPENAI_API_KEY`)
- **sbert**: Uses Sentence-Transformers (runs locally, no API key needed)

### Retrieval Modes

| Mode | Description |
|------|-------------|
| `basic` | Single query retrieval |
| `multiquery` | Generates multiple query variants |
| `fusion` | RAG-Fusion with reciprocal rank fusion |
| `hyde` | Hypothetical Document Embeddings |
| `crag` | Contextual Retrieval with query expansion |
| `self-rag` | Self-reflective RAG |

## Project Structure

```
├── rag_system/
│   ├── __init__.py          # Main exports
│   ├── __main__.py          # CLI entry point
│   ├── core/                # Config and dependencies
│   ├── embeddings/          # Embedding backends
│   ├── loaders/             # Document loaders
│   ├── indexing/            # Vector store builders
│   ├── retrieval/           # Reranking and helpers
│   ├── chains/              # RAG chains
│   ├── routing/             # Semantic routing, CRAG
│   └── utils/               # Utilities
├── rag_app.py               # Streamlit interface
├── requirements.txt
├── .env.example             # Environment template
└── .gitignore
```

## Security Notes

- Never commit your `.env` file (it's in `.gitignore`)
- Use `.env.example` as a template for required variables
- Keep your API keys secure