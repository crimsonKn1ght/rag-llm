# 📚 RAG From Scratch — Streamlit App


<p align="center">
  <a href="https://github.com/crimsonKn1ght/rag-llm/stargazers">
    <img src="https://img.shields.io/github/stars/crimsonKn1ght/rag-llm?style=for-the-badge" alt="GitHub stars">
  </a>
  <a href="https://github.com/crimsonKn1ght/rag-llm/network/members">
    <img src="https://img.shields.io/github/forks/crimsonKn1ght/rag-llm?style=for-the-badge" alt="GitHub forks">
  </a>
  <a href="https://github.com/crimsonKn1ght/rag-llm/graphs/commit-activity">
    <img src="https://img.shields.io/maintenance/yes/2025?style=for-the-badge" alt="Maintained">
  </a>
  <a href="https://github.com/crimsonKn1ght/rag-llm">
    <img src="https://img.shields.io/github/languages/top/crimsonKn1ght/rag-llm?style=for-the-badge" alt="Language">
  </a>
</p>


A simple yet powerful **Retrieval-Augmented Generation (RAG)** demo built with **LangChain**, **Chroma**, and **Streamlit** — from scratch.

This app lets you load web pages or upload files, create vector embeddings (using OpenAI or SentenceTransformers), and ask questions that are **grounded in retrieved context**.

---

## 🚀 Features

✅ Upload `.txt`, `.md`, and `.pdf` files  
✅ Load from a web URL  
✅ Choose between **OpenAI** or **SBERT (GPU)** embeddings  
✅ Adjustable chunk size, overlap, and top-k retrieval  
✅ View retrieved text chunks and the generated answer  
✅ Optional Chroma persistence (reusable vector DB)  
✅ Works locally — no external database required  

---

## 🧠 Tech Stack

- **LangChain** – for document loading, text splitting, and RAG pipelines  
- **Chroma** – local vector store for embeddings  
- **Streamlit** – clean, interactive web interface  
- **OpenAI / SBERT** – for embeddings and LLM responses  

---

## 🛠️ Installation

```bash
# 1. Clone your project folder
git clone <your_repo_url>
cd rag_project

# 2. (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
