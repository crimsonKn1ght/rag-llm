import streamlit as st
import os
from pathlib import Path
import tempfile
from rag_core import (
    load_docs_from_url,
    load_docs_from_paths,
    run_rag,
    build_embedding_fn
)

st.set_page_config(
    page_title="Document Q&A System",
    page_icon="🔍",
    layout="wide"
)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'docs' not in st.session_state:
    st.session_state.docs = None

st.title("🔍 Document Q&A System")

with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Data Source")
    source_type = st.radio("Select input type:", ["URL", "File Upload"])
    
    if source_type == "URL":
        url_input = st.text_input("Enter URL:", placeholder="https://example.com/document")
        if st.button("Load URL"):
            if url_input:
                with st.spinner("Loading document..."):
                    try:
                        st.session_state.docs = load_docs_from_url(url_input)
                        st.session_state.documents_loaded = True
                        st.success(f"Loaded {len(st.session_state.docs)} document(s)")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    else:
        uploaded_files = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'md']
        )
        if st.button("Process Files"):
            if uploaded_files:
                with st.spinner("Processing files..."):
                    try:
                        temp_paths = []
                        for file in uploaded_files:
                            suffix = Path(file.name).suffix
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                tmp.write(file.getvalue())
                                temp_paths.append(Path(tmp.name))
                        
                        st.session_state.docs = load_docs_from_paths(temp_paths)
                        st.session_state.documents_loaded = True
                        st.success(f"Processed {len(uploaded_files)} file(s)")
                        
                        for p in temp_paths:
                            p.unlink()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    st.divider()
    
    st.subheader("Model Settings")
    
    embedding_backend = st.selectbox(
        "Embedding Model:",
        ["openai", "sbert"]
    )
    
    if embedding_backend == "sbert":
        sbert_model = st.text_input(
            "SBERT Model:",
            value="sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        sbert_model = None
    
    mode = st.selectbox(
        "Retrieval Mode:",
        ["basic", "multiquery", "fusion", "hyde", "crag", "self-rag"]
    )
    
    with st.expander("Advanced Options"):
        model_name = st.text_input("LLM Model:", value="gpt-4o-mini")
        k = st.slider("Documents to retrieve:", 1, 10, 4)
        chunk_size = st.slider("Chunk size:", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk overlap:", 0, 500, 200)
        temperature = st.slider("Temperature:", 0.0, 1.0, 0.0)
        rerank = st.checkbox("Enable reranking", value=True)
        router = st.checkbox("Enable routing", value=False)
    
    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.rerun()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ask Questions")
    
    if not st.session_state.documents_loaded:
        st.info("Load documents from the sidebar to begin")
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        question = st.chat_input("Ask a question about your documents...")
        
        if question:
            with st.chat_message("user"):
                st.write(question)
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            with st.chat_message("assistant"):
                with st.spinner("Searching documents..."):
                    try:
                        answer, context_docs = run_rag(
                            docs=st.session_state.docs,
                            question=question,
                            mode=mode,
                            embeddings_backend=embedding_backend,
                            sbert_model=sbert_model,
                            model_name=model_name,
                            temperature=temperature,
                            k=k,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            rerank=rerank,
                            router_enabled=router
                        )
                        st.write(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        
                        with st.expander("View source chunks"):
                            for i, doc in enumerate(context_docs[:3]):
                                st.text_area(
                                    f"Source {i+1}",
                                    getattr(doc, "page_content", str(doc))[:500],
                                    height=100
                                )
                    except Exception as e:
                        error_msg = f"Error processing query: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

with col2:
    st.subheader("System Status")
    
    if st.session_state.documents_loaded:
        st.success("✓ Documents loaded")
        st.metric("Total Documents", len(st.session_state.docs))
        
        total_chars = sum(len(getattr(doc, "page_content", "")) for doc in st.session_state.docs)
        st.metric("Total Characters", f"{total_chars:,}")
    else:
        st.warning("No documents loaded")
    
    st.divider()
    
    st.subheader("Active Settings")
    st.text(f"Mode: {mode}")
    st.text(f"Embeddings: {embedding_backend}")
    st.text(f"LLM: {model_name}")
    st.text(f"Retrieval: {k} docs")
    st.text(f"Rerank: {'On' if rerank else 'Off'}")
    st.text(f"Router: {'On' if router else 'Off'}")

st.divider()
st.caption("Powered by advanced retrieval techniques")
