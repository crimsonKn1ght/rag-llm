import logging

DEFAULT_PROMPT = (
  "You are a helpful assistant. Use ONLY the provided context to answer.\n"
  "If the answer is not in the context, say you don't know.\n\n"
  "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:"
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")
