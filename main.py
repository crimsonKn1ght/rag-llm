from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OLlamaEmbeddings
from langchain_core.output_parsers import StrOutputParser

pdf_file = "this.pdf"
model = "llama3.1"

loader = PyPDFLoader(pdf_file)
pages = loader.load()

print(f"Number of pages: {len(pages)}")
print(f"Length of page: {len(pages[1].page_content)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

chunks = splitter.split_document(pages)

embeddings = OLlamaEmbeddings(model=model)
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever()
retriever.invoke("Tell me about the document")

model = ChatOllama(model=model, temperature=0)
model.invoke("Tell me about the document")

parser = StrOutputParser()

chain = model | parser 
print(chain.invoke("Who is the president of the United States?"))