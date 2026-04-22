from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

from langgraph.graph import StateGraph

# -------------------------------
# LOAD + PROCESS DOCUMENT
# -------------------------------
loader = PyPDFLoader("data/knowledge_base.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)
chunks = text_splitter.split_documents(documents)

print(f"Loaded {len(chunks)} chunks")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./chroma_db"
)

retriever = vector_db.as_retriever(search_kwargs={"k": 2})

llm = OllamaLLM(model="mistral", timeout=60)

# -------------------------------
# STATE (IMPORTANT FOR GRAPH)
# -------------------------------
from typing import TypedDict

class State(TypedDict):
    query: str
    answer: str
    decision: str

