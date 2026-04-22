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

# -------------------------------
# NODE 1: PROCESSING
# -------------------------------
def process_node(state):
    query = state.get("query", "")

    docs = retriever.invoke(query)

    if not docs:
        state["decision"] = "ESCALATE"
        state["answer"] = "No relevant information found."
        return state

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer ONLY using the context.
If answer is not in context, say: NOT_FOUND

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    response_text = response.strip().lower()

    # 🚨 STRONG ESCALATION CONDITIONS
    if (
        "not_found" in response_text or
        "not available" in response_text or
        "does not contain" in response_text or
        "no information" in response_text or
        len(response_text) < 40
    ):
        state["decision"] = "ESCALATE"
    else:
        state["decision"] = "ANSWER"

    state["answer"] = response
    return state

