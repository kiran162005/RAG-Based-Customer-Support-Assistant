# 🚀 RAG-Based Customer Support Assistant  
### Using LangGraph & Human-in-the-Loop (HITL)

This project implements a **Retrieval-Augmented Generation (RAG)** system designed for customer support use cases.  
It retrieves relevant information from a PDF knowledge base and generates accurate, context-aware responses using a local LLM.

---

## 📌 Problem Statement

Traditional chatbots:
- Provide generic responses  
- Fail for paraphrased queries  
- May hallucinate incorrect answers  

This system solves these issues by:
- Retrieving real information from documents  
- Generating grounded responses  
- Escalating uncertain queries to a human  

---

## 🧠 Key Features

- 📄 PDF-based knowledge retrieval  
- 🔍 Semantic search using embeddings  
- 🧠 Context-aware response generation  
- 🔄 Graph-based workflow using LangGraph  
- ⚖️ Decision-based routing (Answer / Escalate)  
- 👨‍💻 Human-in-the-Loop (HITL) escalation  
- 💻 Fully local setup (no API cost)

---

## 🏗️ System Architecture

Flow:

User → LangGraph → Retriever → LLM → Decision → Output / HITL

---

## 🔄 Data Flow

### Ingestion Phase
PDF → Loader → Chunking → Embeddings → ChromaDB

### Query Phase
User Query → Retriever → Context → LLM → Response → Decision

Decision → Answer / Escalation

---

## ⚙️ Tech Stack

- Python  
- LangChain  
- LangGraph  
- ChromaDB  
- Ollama (Mistral - Local LLM)  
- Sentence Transformers  

---

