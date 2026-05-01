"""
RAG Customer Support Assistant — Flask + LangGraph backend
==========================================================
Endpoints:
  POST /chat          — query the RAG pipeline
  POST /upload        — add a PDF to the knowledge base
  POST /remove-doc    — remove a PDF and rebuild the index
  POST /agent-reply   — human agent submits a response
  GET  /docs          — list indexed documents
  GET  /health        — liveness check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import threading, os, shutil, time

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "data"
CHROMA_DIR = "./chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Pipeline singleton (lazy-loaded, thread-safe) ─────────────────────────────
_graph     = None
_retriever = None
_llm       = None
_lock      = threading.Lock()


def _reset_pipeline():
    global _graph, _retriever, _llm
    with _lock:
        _graph = _retriever = _llm = None


def get_pipeline():
    """
    Build (or return cached) LangGraph pipeline.
    Loads ALL PDFs from data/ directory.
    """
    global _graph, _retriever, _llm
    if _graph is not None:
        return _graph, _retriever, _llm

    with _lock:
        if _graph is not None:
            return _graph, _retriever, _llm

        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_ollama import OllamaLLM
        from langgraph.graph import StateGraph
        from typing import TypedDict, List

        # ── 1. Load all PDFs ──────────────────────────────────────────────────
        all_docs = []
        for fname in sorted(os.listdir(UPLOAD_DIR)):
            if fname.lower().endswith(".pdf"):
                fpath = os.path.join(UPLOAD_DIR, fname)
                try:
                    loader = PyPDFLoader(fpath)
                    pages = loader.load()
                    all_docs.extend(pages)
                    print(f"  ✓ Loaded {len(pages)} pages from {fname}")
                except Exception as e:
                    print(f"  ✗ Failed to load {fname}: {e}")

        if not all_docs:
            from langchain.schema import Document
            all_docs = [Document(page_content="No documents loaded.", metadata={})]
            print("  ⚠ No PDFs found in data/ — using stub document")

        # ── 2. Chunk ──────────────────────────────────────────────────────────
        # chunk_size=500, overlap=50 gives better context than 300/30
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(all_docs)
        print(f"  → {len(chunks)} chunks created")

        # ── 3. Embed & store ──────────────────────────────────────────────────
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=CHROMA_DIR,
        )
        _retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}   # retrieve top-3 for better coverage
        )

        # ── 4. LLM ───────────────────────────────────────────────────────────
        _llm = OllamaLLM(model="mistral", timeout=90)

        # ── 5. LangGraph state & nodes ────────────────────────────────────────
        class State(TypedDict):
            query:      str
            context:    str          # retrieved chunks joined
            sources:    List[str]    # page/source metadata
            answer:     str
            decision:   str          # ANSWER | ESCALATE
            confidence: str          # HIGH | LOW

        # Node 1 — Input: validate & retrieve
        def input_node(state: State) -> State:
            query = state.get("query", "").strip()
            if not query:
                state["decision"]   = "ESCALATE"
                state["answer"]     = "Empty query received."
                state["confidence"] = "LOW"
                state["context"]    = ""
                state["sources"]    = []
                return state

            docs = _retriever.invoke(query)

            if not docs:
                state["decision"]   = "ESCALATE"
                state["answer"]     = "No relevant information found."
                state["confidence"] = "LOW"
                state["context"]    = ""
                state["sources"]    = []
                return state

            state["context"] = "\n\n".join([d.page_content for d in docs])
            state["sources"] = list({
                d.metadata.get("source", d.metadata.get("file_path", "Unknown"))
                for d in docs
            })
            state["decision"]   = "CONTINUE"
            state["confidence"] = "HIGH"
            return state

        # Node 2 — Process: generate answer with LLM
        def process_node(state: State) -> State:
            if state.get("decision") == "ESCALATE":
                return state

            query   = state["query"]
            context = state["context"]

            prompt = f"""You are a helpful customer support assistant.
Answer the question using ONLY the information in the context below.
If the answer is not in the context, respond with exactly: NOT_FOUND

Context:
{context}

Question: {query}

Answer:"""

            response      = _llm.invoke(prompt)
            response_text = response.strip()
            lower         = response_text.lower()

            # Escalation conditions
            escalate = (
                "not_found"            in lower or
                "not available"        in lower or
                "does not contain"     in lower or
                "no information"       in lower or
                "i don't know"         in lower or
                "cannot find"          in lower or
                len(response_text)     < 40
            )

            if escalate:
                state["decision"]   = "ESCALATE"
                state["confidence"] = "LOW"
            else:
                state["decision"]   = "ANSWER"
                state["confidence"] = "HIGH"

            state["answer"] = response_text
            return state

        # Node 3 — Output: finalise (web layer handles actual delivery)
        def output_node(state: State) -> State:
            return state

        # ── 6. Build graph ────────────────────────────────────────────────────
        builder = StateGraph(State)
        builder.add_node("input",   input_node)
        builder.add_node("process", process_node)
        builder.add_node("output",  output_node)

        builder.set_entry_point("input")
        builder.add_edge("input",   "process")
        builder.add_edge("process", "output")

        _graph = builder.compile()
        print("  ✓ LangGraph pipeline ready (3 nodes: input → process → output)")

    return _graph, _retriever, _llm


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    data  = request.json or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    t0 = time.time()
    try:
        graph, _, _ = get_pipeline()
        result = graph.invoke({
            "query":      query,
            "context":    "",
            "sources":    [],
            "answer":     "",
            "decision":   "",
            "confidence": "",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    latency = round((time.time() - t0) * 1000)

    return jsonify({
        "answer":     result.get("answer", ""),
        "decision":   result.get("decision", "ANSWER"),
        "confidence": result.get("confidence", "HIGH"),
        "sources":    result.get("sources", []),
        "latency_ms": latency,
        "flow":       ["Input", "Process", "Output"]
                      if result.get("decision") != "ESCALATE"
                      else ["Input", "Process", "HITL Escalate"],
    })


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    dest = os.path.join(UPLOAD_DIR, file.filename)
    file.save(dest)

    # Wipe Chroma so it rebuilds cleanly with the new doc
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    _reset_pipeline()
    return jsonify({"status": "ok", "filename": file.filename})


@app.route("/remove-doc", methods=["POST"])
def remove_doc():
    data     = request.json or {}
    filename = data.get("filename", "").strip()
    if not filename:
        return jsonify({"error": "No filename"}), 400

    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        os.remove(path)

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    _reset_pipeline()
    return jsonify({"status": "ok"})


@app.route("/agent-reply", methods=["POST"])
def agent_reply():
    data  = request.json or {}
    reply = data.get("reply", "").strip()
    if not reply:
        return jsonify({"error": "Empty reply"}), 400
    return jsonify({"answer": reply, "decision": "AGENT"})


