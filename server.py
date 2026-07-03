"""
server.py -- Flask backend for the WHO Knowledge Assistant demo
Run: python server.py
Then open: http://localhost:5000
"""

import os, json, time, threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

app = Flask(__name__)

# ── Lazy-load the RAG systems once on first request ───────────────────────────
_rag = None
_baseline_vs = None
_lock = threading.Lock()

def get_rag():
    global _rag
    if _rag is None:
        with _lock:
            if _rag is None:
                from improved_rag import ImprovedRAG
                _rag = ImprovedRAG()
    return _rag

def get_baseline():
    global _baseline_vs
    if _baseline_vs is None:
        with _lock:
            if _baseline_vs is None:
                from langchain_community.vectorstores import FAISS
                from langchain_huggingface import HuggingFaceEmbeddings
                emb = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                _baseline_vs = FAISS.load_local(
                    "./faiss_index", emb, allow_dangerous_deserialization=True
                )
    return _baseline_vs


def doc_to_dict(doc, score):
    meta = doc.metadata
    src  = meta.get("source", "")
    return {
        "source":  src,
        "short":   short_source(src),
        "domain":  get_domain(src),
        "color":   get_color(src),
        "page":    meta.get("page", "?"),
        "section": meta.get("section_header", "")[:70],
        "score":   round(float(score), 4),
        "content": doc.page_content[:400],
    }


def short_source(src):
    if "9789240097759"  in src: return "Influenza Guidelines"
    if "tuberculosis"   in src: return "TB Evidence Gaps"
    if "policy-brief"   in src: return "COVID-19 Policy Brief"
    if "clinical-management" in src: return "COVID-19 SARI Management"
    return src[:35]

def get_domain(src):
    if "9789240097759"  in src: return "Influenza"
    if "tuberculosis"   in src: return "Tuberculosis"
    if "policy-brief"   in src: return "COVID-19 Policy"
    return "COVID-19"

def get_color(src):
    if "9789240097759"  in src: return "#1F497D"
    if "tuberculosis"   in src: return "#7B3F00"
    return "#155724"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def ask():
    data  = request.json
    query = data.get("query", "").strip()
    key   = data.get("groq_key", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    if key:
        os.environ["GROQ_API_KEY"] = key

    t0  = time.time()
    rag = get_rag()
    results  = rag.retrieve(query)
    sources  = [doc_to_dict(d, s) for d, s in results]
    context  = rag._format_context(results)
    answer   = rag.generate(query, context)
    elapsed  = round(time.time() - t0, 1)

    return jsonify({"answer": answer, "sources": sources, "elapsed": elapsed})


@app.route("/api/compare", methods=["POST"])
def compare():
    data  = request.json
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    rag      = get_rag()
    baseline = get_baseline()

    base_raw = baseline.similarity_search_with_score(query, k=5)
    impr_raw = rag.retrieve(query)

    return jsonify({
        "baseline": [doc_to_dict(d, s) for d, s in base_raw],
        "improved": [doc_to_dict(d, s) for d, s in impr_raw],
    })


@app.route("/api/experiments")
def experiments():
    path = Path("./experiment_results.json")
    if not path.exists():
        return jsonify({"error": "Run experiments.py first"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("\n  WHO Knowledge Assistant")
    print("  Open: http://localhost:5000\n")
    app.run(debug=False, port=5000)
