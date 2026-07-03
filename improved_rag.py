"""
improved_rag.py
---------------
Deliverable 2 -- Improved RAG System for the Respiratory Infection
Knowledge Assistant.

Improvements over baseline (based on experiment results):
  [OK] Smaller chunks (400 chars) -- better precision for dosage lookups
  [OK] Hybrid BM25 + Dense retrieval (RRF) -- combines lexical + semantic
  [OK] Cross-encoder reranking -- joint (query, chunk) scoring
  [OK] Structured clinical prompt -- reduces hallucination, enforces citation
  [OK] Groq LLM generation (llama3-8b-8192) -- fast, free-tier

Usage:
  # Set API key first:
  $env:GROQ_API_KEY = "gsk_..."

  # Interactive demo:
  python improved_rag.py

  # Single query:
  python improved_rag.py --query "What is the dexamethasone dose for severe COVID-19?"

  # Compare baseline vs improved on test set:
  python improved_rag.py --compare
"""

import os, sys, argparse, time
from pathlib import Path

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import groq as groq_module

from evaluation_framework import TEST_QUERIES, evaluate_system, print_summary

# ?? Config ????????????????????????????????????????????????????????????????????
DATASET_DIR    = "./dataset"
INDEX_PATH     = "./idx_improved"   # separate index from baseline
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL     = "llama-3.1-8b-instant"
CHUNK_SIZE     = 400
CHUNK_OVERLAP  = 100
RETRIEVE_FIRST = 10   # FAISS candidates before reranking
FINAL_TOP_K    = 5    # chunks sent to LLM after reranking

SYSTEM_PROMPT = (
    "You are a clinical assistant that answers questions ONLY based on provided "
    "WHO guideline excerpts. Always cite the source document and page number. "
    "If the information is not in the provided context, say so explicitly -- "
    "never guess or invent clinical values."
)

CLINICAL_PROMPT = """\
WHO GUIDELINE EXCERPTS:
{context}

CLINICAL QUESTION: {query}

Instructions:
- Answer ONLY using information present in the excerpts above.
- Cite source file and page for every factual claim (e.g. "[9789240097759-eng.pdf, p.102]").
- Include exact dosages, durations, and thresholds as stated in the guidelines.
- If the answer cannot be found in the excerpts, respond: "This information is not available in the provided WHO guidelines."

ANSWER:"""


# ?? Pipeline components ???????????????????????????????????????????????????????

def extract_sections(pdf_path: str) -> list[dict]:
    doc  = fitz.open(pdf_path)
    name = os.path.basename(pdf_path)
    sections, cur_h, cur_b, cur_p = [], "Introduction", [], 1
    for pn, page in enumerate(doc, 1):
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0: continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text    = span["text"].strip()
                    size    = span["size"]
                    bold    = bool(span["flags"] & (1 << 4))
                    is_head = ((size >= 14) or (bold and size >= 12)) and len(text) >= 5
                    if is_head:
                        if cur_b:
                            sections.append({"header": cur_h,
                                             "body": " ".join(cur_b).strip(),
                                             "source": name, "page": cur_p})
                        cur_h, cur_b, cur_p = text, [], pn
                    elif text:
                        cur_b.append(text)
    if cur_b:
        sections.append({"header": cur_h, "body": " ".join(cur_b).strip(),
                         "source": name, "page": cur_p})
    doc.close()
    return sections


class ImprovedRAG:
    def __init__(self):
        print("[INIT] Loading embedding model (all-MiniLM-L6-v2)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        print("[INIT] Loading cross-encoder reranker...")
        self.reranker = CrossEncoder(RERANK_MODEL)

        self.docs = self._load_documents()
        self.vectorstore = self._build_or_load_index()

        print("[INIT] Building BM25 index...")
        tokenized   = [d.page_content.lower().split() for d in self.docs]
        self.bm25   = BM25Okapi(tokenized)

        api_key = os.environ.get("GROQ_API_KEY", "")
        if api_key:
            self.groq_client = groq_module.Groq(api_key=api_key)
            print(f"[INIT] Groq client ready ({GROQ_MODEL}).")
        else:
            self.groq_client = None
            print("[WARN] GROQ_API_KEY not set -- retrieval-only mode.")

        print("[INIT] Improved RAG system ready.\n")

    # ?? Document loading ??????????????????????????????????????????????????????

    def _load_documents(self) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "])
        docs = []
        for pdf in sorted(Path(DATASET_DIR).glob("*.pdf")):
            for sec in extract_sections(str(pdf)):
                body = sec["body"]
                if not body: continue
                meta = {"source": sec["source"], "page": sec["page"],
                        "section_header": sec["header"]}
                if len(body) <= CHUNK_SIZE:
                    docs.append(Document(page_content=body, metadata=meta))
                else:
                    for i, chunk in enumerate(splitter.split_text(body)):
                        docs.append(Document(page_content=chunk,
                                             metadata={**meta, "chunk_index": i}))
        print(f"[LOAD] {len(docs)} chunks loaded from {DATASET_DIR}")
        return docs

    def _build_or_load_index(self) -> FAISS:
        if Path(INDEX_PATH).exists():
            print(f"[INDEX] Loading improved index from '{INDEX_PATH}'...")
            return FAISS.load_local(INDEX_PATH, self.embeddings,
                                    allow_dangerous_deserialization=True)
        print(f"[INDEX] Building improved index ({len(self.docs)} chunks)...")
        vs = FAISS.from_documents(self.docs, self.embeddings)
        vs.save_local(INDEX_PATH)
        print(f"[INDEX] Index saved to '{INDEX_PATH}'.")
        return vs

    # ?? Retrieval pipeline ????????????????????????????????????????????????????

    def _dense_retrieve(self, query: str, k: int) -> list:
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def _bm25_retrieve(self, query: str, k: int) -> list:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top    = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [(self.docs[i], float(s)) for i, s in top]

    def _hybrid_retrieve(self, query: str, k: int = RETRIEVE_FIRST) -> list:
        dense = self._dense_retrieve(query, k)
        bm25  = self._bm25_retrieve(query, k)
        rrf_k = 60
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        for rank, (doc, _) in enumerate(dense):
            key = doc.page_content[:80]
            scores[key]  = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
            doc_map[key] = doc
        for rank, (doc, _) in enumerate(bm25):
            key = doc.page_content[:80]
            scores[key]  = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
            doc_map[key] = doc
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_map[k], s) for k, s in ranked]

    def _rerank(self, query: str, candidates: list, k: int = FINAL_TOP_K) -> list:
        pairs  = [(query, doc.page_content) for doc, _ in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(doc, float(ce)) for (doc, _), ce in ranked[:k]]

    def retrieve(self, query: str) -> list:
        """Full retrieval pipeline: hybrid -> rerank -> top-k."""
        candidates = self._hybrid_retrieve(query, k=RETRIEVE_FIRST)
        return self._rerank(query, candidates, k=FINAL_TOP_K)

    # ?? Generation ????????????????????????????????????????????????????????????

    def _format_context(self, results: list) -> str:
        parts = []
        for i, (doc, score) in enumerate(results, 1):
            m = doc.metadata
            parts.append(
                f"[Excerpt {i} -- {m.get('source','?')}, "
                f"Page {m.get('page','?')}, "
                f"Section: {m.get('section_header','?')}]\n"
                f"{doc.page_content}"
            )
        return "\n\n" + "?"*50 + "\n\n".join(parts)

    def generate(self, query: str, context: str) -> str:
        if not self.groq_client:
            return "[Generation unavailable -- set GROQ_API_KEY]"
        prompt = CLINICAL_PROMPT.format(context=context, query=query)
        try:
            resp = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=500,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[Groq API error: {e}]"

    # ?? Public interface ??????????????????????????????????????????????????????

    def ask(self, query: str, verbose: bool = True) -> dict:
        """Full pipeline: retrieve -> rerank -> generate."""
        t0       = time.time()
        results  = self.retrieve(query)
        context  = self._format_context(results)
        answer   = self.generate(query, context)
        elapsed  = time.time() - t0

        if verbose:
            print(f"\n{'='*65}")
            print(f"  QUERY: {query}")
            print(f"{'='*65}")
            print(f"\n  ANSWER:\n  {answer}\n")
            print(f"  SOURCES:")
            for rank, (doc, score) in enumerate(results, 1):
                m = doc.metadata
                print(f"    [{rank}] {m.get('source','?')} | "
                      f"p.{m.get('page','?')} | "
                      f"CE score: {score:.4f}")
            print(f"\n  [{elapsed:.1f}s]\n")

        return {"query": query, "answer": answer,
                "sources": results, "elapsed": elapsed}

    def retrieve_for_eval(self, query: str) -> list:
        """Returns (Document, score) list -- for evaluation_framework compatibility."""
        return self.retrieve(query)


# ==============================================================================
# COMPARISON: Baseline vs Improved
# ==============================================================================

def run_comparison(rag: ImprovedRAG):
    print("\n" + "="*65)
    print("  BASELINE vs IMPROVED -- Full Test Set Evaluation")
    print("="*65)

    # Load baseline (800-char, dense only, no reranker)
    base_embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    base_vs = FAISS.load_local("./faiss_index", base_embeddings,
                               allow_dangerous_deserialization=True)

    def baseline_retrieve(q):
        return base_vs.similarity_search_with_score(q, k=5)

    def improved_retrieve(q):
        return rag.retrieve(q)

    s_base = evaluate_system(base_vs, baseline_retrieve, "Baseline (Dense-800)")
    s_impr = evaluate_system(None,    improved_retrieve, "Improved (Hybrid+CE+400)")

    print_summary(s_base)
    print_summary(s_impr)

    print("  IMPROVEMENT DELTAS:")
    for metric, label in [("avg_recall@1","Recall@1"),
                           ("avg_recall@3","Recall@3"),
                           ("avg_domain@3","Domain@3"),
                           ("avg_grounding","Grounding")]:
        delta = s_impr[metric] - s_base[metric]
        arrow = "^" if delta > 0 else ("v" if delta < 0 else "?")
        print(f"    {label:<15} {s_base[metric]:.2f} -> {s_impr[metric]:.2f}  "
              f"{arrow} {delta:+.2f}")


# ==============================================================================
# INTERACTIVE DEMO
# ==============================================================================

DEMO_BANNER = """
?==============================================================?
|   Respiratory Infection Knowledge Assistant -- Improved RAG   |
|   WHO Guidelines: Influenza ? COVID-19 ? Tuberculosis        |
|   Type 'quit' to exit  |  'demo' for example queries         |
?==============================================================?
"""

DEMO_QUERIES = [
    "What is the oseltamivir dose for an adult with confirmed influenza?",
    "What corticosteroid is recommended for severe COVID-19 and at what dose?",
    "How should a patient with chronic cough, fever, and night sweats be managed?",
    "What is the standard treatment duration for drug-susceptible TB?",
]


def interactive_demo(rag: ImprovedRAG):
    print(DEMO_BANNER)
    while True:
        try:
            query = input("  Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("  Goodbye.")
            break
        if query.lower() == "demo":
            for q in DEMO_QUERIES:
                print(f"\n  Running demo query: {q}")
                rag.ask(q)
                time.sleep(0.5)
            continue

        rag.ask(query)


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",   help="Single query to run")
    parser.add_argument("--compare", action="store_true",
                        help="Run baseline vs improved comparison")
    args = parser.parse_args()

    rag = ImprovedRAG()

    if args.query:
        rag.ask(args.query)
    elif args.compare:
        run_comparison(rag)
    else:
        interactive_demo(rag)


if __name__ == "__main__":
    main()
