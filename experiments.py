"""
experiments.py
--------------
Four controlled experiments for Deliverable 2.

Run:  python experiments.py
      Results are printed to terminal AND saved to results.json

Experiments
-----------
  EXP 1 -- Chunk Size       : 400 / 800 (baseline) / 1200 characters
  EXP 2 -- Cross-Encoder    : No reranker vs. ms-marco cross-encoder
  EXP 3 -- Hybrid Retrieval : Dense-only vs. BM25-only vs. BM25+Dense (RRF)
  EXP 4 -- Prompt Design    : Generic vs. Clinical Structured vs. Chain-of-Thought

Set your Groq API key before running:
  $env:GROQ_API_KEY = "gsk_..."
"""

import os, sys, json, time
from pathlib import Path

# ?? shared pipeline imports ????????????????????????????????????????????????????
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import groq as groq_module

from evaluation_framework import (
    TEST_QUERIES, evaluate_system, print_summary, hallucination_score
)

DATASET_DIR  = "./dataset"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL   = "llama-3.1-8b-instant"
RESULTS_FILE = "./experiment_results.json"

# ?? Shared embedding object (built once, reused) ??????????????????????????????
print("[INIT] Loading embedding model...")
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("[INIT] Embedding model ready.\n")

# ?? Shared cross-encoder (loaded once) ???????????????????????????????????????
print("[INIT] Loading cross-encoder reranker...")
CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("[INIT] Cross-encoder ready.\n")


# ==============================================================================
# PDF LOADING  (shared across experiments)
# ==============================================================================

def extract_sections(pdf_path: str) -> list[dict]:
    doc  = fitz.open(pdf_path)
    name = os.path.basename(pdf_path)
    sections, cur_header, cur_body, cur_page = [], "Introduction", [], 1
    for page_num, page in enumerate(doc, 1):
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0: continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text      = span["text"].strip()
                    size      = span["size"]
                    is_bold   = bool(span["flags"] & (1 << 4))
                    is_head   = ((size >= 14) or (is_bold and size >= 12)) and len(text) >= 5
                    if is_head:
                        if cur_body:
                            sections.append({"header": cur_header,
                                             "body": " ".join(cur_body).strip(),
                                             "source": name, "page": cur_page})
                        cur_header, cur_body, cur_page = text, [], page_num
                    elif text:
                        cur_body.append(text)
    if cur_body:
        sections.append({"header": cur_header,
                         "body": " ".join(cur_body).strip(),
                         "source": name, "page": cur_page})
    doc.close()
    return sections


def build_docs(chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "])
    docs = []
    for pdf in sorted(Path(DATASET_DIR).glob("*.pdf")):
        for sec in extract_sections(str(pdf)):
            body = sec["body"]
            if not body: continue
            meta = {"source": sec["source"], "page": sec["page"],
                    "section_header": sec["header"]}
            if len(body) <= chunk_size:
                docs.append(Document(page_content=body, metadata=meta))
            else:
                for i, chunk in enumerate(splitter.split_text(body)):
                    docs.append(Document(page_content=chunk,
                                         metadata={**meta, "chunk_index": i}))
    return docs


def build_index(docs: list[Document], index_path: str) -> FAISS:
    if Path(index_path).exists():
        print(f"  [cache] Loading existing index: {index_path}")
        return FAISS.load_local(index_path, EMBEDDINGS,
                                allow_dangerous_deserialization=True)
    print(f"  [build] Embedding {len(docs)} chunks -> {index_path}")
    vs = FAISS.from_documents(docs, EMBEDDINGS)
    vs.save_local(index_path)
    return vs


# ==============================================================================
# EXPERIMENT 1 -- Chunk Size
# Hypothesis: 400-char chunks improve Source Recall@1 for dosage-lookup queries
# because each chunk focuses on a single table row / recommendation sentence.
# However, 1200-char chunks may improve Grounding Score for broad queries by
# keeping recommendation + rationale together.
# ==============================================================================

def run_exp1_chunk_size() -> dict:
    print("\n" + "="*65)
    print("  EXPERIMENT 1: Chunk Size (400 / 800 / 1200 chars)")
    print("="*65)

    configs = [
        (400,  100, "./idx_400",  "Chunk-400"),
        (800,  150, "./faiss_index", "Chunk-800 (Baseline)"),
        (1200, 200, "./idx_1200", "Chunk-1200"),
    ]

    summaries = []
    for csize, overlap, idx_path, label in configs:
        print(f"\n  -> Config: {label}")
        docs = build_docs(csize, overlap)
        print(f"    Chunks produced: {len(docs)}")
        vs   = build_index(docs, idx_path)

        def retrieve(q, vs=vs):
            return vs.similarity_search_with_score(q, k=5)

        s = evaluate_system(vs, retrieve, label)
        print_summary(s)
        summaries.append(s)

    return {"experiment": "Chunk Size", "configs": [
        {k: v for k, v in s.items() if k != "rows"} for s in summaries
    ], "per_query": {s["label"]: s["rows"] for s in summaries}}


# ==============================================================================
# EXPERIMENT 2 -- Cross-Encoder Reranking
# Hypothesis: Reranking FAISS top-10 with a cross-encoder improves Source
# Recall@1 because the cross-encoder jointly scores (query, chunk) pairs
# instead of comparing independently encoded vectors.
# ==============================================================================

def rerank(query: str, results: list, k: int = 5) -> list:
    """Apply cross-encoder to FAISS top-N, return top-k reranked."""
    pairs  = [(query, doc.page_content) for doc, _ in results]
    scores = CROSS_ENCODER.predict(pairs)
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [(doc, float(ce_score)) for (doc, _faiss_score), ce_score in ranked[:k]]


def run_exp2_reranking() -> dict:
    print("\n" + "="*65)
    print("  EXPERIMENT 2: Cross-Encoder Reranking (No Reranker vs. CE)")
    print("="*65)

    docs = build_docs(800, 150)
    vs   = build_index(docs, "./faiss_index")

    # Baseline: FAISS top-5
    def retrieve_baseline(q):
        return vs.similarity_search_with_score(q, k=5)

    # Improved: FAISS top-10 -> cross-encoder rerank -> top-5
    def retrieve_reranked(q):
        candidates = vs.similarity_search_with_score(q, k=10)
        return rerank(q, candidates, k=5)

    s_base = evaluate_system(vs, retrieve_baseline, "No Reranker (Baseline)")
    s_ce   = evaluate_system(vs, retrieve_reranked,  "Cross-Encoder Reranked")
    print_summary(s_base)
    print_summary(s_ce)

    return {"experiment": "Cross-Encoder Reranking",
            "baseline":  {k: v for k, v in s_base.items() if k != "rows"},
            "improved":  {k: v for k, v in s_ce.items()   if k != "rows"},
            "per_query": {"baseline": s_base["rows"], "reranked": s_ce["rows"]}}


# ==============================================================================
# EXPERIMENT 3 -- Hybrid Retrieval (BM25 + Dense)
# Hypothesis: Hybrid retrieval outperforms dense-only on drug-name queries
# (exact lexical match) while dense-only outperforms BM25-only on semantic
# queries. Reciprocal Rank Fusion combines both advantages.
# ==============================================================================

class BM25Retriever:
    def __init__(self, docs: list[Document]):
        self.docs    = docs
        tokenized    = [d.page_content.lower().split() for d in docs]
        self.bm25    = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 5) -> list:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_k  = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [(self.docs[i], float(s)) for i, s in top_k]


def reciprocal_rank_fusion(dense_results: list, bm25_results: list,
                            k: int = 5, rrf_k: int = 60) -> list:
    """Combine two ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, (doc, _) in enumerate(dense_results):
        key = doc.page_content[:80]
        scores[key]  = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        doc_map[key] = doc

    for rank, (doc, _) in enumerate(bm25_results):
        key = doc.page_content[:80]
        scores[key]  = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(doc_map[key], score) for key, score in ranked]


def run_exp3_hybrid() -> dict:
    print("\n" + "="*65)
    print("  EXPERIMENT 3: Hybrid Retrieval (Dense / BM25 / BM25+Dense RRF)")
    print("="*65)

    docs    = build_docs(800, 150)
    vs      = build_index(docs, "./faiss_index")
    bm25_r  = BM25Retriever(docs)

    def retrieve_dense(q):
        return vs.similarity_search_with_score(q, k=5)

    def retrieve_bm25(q):
        return bm25_r.retrieve(q, k=5)

    def retrieve_hybrid(q):
        dense = vs.similarity_search_with_score(q, k=10)
        bm25  = bm25_r.retrieve(q, k=10)
        return reciprocal_rank_fusion(dense, bm25, k=5)

    s_dense  = evaluate_system(vs, retrieve_dense,  "Dense-Only (Baseline)")
    s_bm25   = evaluate_system(vs, retrieve_bm25,   "BM25-Only")
    s_hybrid = evaluate_system(vs, retrieve_hybrid, "Hybrid BM25+Dense (RRF)")
    print_summary(s_dense)
    print_summary(s_bm25)
    print_summary(s_hybrid)

    return {"experiment": "Hybrid Retrieval",
            "dense":  {k: v for k, v in s_dense.items()  if k != "rows"},
            "bm25":   {k: v for k, v in s_bm25.items()   if k != "rows"},
            "hybrid": {k: v for k, v in s_hybrid.items() if k != "rows"},
            "per_query": {"dense": s_dense["rows"],
                          "bm25":  s_bm25["rows"],
                          "hybrid":s_hybrid["rows"]}}


# ==============================================================================
# EXPERIMENT 4 -- Prompt Design (requires Groq API key)
# Hypothesis: A structured clinical prompt that explicitly instructs the LLM
# to cite sources and refuse unanswerable questions reduces hallucination and
# improves grounding compared to a generic prompt.
# ==============================================================================

PROMPT_GENERIC = """\
Context:
{context}

Question: {query}

Answer:"""

PROMPT_CLINICAL = """\
You are a clinical assistant that answers questions ONLY based on provided WHO guidelines.

GUIDELINES CONTEXT:
{context}

CLINICAL QUESTION: {query}

Instructions:
- Answer using ONLY information from the context above.
- Cite the source document and section for every factual claim.
- Include specific values (dosages, durations, thresholds) exactly as stated.
- If the answer is not in the context, respond with: "This information is not available in the provided WHO guidelines."

ANSWER:"""

PROMPT_COT = """\
You are a medical knowledge assistant. Use the following WHO guideline excerpts to answer the question.

CONTEXT:
{context}

QUESTION: {query}

Think step by step:
1. Identify which disease domain this question relates to (influenza / COVID-19 / TB).
2. Locate the most relevant guideline section in the context.
3. Extract the specific clinical recommendation or value.
4. Cite the source document.

ANSWER:"""


def call_groq(prompt: str, client) -> str:
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[API ERROR: {e}]"


def run_exp4_prompts() -> dict:
    print("\n" + "="*65)
    print("  EXPERIMENT 4: Prompt Design (Generic / Clinical / Chain-of-Thought)")
    print("="*65)

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("  [SKIP] GROQ_API_KEY not set. Skipping prompt experiment.")
        print("  Set it with:  $env:GROQ_API_KEY = 'gsk_...'")
        return {"experiment": "Prompt Design", "skipped": True}

    client  = groq_module.Groq(api_key=api_key)
    docs    = build_docs(800, 150)
    vs      = build_index(docs, "./faiss_index")

    # Use a representative subset (first 5 queries) to stay within rate limits
    eval_queries = TEST_QUERIES[:5]

    configs = [
        ("Generic Prompt",           PROMPT_GENERIC),
        ("Clinical Structured Prompt", PROMPT_CLINICAL),
        ("Chain-of-Thought Prompt",   PROMPT_COT),
    ]

    results = {}
    for prompt_name, prompt_template in configs:
        print(f"\n  -> Prompt: {prompt_name}")
        rows = []
        for tq in eval_queries:
            # Retrieve top-5 chunks
            retrieved = vs.similarity_search_with_score(tq["query"], k=5)
            context   = "\n\n---\n\n".join(
                f"[{doc.metadata.get('source','?')}, p.{doc.metadata.get('page','?')}]\n"
                f"{doc.page_content}"
                for doc, _ in retrieved
            )
            prompt  = prompt_template.format(context=context, query=tq["query"])
            answer  = call_groq(prompt, client)
            gnd     = hallucination_score(answer, context, tq["key_terms"])
            cites   = any(kw in answer for kw in
                          ["source", "Source", "pdf", ".pdf", "page", "Page",
                           "section", "Section", "guideline"])
            refuses = "not available" in answer.lower() or "not found" in answer.lower()

            row = {
                "id":           tq["id"],
                "query":        tq["query"][:50] + "...",
                "grounding":    round(gnd, 2),
                "cites_source": cites,
                "appropriate_refusal": refuses if tq["category"] == "failure_case" else None,
                "answer_preview": answer[:200],
            }
            rows.append(row)
            print(f"    {tq['id']}: grounding={gnd:.2f}  cites={cites}")
            time.sleep(0.5)   # respect rate limits

        avg_gnd   = sum(r["grounding"]    for r in rows) / len(rows)
        avg_cites = sum(r["cites_source"] for r in rows) / len(rows)
        results[prompt_name] = {"rows": rows,
                                 "avg_grounding":    round(avg_gnd,   2),
                                 "avg_cites_source": round(avg_cites, 2)}
        print(f"    Average grounding: {avg_gnd:.2f} | Avg cites: {avg_cites:.2f}")

    return {"experiment": "Prompt Design", "results": results}


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    all_results = {}

    all_results["exp1"] = run_exp1_chunk_size()
    all_results["exp2"] = run_exp2_reranking()
    all_results["exp3"] = run_exp3_hybrid()
    all_results["exp4"] = run_exp4_prompts()

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[DONE] All results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
