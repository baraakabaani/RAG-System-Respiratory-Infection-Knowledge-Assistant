# RAG System — Respiratory Infection Knowledge Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** system that answers clinical questions about Influenza, COVID-19, and Tuberculosis using WHO guideline documents. Built with a full experimental pipeline to measure and improve retrieval quality.

> **Course:** CSAI-413 Natural Language Processing Applications  
> **Team:** Baraa Kabbani · Faisal Sahloul · Khalid Daqqaq · Salim Arnous · Yousef Osama

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Results](#key-results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Experiments](#experiments)
- [Installation & Usage](#installation--usage)
- [Skills Demonstrated](#skills-demonstrated)

---

## Overview

This system answers clinical questions like:

> *"What is the recommended oseltamivir dosage for adults with influenza?"*  
> *"Which corticosteroid is used for severe COVID-19 and at what dose?"*  
> *"What is the first-line treatment regimen for tuberculosis?"*

It grounds every answer in WHO-published guidelines and returns cited, traceable responses — eliminating hallucination through **hybrid retrieval** and a **structured clinical prompt**.

**Pipeline at a glance:**

```
WHO PDF Documents
      │
      ▼
Font-Aware Section Extraction (PyMuPDF)
      │
      ▼
Two-Stage Chunking (800 chars / 150 overlap)
      │
      ▼
Dense Embedding (all-MiniLM-L6-v2, 384-dim)
      │
      ├──────────────────────────┐
      ▼                          ▼
FAISS Dense Retrieval       BM25 Lexical Retrieval
(top-10)                    (top-10)
      │                          │
      └──────────┬───────────────┘
                 ▼
      Reciprocal Rank Fusion (RRF, k=60)
                 │
                 ▼
      Cross-Encoder Reranking (ms-marco, top-5)
                 │
                 ▼
      Groq LLaMA-3.1 Generation (Clinical Prompt)
                 │
                 ▼
      Cited Answer + Source Metadata
```

---

## System Architecture

### Baseline RAG (Deliverable 1)

A 6-stage pipeline built from scratch without RAG frameworks:

| Stage | Component | Detail |
|-------|-----------|--------|
| 1 | PDF Extraction | PyMuPDF with font-size heading detection (≥14pt) |
| 2 | Section Segmentation | Header-bounded sections with metadata (source, page, section) |
| 3 | Two-Stage Chunking | Short sections preserved intact; long sections split recursively |
| 4 | Embedding | `sentence-transformers/all-MiniLM-L6-v2`, L2-normalized |
| 5 | Vector Indexing | FAISS `IndexFlatIP` (cosine via inner product), persisted to disk |
| 6 | Dense Retrieval | `similarity_search_with_score(query, k=5)` |

### Improved RAG (Deliverable 2)

Adds three improvements validated by controlled experiments:

| Component | Method | Gain |
|-----------|--------|------|
| **Hybrid Retrieval** | Dense (FAISS) + Lexical (BM25) fused via Reciprocal Rank Fusion | +17% Recall@1 |
| **Cross-Encoder Reranking** | `ms-marco-MiniLM-L-6-v2` joint (query, chunk) scoring | +33% Recall@1 |
| **Structured Clinical Prompt** | Enforces source citation, refuses unanswerable queries | +90% citation rate |

---

## Key Results

Evaluated on a hand-labeled test set of **10 clinical queries** with ground-truth source documents.

| Metric | Baseline | Improved | Δ |
|--------|----------|----------|---|
| **Recall@1** | 0.60 | **0.80** | +33% |
| **Recall@3** | 0.90 | 0.80 | −11% |
| **Recall@5** | 0.90 | **0.90** | = |
| **Domain Accuracy@3** | 0.90 | **0.90** | = |
| **Grounding Score** | 0.61 | **0.71** | +16% |

> **Grounding Score** measures how many key clinical terms (e.g., "75 mg", "twice daily", "5 days") appear in the retrieved context — a proxy for hallucination risk.

### Experiment Highlights

- **Chunk size = 800 chars** was optimal; 400-char chunks fragmented clinical tables, 1200-char diluted embedding signal.
- **Cross-encoder reranking** fixed Q08 (COVID antivirals) and Q10 (IPC measures) which baseline ranked to wrong documents.
- **Hybrid RRF** fixed Q03 (COVID corticosteroid) — BM25 recovered exact drug names missed by dense retrieval.
- **Q09 (chronic cough, fever, night sweats)** remains a known failure: symptom overlap causes cross-disease contamination. Proposed fix: pre-retrieval disease domain classifier.

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **LLM** | Groq API · `llama-3.1-8b-instant` |
| **Embedding** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| **Vector DB** | FAISS `IndexFlatIP` (cosine similarity, persisted) |
| **Lexical Retrieval** | BM25 (`rank-bm25`) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Retrieval Fusion** | Reciprocal Rank Fusion (RRF, k=60) |
| **PDF Processing** | PyMuPDF (`fitz`) — font-aware header detection |
| **LLM Orchestration** | LangChain (text splitters, vectorstores, HuggingFace) |
| **Web App** | Streamlit (demo) · Flask (REST API backend) |
| **Visualization** | Plotly · Pandas |
| **PDF Reports** | ReportLab |

---

## Project Structure

```
anlp project/
│
├── dataset/                        # WHO source documents (4 PDFs)
│   ├── 9789240097759-eng.pdf       # Influenza clinical guidelines
│   ├── clinical-management-of-novel-cov.pdf  # COVID-19 SARI management
│   ├── policy-brief_covid-19_*.pdf # COVID-19 policy brief
│   └── evidence-gaps-tuberculosis-*.pdf      # TB evidence & research
│
├── baseline_rag.py                 # Baseline RAG — 6-stage pipeline
├── improved_rag.py                 # Improved RAG — hybrid + CE + LLM
├── evaluation_framework.py         # Shared test set (10 queries) & metrics
├── experiments.py                  # 4 controlled experiments
├── app.py                          # Streamlit interactive demo (4 tabs)
├── server.py                       # Flask REST API backend
├── generate_team_report.py         # PDF report generator (ReportLab)
│
├── faiss_index/                    # Baseline FAISS index (800-char chunks)
├── idx_400/ idx_1200/              # Experiment 1 indices
├── idx_improved/                   # Improved system index
├── experiment_results.json         # Results from all 4 experiments (39 KB)
│
└── templates/
    └── index.html                  # Flask frontend UI
```

---

## Experiments

Four controlled experiments, each isolating one variable:

### Experiment 1 — Chunk Size
Tested chunk sizes of 400, 800, and 1200 characters on 10 queries.

| Chunk Size | Recall@1 | Grounding |
|-----------|----------|-----------|
| 400 chars | 0.60 | 0.46 |
| **800 chars** | **0.60** | **0.61** |
| 1200 chars | 0.50 | 0.61 |

**Finding:** 400-char chunks fragment multi-line drug tables; 1200-char chunks dilute embedding signal. 800 chars is optimal.

---

### Experiment 2 — Cross-Encoder Reranking
Compared FAISS top-5 (baseline) vs. FAISS top-10 → CE rerank → top-5 (improved).

| System | Recall@1 |
|--------|----------|
| Baseline (no reranker) | 0.60 |
| Cross-Encoder Reranker | **0.80** |

**Finding:** +33% Recall@1. CE joint scoring eliminates false positives where cosine similarity picks a topically adjacent but factually wrong chunk.

---

### Experiment 3 — Hybrid Retrieval (RRF)
Compared dense-only, BM25-only, and Hybrid RRF retrieval.

| System | Recall@1 |
|--------|----------|
| Dense only (FAISS) | 0.60 |
| BM25 only | 0.50 |
| **Hybrid RRF** | **0.70** |

**Finding:** BM25 recovers exact clinical terms (drug names, dosages) missed by dense retrieval. RRF fairly merges both rank lists.

---

### Experiment 4 — Prompt Design
Evaluated three prompt templates on grounding score and citation rate.

| Template | Grounding | Citation Rate |
|----------|-----------|--------------|
| Generic | 0.52 | ~20% |
| **Structured Clinical** | **0.61** | **~90%** |
| Chain-of-Thought | 0.59 | ~85% |

**Finding:** Explicit instructions to cite sources and refuse unanswerable queries dramatically improve answer traceability.

---

## Installation & Usage

### Prerequisites

```bash
pip install pymupdf langchain langchain-text-splitters langchain-community \
            langchain-huggingface faiss-cpu sentence-transformers \
            rank-bm25 groq streamlit plotly pandas flask reportlab
```

### Set API Key (for LLM generation)

```bash
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_your_key_here"

# Linux / macOS
export GROQ_API_KEY="gsk_your_key_here"
```

Get a free key at [console.groq.com](https://console.groq.com).

### Run

```bash
# Baseline retrieval (no API key required)
python baseline_rag.py

# Improved RAG with LLM generation
python improved_rag.py
python improved_rag.py --query "What is the oseltamivir dose for adults?"
python improved_rag.py --compare   # side-by-side baseline vs improved

# Run all 4 experiments (saves experiment_results.json)
python experiments.py

# Streamlit interactive demo (4 tabs: Chat, Experiments, Comparison, Architecture)
streamlit run app.py

# Flask REST API backend (port 5000)
python server.py
```

### API Endpoints (Flask)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/ask` | Query the improved RAG system |
| `POST` | `/api/compare` | Run baseline vs improved side-by-side |
| `GET` | `/api/experiments` | Return all experiment results as JSON |
| `GET` | `/api/health` | Health check |

---

## Skills Demonstrated

This project covers the full NLP engineering stack — from raw PDF parsing to a deployed web demo:

**Information Retrieval**
- Dual retrieval: dense embedding search (FAISS) + lexical keyword matching (BM25)
- Score fusion via Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking for joint (query, document) relevance scoring
- FAISS index construction, persistence, and loading

**NLP & Embeddings**
- Sentence-level semantic embedding with `sentence-transformers`
- Font-aware PDF parsing with structural metadata extraction (PyMuPDF)
- Recursive character-level chunking with configurable overlap
- BM25 tokenization and term frequency scoring

**Large Language Models**
- Prompt engineering for clinical grounding (citation enforcement, refusal handling)
- Chain-of-thought prompting for multi-step reasoning
- LLM API integration (Groq) with structured output parsing
- Measuring and reducing hallucination via grounding score

**Experimental Design & Evaluation**
- Controlled ablation studies (chunk size, retrieval strategy, reranker, prompt)
- Custom evaluation metrics: Recall@K, Domain Accuracy, Grounding Score
- 10-query labeled test set with ground-truth source attribution
- Results serialized to JSON and visualized with Plotly

**Software Engineering**
- Modular Python design (baseline / improved / evaluation / experiments separated)
- Thread-safe lazy initialization in Flask (singleton pattern with `threading.Lock`)
- Streamlit session state and `@st.cache_resource` for persistent model loading
- REST API design with Flask (JSON contracts, error handling)
- ReportLab PDF report generation

**Tools & Ecosystem**
- LangChain (document abstraction, text splitters, vector store wrappers)
- FAISS (vector database, `IndexFlatIP`, cosine via L2-normalization)
- Streamlit (interactive data app with custom CSS theming)
- Flask (lightweight REST API server)
- Plotly (interactive experiment charts and radar plots)
- PyMuPDF (low-level PDF parsing with span-level font metadata)

---

## Knowledge Base

| Document | Domain | Key Clinical Content |
|----------|--------|----------------------|
| WHO Influenza Clinical Guidelines | Influenza | Oseltamivir 75 mg, baloxavir dosing, treatment duration |
| COVID-19 SARI Management | COVID-19 | Dexamethasone 6 mg, SpO2 thresholds, ICU protocols |
| COVID-19 Policy Brief | COVID-19 | Nirmatrelvir, molnupiravir, remdesivir for non-severe high-risk |
| TB Evidence Gaps & Research | Tuberculosis | 6-month isoniazid/rifampicin regimen, pediatric diagnosis |

---

## Known Limitations & Future Work

| Issue | Description | Proposed Fix |
|-------|-------------|-------------|
| **Q09 Cross-Disease Contamination** | Symptom overlap (cough, fever, night sweats) causes COVID chunks to outrank TB | Pre-retrieval disease domain classifier |
| **Q03 Incomplete Grounding** | Correct PDF retrieved but dosage table spans multiple chunks, not all in top-5 | Table-aware extraction with multi-chunk synthesis |
| **No Re-ranking Diversity** | CE reranker maximizes relevance but may return near-duplicate chunks | Maximal Marginal Relevance (MMR) for diversity |

---

*Built for CSAI-413 · Applied Natural Language Processing · 2025*
