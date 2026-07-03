"""
baseline_rag.py
---------------
Baseline Retrieval-Augmented Generation (RAG) system for the
Respiratory Infection Knowledge Assistant.

Course  : CSAI-413 Natural Language Processing Applications
Team    : Baraa Kabbani (22001553), Faisal Sahloul (22000669),
          Khalid Daqqaq (22000958), Salim Arnous (22001301),
          Yousef Osama (22000959)

Pipeline overview:
  1. Load 4 WHO PDFs from ./dataset using PyMuPDF
  2. Detect headings via font-size metadata → segment into sections
  3. Two-stage chunking: keep short sections intact; split large ones
     with RecursiveCharacterTextSplitter respecting paragraph/sentence bounds
  4. Embed chunks with HuggingFace all-MiniLM-L6-v2 (local, no API key)
  5. Index in FAISS (cosine similarity via L2-normalised inner product)
  6. Query function retrieves top-k chunks with full provenance metadata

Dependencies (install before running):
  pip install pymupdf langchain langchain-text-splitters langchain-community
              langchain-huggingface faiss-cpu sentence-transformers
"""

import os
import sys
from pathlib import Path

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_DIR = "./dataset"          # folder containing the 4 WHO PDFs
INDEX_PATH  = "./faiss_index"      # directory where the FAISS index is saved

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking parameters (calibrated for WHO typography and 512-token model limit)
CHUNK_SIZE    = 800   # characters; ≈ 150-200 tokens
CHUNK_OVERLAP = 150   # characters; preserves cross-boundary context

# Heading detection thresholds (empirically tuned against WHO PDFs)
HEADING_MIN_FONT_SIZE  = 14    # any text ≥ 14pt is treated as a heading
HEADING_BOLD_FONT_SIZE = 12    # bold text ≥ 12pt is also treated as a heading
HEADING_MIN_TEXT_LEN   = 5     # ignore very short "headings" (e.g., page numbers)

# Retrieval
DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Stage 1 & 2 — PDF Loading and Header-Aware Section Extraction
# ---------------------------------------------------------------------------

def extract_sections_from_pdf(pdf_path: str) -> list[dict]:
    """
    Open a WHO PDF and segment it into header-bounded sections.

    Uses PyMuPDF span-level font metadata (size + bold flag) to identify
    headings without relying on fragile regex patterns.  Each section
    inherits the heading text as its metadata label for provenance tracking.

    Returns
    -------
    List of dicts:  {header: str, body: str, source: str, page: int}
    """
    doc = fitz.open(pdf_path)
    source_name = os.path.basename(pdf_path)

    sections: list[dict] = []
    current_header = "Introduction"
    current_body:  list[str] = []
    current_page   = 1

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] != 0:
                # Skip image blocks
                continue

            for line in block["lines"]:
                for span in line["spans"]:
                    text      = span["text"].strip()
                    font_size = span["size"]
                    # PyMuPDF encodes bold in bit 4 of the flags integer
                    is_bold   = bool(span["flags"] & (1 << 4))

                    is_heading = (
                        (font_size >= HEADING_MIN_FONT_SIZE) or
                        (is_bold and font_size >= HEADING_BOLD_FONT_SIZE)
                    ) and len(text) >= HEADING_MIN_TEXT_LEN

                    if is_heading:
                        # Persist the completed section before opening a new one
                        body_text = " ".join(current_body).strip()
                        if body_text:
                            sections.append({
                                "header": current_header,
                                "body":   body_text,
                                "source": source_name,
                                "page":   current_page,
                            })
                        current_header = text
                        current_body   = []
                        current_page   = page_num
                    else:
                        if text:
                            current_body.append(text)

    # Flush the last open section
    body_text = " ".join(current_body).strip()
    if body_text:
        sections.append({
            "header": current_header,
            "body":   body_text,
            "source": source_name,
            "page":   current_page,
        })

    doc.close()
    return sections


# ---------------------------------------------------------------------------
# Stage 3 — Two-Stage Chunking → LangChain Documents
# ---------------------------------------------------------------------------

def build_documents(dataset_dir: str) -> list[Document]:
    """
    Load all PDFs from dataset_dir, extract sections, apply two-stage
    chunking, and return a flat list of LangChain Document objects.

    Stage 1: sections ≤ CHUNK_SIZE characters are kept as single chunks
             (preserves recommendation boxes and dosage tables intact).
    Stage 2: oversized sections are split with RecursiveCharacterTextSplitter
             using a separator priority list that respects paragraph and
             sentence boundaries before resorting to word-level splits.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    all_documents: list[Document] = []
    pdf_files = sorted(Path(dataset_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"[ERROR] No PDF files found in '{dataset_dir}'. "
              "Check that DATASET_DIR points to the correct folder.")
        sys.exit(1)

    for pdf_path in pdf_files:
        print(f"\n[LOAD] Processing: {pdf_path.name}")
        sections = extract_sections_from_pdf(str(pdf_path))
        print(f"       Sections detected : {len(sections)}")

        doc_chunks = 0
        for section in sections:
            body = section["body"]
            if not body:
                continue

            metadata = {
                "source":         section["source"],
                "page":           section["page"],
                "section_header": section["header"],
            }

            if len(body) <= CHUNK_SIZE:
                # Stage 1: section fits in one chunk
                all_documents.append(Document(page_content=body, metadata=metadata))
                doc_chunks += 1
            else:
                # Stage 2: recursively split and tag each sub-chunk
                sub_chunks = splitter.split_text(body)
                for i, chunk_text in enumerate(sub_chunks):
                    chunk_meta = {**metadata, "chunk_index": i}
                    all_documents.append(
                        Document(page_content=chunk_text, metadata=chunk_meta)
                    )
                doc_chunks += len(sub_chunks)

        print(f"       Chunks produced   : {doc_chunks}")

    print(f"\n[INFO] Total chunks across all PDFs: {len(all_documents)}")
    return all_documents


# ---------------------------------------------------------------------------
# Stage 4 & 5 — Embedding + FAISS Indexing
# ---------------------------------------------------------------------------

def build_or_load_vectorstore(
    documents: list[Document],
    index_path: str,
) -> FAISS:
    """
    Build a FAISS vector store from documents (embedding with all-MiniLM-L6-v2)
    or load an existing index from index_path if it already exists.

    Embeddings are computed locally — no data leaves the machine.
    The index is serialized with allow_dangerous_deserialization because
    it is produced and consumed entirely by this script on the same machine.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # enables cosine via inner product
    )

    if Path(index_path).exists():
        print(f"\n[INDEX] Loading existing FAISS index from '{index_path}'...")
        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"[INDEX] Index loaded successfully.")
    else:
        print(f"\n[INDEX] Building new FAISS index — this may take a minute...")
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(index_path)
        print(f"[INDEX] Index saved to '{index_path}'.")

    return vectorstore


# ---------------------------------------------------------------------------
# Stage 6 — Retrieval Query Interface
# ---------------------------------------------------------------------------

def query_rag(vectorstore: FAISS, query: str, k: int = DEFAULT_TOP_K) -> list[Document]:
    """
    Retrieve the top-k most similar chunks for a natural language query.

    Prints a formatted report showing cosine similarity scores and full
    provenance metadata for each retrieved chunk.  Returns the raw
    Document list for downstream LLM prompting.
    """
    results = vectorstore.similarity_search_with_score(query, k=k)

    print(f"\n{'='*65}")
    print(f"  QUERY: {query}")
    print(f"{'='*65}")

    for rank, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        print(f"\n  [Rank {rank}]  Similarity Score : {score:.4f}")
        print(f"  {'─'*60}")
        print(f"  Source  : {meta.get('source', 'N/A')}")
        print(f"  Page    : {meta.get('page', 'N/A')}")
        print(f"  Section : {meta.get('section_header', 'N/A')}")
        if "chunk_index" in meta:
            print(f"  Chunk # : {meta['chunk_index']}")
        preview = doc.page_content[:400].replace("\n", " ")
        print(f"  Content : {preview}{'...' if len(doc.page_content) > 400 else ''}")

    print(f"\n{'='*65}\n")
    return [doc for doc, _ in results]


def format_context_for_llm(retrieved_docs: list[Document]) -> str:
    """
    Format retrieved chunks into a structured context string suitable
    for injection into an LLM prompt.
    """
    context_parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        meta = doc.metadata
        citation = (
            f"[Source {i}: {meta.get('source', 'N/A')}, "
            f"Page {meta.get('page', 'N/A')}, "
            f"Section: {meta.get('section_header', 'N/A')}]"
        )
        context_parts.append(f"{citation}\n{doc.page_content}")
    return "\n\n---\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# Main — Build Index and Run Evaluation Queries
# ---------------------------------------------------------------------------

EVALUATION_QUERIES = [
    # Query 1: Influenza antiviral dosage (targets 9789240097759-eng.pdf)
    "What is the recommended oseltamivir dosage for an adult patient with confirmed influenza?",

    # Query 2: TB treatment regimen (targets evidence-gaps TB PDF)
    "What is the standard first-line treatment duration for drug-susceptible tuberculosis according to WHO?",

    # Query 3: COVID-19 corticosteroid (targets clinical-management-of-novel-cov.pdf)
    "What corticosteroid is recommended for severe COVID-19 and at what dose?",

    # Query 4: Cross-disease failure case (should exhibit retrieval contamination)
    "How should I manage a patient with chronic cough, fever, and night sweats?",
]


def main() -> None:
    print("=" * 65)
    print("  Respiratory Infection Knowledge Assistant — Baseline RAG")
    print("=" * 65)

    # ── Stage 1–3: Load PDFs and build chunks ──────────────────────────────
    documents = build_documents(DATASET_DIR)

    if not documents:
        print("[ERROR] No documents were produced. Check the dataset directory.")
        sys.exit(1)

    # ── Stage 4–5: Build or load FAISS index ──────────────────────────────
    vectorstore = build_or_load_vectorstore(documents, INDEX_PATH)

    # ── Stage 6: Run evaluation queries ───────────────────────────────────
    print("\n[EVAL] Running evaluation queries...\n")

    for query in EVALUATION_QUERIES:
        retrieved = query_rag(vectorstore, query, k=DEFAULT_TOP_K)

        # Format context that would be sent to an LLM
        context = format_context_for_llm(retrieved)
        print("  [CONTEXT FOR LLM — first 500 chars]")
        print(f"  {context[:500].replace(chr(10), ' ')}")
        print()

    print("[DONE] Baseline RAG evaluation complete.")
    print(f"       FAISS index stored at: {Path(INDEX_PATH).resolve()}")


if __name__ == "__main__":
    main()
