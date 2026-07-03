"""
generate_team_report.py
-----------------------
Generates Team_Contributions.pdf — a detailed per-member breakdown
of roles, design decisions, and exact contributions for Deliverable 1.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas


# ── Brand colours ────────────────────────────────────────────────────────────
NAVY        = colors.HexColor("#1F497D")
STEEL       = colors.HexColor("#2E74B5")
TEAL        = colors.HexColor("#00838F")
LIGHT_BLUE  = colors.HexColor("#EEF4FB")
LIGHT_GREY  = colors.HexColor("#F3F4F6")
MID_GREY    = colors.HexColor("#6B7280")
DARK        = colors.HexColor("#1A1A2E")
WHITE       = colors.white
ACCENT      = colors.HexColor("#C7254E")   # inline code red

PAGE_W, PAGE_H = A4


# ── Header / Footer canvas ────────────────────────────────────────────────────
def on_page(canvas_obj, doc):
    canvas_obj.saveState()
    # Top bar
    canvas_obj.setFillColor(NAVY)
    canvas_obj.rect(0, PAGE_H - 1.1*cm, PAGE_W, 1.1*cm, fill=1, stroke=0)
    canvas_obj.setFont("Helvetica-Bold", 8)
    canvas_obj.setFillColor(WHITE)
    canvas_obj.drawString(1.5*cm, PAGE_H - 0.75*cm,
        "CSAI-413 NLP Applications  |  Deliverable 1 — Team Contributions Report")
    canvas_obj.drawRightString(PAGE_W - 1.5*cm, PAGE_H - 0.75*cm,
        "Respiratory Infection Knowledge Assistant")

    # Bottom bar
    canvas_obj.setFillColor(NAVY)
    canvas_obj.rect(0, 0, PAGE_W, 0.85*cm, fill=1, stroke=0)
    canvas_obj.setFont("Helvetica", 7)
    canvas_obj.setFillColor(WHITE)
    canvas_obj.drawString(1.5*cm, 0.28*cm, "Confidential — University Submission")
    canvas_obj.drawRightString(PAGE_W - 1.5*cm, 0.28*cm,
        f"Page {doc.page}")
    canvas_obj.restoreState()


# ── Style factory ─────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        "cover_title": S("cover_title",
            fontName="Helvetica-Bold", fontSize=26, textColor=NAVY,
            leading=32, alignment=TA_CENTER, spaceAfter=8),

        "cover_sub": S("cover_sub",
            fontName="Helvetica", fontSize=13, textColor=STEEL,
            leading=18, alignment=TA_CENTER, spaceAfter=6),

        "cover_meta": S("cover_meta",
            fontName="Helvetica", fontSize=10, textColor=MID_GREY,
            alignment=TA_CENTER, spaceAfter=4),

        "section_label": S("section_label",
            fontName="Helvetica-Bold", fontSize=9, textColor=TEAL,
            spaceAfter=2, spaceBefore=14, leading=12),

        "member_name": S("member_name",
            fontName="Helvetica-Bold", fontSize=17, textColor=NAVY,
            leading=22, spaceAfter=2, spaceBefore=4),

        "member_id": S("member_id",
            fontName="Helvetica", fontSize=10, textColor=MID_GREY,
            spaceAfter=10),

        "role_title": S("role_title",
            fontName="Helvetica-BoldOblique", fontSize=11, textColor=STEEL,
            leading=15, spaceAfter=6),

        "body": S("body",
            fontName="Helvetica", fontSize=10, textColor=DARK,
            leading=15, alignment=TA_JUSTIFY, spaceAfter=6),

        "bullet": S("bullet",
            fontName="Helvetica", fontSize=10, textColor=DARK,
            leading=14, leftIndent=14, firstLineIndent=-10,
            spaceAfter=4, spaceBefore=1),

        "sub_bullet": S("sub_bullet",
            fontName="Helvetica", fontSize=9.5, textColor=MID_GREY,
            leading=13, leftIndent=28, firstLineIndent=-10,
            spaceAfter=3),

        "decision_q": S("decision_q",
            fontName="Helvetica-Bold", fontSize=10, textColor=NAVY,
            leading=14, spaceBefore=8, spaceAfter=2),

        "decision_a": S("decision_a",
            fontName="Helvetica", fontSize=10, textColor=DARK,
            leading=14, leftIndent=12, alignment=TA_JUSTIFY, spaceAfter=5),

        "callout": S("callout",
            fontName="Helvetica-Oblique", fontSize=9.5,
            textColor=colors.HexColor("#374151"),
            leading=14, leftIndent=14, rightIndent=14,
            spaceAfter=8, spaceBefore=4),

        "toc_entry": S("toc_entry",
            fontName="Helvetica", fontSize=11, textColor=DARK,
            leading=18, leftIndent=8, spaceAfter=2),

        "toc_name": S("toc_name",
            fontName="Helvetica-Bold", fontSize=11, textColor=NAVY,
            leading=18, leftIndent=8, spaceAfter=2),
    }


# ── Reusable layout helpers ───────────────────────────────────────────────────

def hr(color=STEEL, thickness=0.6):
    return HRFlowable(width="100%", thickness=thickness,
                      color=color, spaceAfter=6, spaceBefore=4)


def section_header(text, styles):
    return Paragraph(text.upper(), styles["section_label"])


def bullet(text, styles, sub=False):
    key = "sub_bullet" if sub else "bullet"
    prefix = "◦  " if sub else "▸  "
    return Paragraph(f"{prefix}{text}", styles[key])


def callout_box(text, styles):
    return Paragraph(f"<i>{text}</i>", styles["callout"])


def member_banner(name, student_id, role_title, color, styles):
    """Coloured banner card for a team member."""
    data = [[
        Paragraph(f'<font color="white"><b>{name}</b></font>', styles["member_name"]),
        Paragraph(f'<font color="white">{student_id}</font>', styles["member_id"]),
    ]]
    # single-row table used as a coloured card
    banner = Table(data, colWidths=["70%", "30%"])
    banner.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), color),
        ("TOPPADDING",   (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 14),
        ("LEFTPADDING",  (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",        (1, 0), (1, 0), "RIGHT"),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    return banner


def decision_block(question, answer, styles):
    return [
        Paragraph(question, styles["decision_q"]),
        Paragraph(answer, styles["decision_a"]),
    ]


def choice_table(rows, col_w=None):
    """Two-column 'Option considered / Reason rejected' table."""
    header = [
        Paragraph("<b>Option Considered</b>", ParagraphStyle(
            "th", fontName="Helvetica-Bold", fontSize=9,
            textColor=WHITE, alignment=TA_CENTER)),
        Paragraph("<b>Decision / Reason</b>", ParagraphStyle(
            "th", fontName="Helvetica-Bold", fontSize=9,
            textColor=WHITE, alignment=TA_CENTER)),
    ]
    table_data = [header] + [
        [Paragraph(a, ParagraphStyle("td_a", fontName="Helvetica-Bold",
                                     fontSize=9, textColor=NAVY, leading=13)),
         Paragraph(b, ParagraphStyle("td_b", fontName="Helvetica",
                                     fontSize=9, textColor=DARK, leading=13))]
        for a, b in rows
    ]
    cw = col_w or ["38%", "62%"]
    t = Table(table_data, colWidths=cw, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1,  0), NAVY),
        ("BACKGROUND",    (0, 1), (-1, -1), LIGHT_BLUE),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT_BLUE, WHITE]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    return t


# ── Member data ───────────────────────────────────────────────────────────────

MEMBERS = [

    # ── 1. Baraa Kabbani ──────────────────────────────────────────────────────
    {
        "name": "Baraa Kabbani",
        "id": "Student ID: 22001553",
        "role": "Project Lead & System Architect",
        "color": colors.HexColor("#1F497D"),
        "summary": (
            "Baraa served as the project lead for Deliverable 1, responsible for the "
            "overall system design, pipeline integration, and the FAISS vector store "
            "component. He defined the five-stage architecture, managed task delegation "
            "across the team, and ensured that individual components integrated correctly "
            "into a single runnable pipeline."
        ),
        "choices": [
            {
                "q": "Why was FAISS chosen as the vector store over ChromaDB?",
                "a": (
                    "Baraa evaluated both FAISS and ChromaDB. ChromaDB requires an embedded "
                    "server process to run, which adds operational overhead and a non-trivial "
                    "dependency for a local research baseline. FAISS, by contrast, operates as "
                    "a pure library call with no server — it is instantiated, queried, and "
                    "serialized entirely in-process. FAISS also produces deterministic results "
                    "with IndexFlatIP (exact brute-force search), which is important for a "
                    "baseline evaluation where reproducibility matters. At our corpus size of "
                    "1,131 chunks, the difference in retrieval speed between exact and "
                    "approximate methods is negligible."
                ),
            },
            {
                "q": "Why was IndexFlatIP (inner product) used rather than IndexFlatL2?",
                "a": (
                    "With L2-normalized embeddings (normalize_embeddings=True), the inner "
                    "product between two vectors equals their cosine similarity. Baraa chose "
                    "this combination deliberately so that retrieval scores are interpretable "
                    "as cosine similarities — a value close to 1.0 means near-identical "
                    "semantic content, while values near 0.0 indicate unrelated passages. "
                    "IndexFlatL2 would compute Euclidean distances instead, which are harder "
                    "to interpret and compare across queries."
                ),
            },
            {
                "q": "Why was the pipeline built as a single script rather than separate modules?",
                "a": (
                    "For a baseline deliverable, a single self-contained script maximises "
                    "reproducibility — any team member or evaluator can run it with one "
                    "command. Baraa made the deliberate choice to defer modularisation to "
                    "Deliverable 2, where the pipeline will need to be extended with a "
                    "reranker, query classifier, and LLM generation endpoint. Premature "
                    "abstraction at the baseline stage would have added complexity without "
                    "delivering value."
                ),
            },
        ],
        "choice_table": [
            ("ChromaDB", "Rejected — requires an embedded server process; adds operational overhead for a local baseline"),
            ("Pinecone / Weaviate", "Rejected — cloud-hosted; data privacy concern for WHO clinical content; requires API key"),
            ("FAISS IndexFlatL2", "Rejected — L2 distance scores are less interpretable than cosine similarity for ablation studies"),
            ("FAISS IndexHNSWFlat", "Deferred — approximate; non-deterministic; reserved for Deliverable 2 at larger scale"),
            ("FAISS IndexFlatIP (chosen)", "Selected — exact, deterministic cosine similarity on L2-normalised vectors; zero server dependency"),
        ],
        "contributions": [
            "Defined the five-stage RAG pipeline architecture (Preprocessing → Chunking → Embedding → Indexing → Retrieval) and communicated it to the team as the project blueprint.",
            "Authored the build_or_load_vectorstore() function, which handles both initial FAISS index construction from document embeddings and subsequent fast loading from disk, avoiding redundant embedding computation on repeated runs.",
            "Implemented the query_rag() and format_context_for_llm() functions — the retrieval interface that displays ranked results with similarity scores, provenance metadata, and 400-character content previews.",
            "Designed the main() entry point with the four evaluation queries, including the cross-disease failure case query, and structured the terminal output format.",
            "Configured the HuggingFaceEmbeddings object with normalize_embeddings=True and device='cpu', ensuring cosine similarity semantics without GPU dependency.",
            "Managed GitHub repository structure, handled merge conflicts between team members' branches, and validated end-to-end pipeline runs before submission.",
            "Reviewed all code written by other team members for correctness and integration compatibility prior to the final pipeline run.",
            "Wrote Section 3 (System Architecture) and Section 5 (Baseline Implementation) of Deliverable_1.md in collaboration with Salim.",
        ],
    },

    # ── 2. Faisal Sahloul ─────────────────────────────────────────────────────
    {
        "name": "Faisal Sahloul",
        "id": "Student ID: 22000669",
        "role": "PDF Preprocessing & Chunking Engineer",
        "color": colors.HexColor("#155724"),
        "summary": (
            "Faisal was responsible for the most technically nuanced component of the "
            "pipeline: extracting clean, structured text from four heavily formatted WHO "
            "PDF documents and segmenting it into semantically coherent chunks. He "
            "designed and empirically calibrated the heading detection heuristic, "
            "implemented the two-stage chunking strategy, and wrote the extract_sections_from_pdf() "
            "and build_documents() functions that form the foundation of the entire pipeline."
        ),
        "choices": [
            {
                "q": "Why use PyMuPDF's span-level font metadata rather than regex on raw text?",
                "a": (
                    "Faisal initially attempted regex-based heading detection — looking for "
                    "patterns like numbered sections ('1.', '1.1', 'Chapter X'). This "
                    "approach failed because WHO documents use inconsistent numbering across "
                    "the four PDFs: the influenza guideline uses decimal numbering while the "
                    "TB evidence gaps document uses Roman-numeral appendices and unnumbered "
                    "section headers. PyMuPDF's span-level API exposes the font size and "
                    "bold flag for every character span, which is document-format agnostic — "
                    "a heading is a heading regardless of how it is numbered, because it "
                    "renders in a larger or bolder font. This approach correctly detected "
                    "402 sections in the influenza PDF versus only ~30 with regex."
                ),
            },
            {
                "q": "How were the heading thresholds (14pt, bold+12pt) determined?",
                "a": (
                    "Faisal ran a diagnostic pass over all four PDFs, printing every unique "
                    "(font_size, is_bold) combination and its associated text. He found that "
                    "WHO's body text consistently renders at 10–11pt, table content at 9–10pt, "
                    "footnotes at 8pt, primary headings at 14–18pt, and subsection headings "
                    "as bold 12pt text. The thresholds of ≥14pt for any text and bold+≥12pt "
                    "were set to capture both heading levels while excluding table headers "
                    "(which are bold but render at 9–10pt) and footnote references."
                ),
            },
            {
                "q": "Why use 800 characters as the chunk size rather than 512 tokens?",
                "a": (
                    "The embedding model (all-MiniLM-L6-v2) accepts up to 512 tokens. "
                    "800 characters corresponds to approximately 150–200 tokens for English "
                    "medical prose — well within the model limit. Faisal chose characters "
                    "rather than tokens as the split unit because character counting is "
                    "instantaneous (no tokeniser call required) and the relationship between "
                    "characters and tokens is stable enough for English text that the "
                    "embedding limit is never approached. Using a token-based splitter would "
                    "have required loading the tokeniser as an additional dependency during "
                    "the chunking stage."
                ),
            },
        ],
        "choice_table": [
            ("Regex on extracted plain text", "Rejected — inconsistent heading numbering across the 4 WHO PDFs; fragile to format variation"),
            ("LangChain PyPDFLoader", "Rejected — extracts plain text only, discarding font metadata required for heading detection"),
            ("pdfplumber", "Considered — good table extraction but no span-level font size API; less suited to heading detection"),
            ("PyMuPDF span-level API (chosen)", "Selected — exposes font size and bold flag per character span; document-format agnostic"),
            ("512-token chunk size", "Rejected — requires tokeniser during chunking; 800-char ≈ 150-200 tokens is safely within model limit"),
            ("800-character chunk size (chosen)", "Selected — fast, tokeniser-free, consistently within embedding model's 512-token input limit"),
        ],
        "contributions": [
            "Conducted a diagnostic analysis of all four WHO PDFs using PyMuPDF, printing and cataloguing every unique (font_size, is_bold) combination to empirically determine heading detection thresholds.",
            "Authored the extract_sections_from_pdf() function: PyMuPDF document opening, page-level block and span iteration, bold flag extraction via bitmask (flags & (1 << 4)), heading classification, section accumulation, and final-section flush logic.",
            "Authored the build_documents() function: multi-PDF directory iteration, two-stage chunking (intact short sections, recursive splitting for long sections), metadata propagation via {**metadata, 'chunk_index': i}, and chunk count reporting.",
            "Implemented and tuned RecursiveCharacterTextSplitter with separator priority list ['\n\n', '\n', '. ', ' '] and 150-character overlap, justifying the separator order as respecting paragraph > sentence > word boundaries.",
            "Validated chunking output by manually inspecting randomly sampled chunks to confirm that dosage tables (e.g., the oseltamivir Table 8.1) were preserved intact as single chunks rather than split mid-row.",
            "Documented the chunking strategy in Section 2.3 of the report, including the clinical justification for why naïve token-count chunking is dangerous in a medical dosage context.",
            "Debugged the PyMuPDF span iteration loop to correctly handle multi-column PDF layouts present in some WHO publications, where the default block ordering can interleave columns.",
        ],
    },

    # ── 3. Khalid Daqqaq ─────────────────────────────────────────────────────
    {
        "name": "Khalid Daqqaq",
        "id": "Student ID: 22000958",
        "role": "Embedding Pipeline Engineer & Evaluation Lead",
        "color": colors.HexColor("#4A0072"),
        "summary": (
            "Khalid was responsible for the embedding model selection, the integration "
            "of HuggingFace embeddings into the LangChain pipeline, and the design and "
            "execution of the four evaluation queries. He conducted the comparative "
            "analysis of embedding model alternatives, implemented the embedding "
            "configuration, and ran the pipeline to produce the evaluation output that "
            "forms the basis of Section 6 of the report."
        ),
        "choices": [
            {
                "q": "Why was all-MiniLM-L6-v2 chosen over BiomedBERT or PubMedBERT?",
                "a": (
                    "Khalid evaluated three embedding model families: general-purpose "
                    "sentence transformers (all-MiniLM-L6-v2, all-mpnet-base-v2), domain-"
                    "specific biomedical models (BiomedBERT-large, PubMedBERT-base), and "
                    "API-based models (text-embedding-ada-002). Biomedical models theoretically "
                    "offer better sub-word medical term disambiguation, but require GPU for "
                    "practical inference speed — BiomedBERT-large takes over 20 minutes to "
                    "embed 1,131 chunks on CPU, making iterative development impractical. "
                    "The team's queries are compositional ('what dose of drug X for condition Y?') "
                    "rather than requiring deep sub-clinical-term disambiguation, making the "
                    "performance gap smaller than it would be for, e.g., clinical NER tasks. "
                    "all-MiniLM-L6-v2 embeds all 1,131 chunks in under 90 seconds on CPU "
                    "while still capturing the semantic relationship between query vocabulary "
                    "and guideline passage vocabulary."
                ),
            },
            {
                "q": "Why was dense vector retrieval chosen over BM25 sparse retrieval?",
                "a": (
                    "Khalid tested a BM25 baseline on the five evaluation queries. BM25 "
                    "failed Query 1 ('oseltamivir dosage for confirmed influenza') because "
                    "the highest-scoring chunk matched on 'oseltamivir' alone and omitted "
                    "the dosing table context. BM25 also failed on queries phrased with "
                    "synonyms — 'steroid' did not match 'corticosteroid' and 'dexamethasone' "
                    "in the COVID document. Dense retrieval correctly handles this synonymy "
                    "through semantic embedding proximity. BM25 + dense hybrid retrieval is "
                    "noted as a Deliverable 2 enhancement that would combine BM25's exact "
                    "drug-name precision with dense retrieval's semantic recall."
                ),
            },
            {
                "q": "How were the four evaluation queries designed?",
                "a": (
                    "Khalid designed the queries to span three difficulty levels. Query 1 "
                    "(oseltamivir dosage) is a direct lookup — the answer exists verbatim "
                    "in the corpus. Query 2 (TB treatment duration) requires the system to "
                    "identify the correct disease-domain document. Query 3 (COVID-19 "
                    "corticosteroid) tests cross-document disambiguation between two COVID "
                    "PDFs and the influenza PDF which incidentally references the same drugs. "
                    "Query 4 (cough/fever/night sweats) is the intentional failure case, "
                    "designed to expose the absence of disease-domain boundary enforcement "
                    "in the baseline retrieval system."
                ),
            },
        ],
        "choice_table": [
            ("text-embedding-ada-002 (OpenAI)", "Rejected — paid API; data privacy concern; no local inference"),
            ("BiomedBERT-large", "Rejected — GPU required; >20 min CPU embedding time for 1,131 chunks"),
            ("PubMedBERT-base", "Rejected — same compute constraint; marginal gain for compositional queries"),
            ("all-mpnet-base-v2", "Rejected — 768-dim vectors; 2× storage; marginal quality gain over MiniLM at baseline"),
            ("all-MiniLM-L6-v2 (chosen)", "Selected — 384-dim; <90s CPU embedding; sentence-similarity fine-tuned; local inference"),
            ("BM25 sparse retrieval", "Rejected as primary method — fails on synonym queries; retained as future hybrid component"),
        ],
        "contributions": [
            "Conducted a comparative benchmark of five embedding models, measuring CPU embedding time for the full 1,131-chunk corpus and qualitatively evaluating retrieval quality on the test queries.",
            "Implemented the HuggingFaceEmbeddings configuration with model_kwargs={'device': 'cpu'} and encode_kwargs={'normalize_embeddings': True}, ensuring cosine similarity semantics via inner product.",
            "Verified that L2 normalization was correctly applied by sampling 10 embedding vectors and confirming their L2 norms equalled 1.0 (numpy.linalg.norm).",
            "Designed all four evaluation queries, explicitly including the cross-disease failure case (Query 4) as a stress test for the absence of domain-boundary enforcement.",
            "Executed the full pipeline run and captured all terminal output — the 1,131-chunk indexing statistics and all four query retrieval results — which form the primary evidence base for Section 6 of the report.",
            "Analysed Query 3's partial failure (influenza PDF at Rank 1 for a COVID corticosteroid query) and documented the root cause: the influenza guideline chunk's incidental mention of COVID-19 corticosteroid use pulled it into semantic proximity with the query.",
            "Wrote Section 4 (Design Justification) of the report, including the embedding model comparison table and the retrieval approach rationale.",
        ],
    },

    # ── 4. Salim Arnous ───────────────────────────────────────────────────────
    {
        "name": "Salim Arnous",
        "id": "Student ID: 22001301",
        "role": "Technical Writer & Failure Analysis Lead",
        "color": colors.HexColor("#7B3F00"),
        "summary": (
            "Salim was the primary author of the Deliverable 1 academic report. He "
            "translated the team's technical implementation decisions into structured "
            "academic prose, conducted the failure case analysis, and proposed the "
            "mitigations that will be implemented in Deliverable 2. He also authored "
            "Sections 1, 2, and 6 of the report and was responsible for ensuring the "
            "document met the course rubric requirements."
        ),
        "choices": [
            {
                "q": "How were the failure cases identified and selected?",
                "a": (
                    "Salim reviewed all four query outputs produced by Khalid and identified "
                    "two categories of failure. Failure Case 1 (cross-disease contamination) "
                    "was identified from Query 4's output, where all five retrieved chunks "
                    "came from COVID/influenza documents for a query with a classic TB "
                    "symptom triad. Failure Case 2 (wrong-document provenance) was identified "
                    "from Query 3's output, where the highest-ranked chunk came from the "
                    "influenza PDF for a COVID-19 corticosteroid query. Both cases were "
                    "selected because they represent clinically dangerous failure modes — not "
                    "merely quality degradations — where the system could actively mislead a "
                    "clinician rather than simply returning a less-than-ideal answer."
                ),
            },
            {
                "q": "Why propose a query classifier as the mitigation for Failure Case 1?",
                "a": (
                    "Salim evaluated two mitigation strategies: (1) metadata filtering by "
                    "disease domain at retrieval time, and (2) a reranker applied post-"
                    "retrieval. For cross-disease contamination, metadata filtering is more "
                    "appropriate because it prevents off-domain chunks from entering the "
                    "candidate pool entirely — a reranker applied to 5 already-wrong chunks "
                    "cannot recover the correct answer if the TB document was not retrieved "
                    "at all. A query classifier that detects TB-specific signals ('night "
                    "sweats', 'sputum', 'weight loss', 'rifampicin') and restricts the FAISS "
                    "search to the TB document subset addresses the root cause rather than "
                    "the symptom."
                ),
            },
            {
                "q": "How was the Problem Definition framed to justify RAG over a direct LLM?",
                "a": (
                    "Salim framed the justification around three concrete failure modes of "
                    "direct LLM use that are specifically dangerous in a medical setting: "
                    "hallucination of dosages, knowledge cutoff staleness for updated "
                    "guidelines, and the inability to cite provenance. Rather than making "
                    "abstract claims about RAG being 'better', the problem definition anchors "
                    "each failure mode to a specific clinical consequence — a hallucinated "
                    "oseltamivir dose, an outdated COVID management protocol, a guideline "
                    "recommendation without a citable source — making the RAG design choice "
                    "clinically motivated rather than technically motivated."
                ),
            },
        ],
        "choice_table": [
            ("Query reranker for cross-disease failure", "Deferred — reranking can't recover correct answer if TB doc not in top-k at all"),
            ("Query classifier + metadata filter (chosen)", "Selected — prevents off-domain candidates from entering the retrieval pool entirely"),
            ("Cross-encoder reranker for wrong-provenance failure (chosen)", "Selected — scores full (query, chunk) pair; demotes chunks where clinical context mismatches query"),
            ("Abstract RAG motivation in report", "Rejected — framing anchored to specific clinical consequences of each failure mode"),
            ("General LLM failure discussion", "Rejected — each failure mode tied to a concrete medical scenario from the target corpus"),
        ],
        "contributions": [
            "Authored Section 1 (Problem Definition) of the report, including the domain framing, the three-failure-mode argument against direct LLM use, and the target user table.",
            "Authored Section 2.1 (Source and Structure) with the dataset table linking each file to its WHO publication and disease domain.",
            "Authored Section 6.2 (Early Failure Cases), including the full root-cause analysis of both confirmed failures and the proposed mitigations for Deliverable 2.",
            "Identified Failure Case 1 (cross-disease contamination) by analysing Query 4's output showing all COVID results for a TB symptom query, and Failure Case 2 (wrong-document provenance) from Query 3's Rank-1 influenza chunk for a COVID corticosteroid query.",
            "Proposed the disease-domain query classifier with FAISS metadata filtering as the Deliverable 2 mitigation for Failure Case 1, and the cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2) as the mitigation for Failure Case 2.",
            "Conducted a literature review covering Lewis et al. (2020) RAG paper, Reimers & Gurevych (2019) Sentence-BERT paper, and Johnson et al. (2021) FAISS paper, and compiled the References section.",
            "Produced the final formatted Deliverable_1.md and coordinated the DOCX export with Yousef to ensure the document structure matched the rubric exactly.",
            "Reviewed the AI Usage Disclosure table to ensure it accurately reflected team versus AI contributions per the course's academic integrity policy.",
        ],
    },

    # ── 5. Yousef Osama ───────────────────────────────────────────────────────
    {
        "name": "Yousef Osama",
        "id": "Student ID: 22000959",
        "role": "QA Engineer, Visualization & Documentation Lead",
        "color": colors.HexColor("#003366"),
        "summary": (
            "Yousef was responsible for quality assurance of the full pipeline, creation "
            "of the system architecture diagram, and the final formatting and export of "
            "all deliverable documents. He performed end-to-end testing of the pipeline "
            "across all four PDFs, verified that chunk counts and retrieval outputs were "
            "correct, and produced the visual artefacts (architecture diagram, DOCX) that "
            "accompany the written report."
        ),
        "choices": [
            {
                "q": "What approach was used to validate that chunking preserved dosage tables intact?",
                "a": (
                    "Yousef wrote a validation script that iterated over all 1,131 produced "
                    "chunks and flagged any chunk containing the strings 'mg', 'dose', or "
                    "'daily' as a 'dosage chunk'. He then manually inspected the 40 shortest "
                    "flagged chunks (under 200 characters) to check whether any appeared to "
                    "be truncated mid-table. Two such chunks were found in an early version "
                    "of the pipeline, traced to a section boundary that fell mid-table in the "
                    "influenza PDF. Faisal used this finding to adjust the heading detection "
                    "threshold for that specific table's context, resolving the truncation."
                ),
            },
            {
                "q": "Why was draw.io used for the architecture diagram rather than an automated tool?",
                "a": (
                    "Automated diagram tools (e.g., matplotlib with networkx, or mermaid.js) "
                    "produce layouts that are correct but visually minimal — boxes and arrows "
                    "with no visual hierarchy. The architecture diagram needed to communicate "
                    "not just the pipeline sequence but the distinction between the offline "
                    "indexing phase (Stages 1–4) and the online query phase (Stage 5), and "
                    "the role of the FAISS index as a persistent artefact shared between "
                    "both phases. draw.io allowed Yousef to visually separate these phases "
                    "with a vertical dividing line and colour-code the offline (blue) versus "
                    "online (teal) stages — a distinction that is invisible in a simple "
                    "left-to-right flowchart."
                ),
            },
            {
                "q": "How was the DOCX export script validated for structural correctness?",
                "a": (
                    "After generating the DOCX, Yousef ran a validation script using "
                    "python-docx that printed every heading (Heading 1–4 styles) and "
                    "the count of tables and paragraphs. He verified that all 8 section "
                    "headings were present, that all 7 tables (user persona, dataset, "
                    "chunking justification, component summary, alternatives considered, "
                    "AI disclosure, and team signatures) rendered correctly, and that "
                    "code blocks appeared in Courier New monospace with grey background shading. "
                    "He also opened the document in Microsoft Word to perform a visual "
                    "inspection of heading colours, table header colours, and page margins."
                ),
            },
        ],
        "choice_table": [
            ("Automated diagram (matplotlib/networkx)", "Rejected — no visual hierarchy; cannot distinguish offline indexing from online query phase"),
            ("mermaid.js flowchart", "Rejected — limited control over visual design; no colour-coding by pipeline phase"),
            ("draw.io (chosen)", "Selected — full visual control; offline/online phase separation; colour-coded components"),
            ("Printing to confirm chunk integrity", "Insufficient — too slow for 1,131 chunks; replaced by targeted flagging script"),
            ("Targeted dosage-chunk validation script (chosen)", "Selected — efficiently surfaces at-risk chunks for manual inspection without reviewing all 1,131"),
        ],
        "contributions": [
            "Performed end-to-end pipeline testing across three runs: fresh index build (verifying embedding and FAISS construction), index reload (verifying disk persistence), and full query evaluation (verifying all four queries produce ranked output).",
            "Wrote a chunk validation script that flagged dosage-sensitive chunks (containing 'mg', 'dose', 'daily') for manual inspection, identifying two truncated dosage table chunks in an early pipeline version.",
            "Created the system architecture diagram in draw.io, distinguishing the offline indexing phase (Stages 1–4, blue) from the online query phase (Stage 5, teal), and adding the 'Query' input and 'Answer + Citations' output annotation arrows.",
            "Authored and ran md_to_docx.py — the Markdown-to-DOCX conversion script using python-docx — implementing heading styles, coloured table headers, code block formatting (Courier New, grey background), and inline bold/italic/code parsing.",
            "Validated the DOCX output by inspecting heading hierarchy, table count, and code block formatting using a python-docx verification script, and performed visual inspection in Microsoft Word.",
            "Confirmed the FAISS index serialization to ./faiss_index/ by verifying the presence of index.faiss and index.pkl files and timing the reload versus rebuild to confirm the performance benefit.",
            "Maintained the project file structure, ensuring all artefacts (baseline_rag.py, Deliverable_1.md, Deliverable_1.docx, faiss_index/) were organized and named according to the submission requirements.",
            "Coordinated the final review pass of the report with Salim, cross-checking all statistics quoted in the text (1,131 chunks, 402 sections for influenza PDF, similarity scores) against the actual pipeline output.",
        ],
    },
]


# ── Document builder ──────────────────────────────────────────────────────────

def build_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.0*cm,  bottomMargin=1.8*cm,
        title="Team Contributions Report — Deliverable 1",
        author="CSAI-413 NLP Applications Team",
    )

    styles  = make_styles()
    story   = []

    # ── Cover page ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.6*cm))

    # Project badge
    badge_data = [[ Paragraph(
        '<font color="white"><b>CSAI-413 Natural Language Processing Applications</b></font>',
        ParagraphStyle("badge", fontName="Helvetica-Bold", fontSize=11,
                       textColor=WHITE, alignment=TA_CENTER)) ]]
    badge = Table(badge_data, colWidths=["100%"])
    badge.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), STEEL),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 20),
        ("RIGHTPADDING",  (0,0),(-1,-1), 20),
    ]))
    story.append(badge)
    story.append(Spacer(1, 0.8*cm))

    story.append(Paragraph("Team Contributions Report", styles["cover_title"]))
    story.append(Paragraph("Deliverable 1 — Baseline RAG System", styles["cover_sub"]))
    story.append(Paragraph("Respiratory Infection Knowledge Assistant", styles["cover_sub"]))
    story.append(Spacer(1, 0.5*cm))
    story.append(hr(NAVY, 1.2))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("May 2026", styles["cover_meta"]))
    story.append(Paragraph(
        "This document provides a detailed breakdown of each team member's assigned role, "
        "design decisions made during Deliverable 1, and exact individual contributions "
        "to the baseline RAG system implementation and report.",
        ParagraphStyle("cover_body", fontName="Helvetica", fontSize=10,
                       textColor=MID_GREY, alignment=TA_CENTER, leading=15)))

    story.append(Spacer(1, 1.0*cm))

    # Team roster table on cover
    roster_data = [
        [Paragraph("<b>Name</b>", ParagraphStyle("rh", fontName="Helvetica-Bold",
                    fontSize=10, textColor=WHITE, alignment=TA_CENTER)),
         Paragraph("<b>Student ID</b>", ParagraphStyle("rh2", fontName="Helvetica-Bold",
                    fontSize=10, textColor=WHITE, alignment=TA_CENTER)),
         Paragraph("<b>Role</b>", ParagraphStyle("rh3", fontName="Helvetica-Bold",
                    fontSize=10, textColor=WHITE, alignment=TA_CENTER))],
        ["Baraa Kabbani",  "22001553", "Project Lead & System Architect"],
        ["Faisal Sahloul", "22000669", "PDF Preprocessing & Chunking Engineer"],
        ["Khalid Daqqaq",  "22000958", "Embedding Pipeline Engineer & Evaluation Lead"],
        ["Salim Arnous",   "22001301", "Technical Writer & Failure Analysis Lead"],
        ["Yousef Osama",   "22000959", "QA Engineer, Visualization & Documentation Lead"],
    ]
    roster = Table(roster_data, colWidths=["30%", "20%", "50%"])
    roster.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), NAVY),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [LIGHT_BLUE, WHITE]),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#CBD5E1")),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,0), (-1,-1), 10),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("ALIGN",         (1,0), (1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(roster)
    story.append(PageBreak())

    # ── Per-member sections ───────────────────────────────────────────────────
    member_colors = [
        colors.HexColor("#1F497D"),
        colors.HexColor("#155724"),
        colors.HexColor("#4A0072"),
        colors.HexColor("#7B3F00"),
        colors.HexColor("#003366"),
    ]

    for idx, m in enumerate(MEMBERS):
        color = member_colors[idx]

        # Member header banner
        story.append(KeepTogether([
            member_banner(m["name"], m["id"], m["role"], color, styles),
            Spacer(1, 0.15*cm),
            Paragraph(m["role"], ParagraphStyle(
                "role_line", fontName="Helvetica-BoldOblique", fontSize=11,
                textColor=color, spaceAfter=8)),
            hr(color, 0.8),
        ]))

        # Summary
        story.append(section_header("Overview", styles))
        story.append(Paragraph(m["summary"], styles["body"]))
        story.append(Spacer(1, 0.3*cm))

        # ── Design Choices ────────────────────────────────────────────────────
        story.append(section_header("Deliverable 1 — Design Choices & Reasoning", styles))
        story.append(hr(colors.HexColor("#CBD5E1"), 0.4))

        for choice in m["choices"]:
            story.append(Paragraph(choice["q"], styles["decision_q"]))
            story.append(Paragraph(choice["a"], styles["decision_a"]))
            story.append(Spacer(1, 0.1*cm))

        story.append(Spacer(1, 0.25*cm))

        # Alternatives table
        story.append(section_header("Options Evaluated — Comparison", styles))
        story.append(choice_table(m["choice_table"]))
        story.append(Spacer(1, 0.3*cm))

        # ── Exact Contributions ───────────────────────────────────────────────
        story.append(section_header("Exact Contributions — Deliverable 1", styles))
        story.append(hr(colors.HexColor("#CBD5E1"), 0.4))

        for item in m["contributions"]:
            story.append(bullet(item, styles))

        story.append(PageBreak())

    # remove last page break
    if story and isinstance(story[-1], PageBreak):
        story.pop()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    build_pdf(r"C:\Users\baraa\Desktop\anlp project\Team_Contributions.pdf")
