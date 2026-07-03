"""
app.py  --  Streamlit demo for the Respiratory Infection Knowledge Assistant
Run:  streamlit run app.py
"""

import os, json, time
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="WHO Respiratory Knowledge Assistant",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
[data-testid="stAppViewContainer"] { background: #F8FAFC; }
[data-testid="stSidebar"] { background: #1F497D; }
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox label { color: white !important; }

/* Headings */
h1 { color: #1F497D; font-weight: 800; }
h2 { color: #2E74B5; }
h3 { color: #374151; }

/* Answer box */
.answer-box {
    background: linear-gradient(135deg, #EEF4FB 0%, #F0FDF4 100%);
    border-left: 5px solid #1F497D;
    border-radius: 8px;
    padding: 20px 24px;
    margin: 12px 0;
    font-size: 15px;
    line-height: 1.7;
    color: #1A1A2E;
}

/* Source card */
.source-card {
    background: white;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 13px;
}
.source-rank {
    background: #1F497D;
    color: white;
    border-radius: 50%;
    width: 24px; height: 24px;
    display: inline-flex;
    align-items: center; justify-content: center;
    font-weight: bold; font-size: 12px;
    margin-right: 8px;
}
.score-bar-fill {
    background: linear-gradient(90deg, #1F497D, #2E74B5);
    height: 8px; border-radius: 4px;
}

/* Metric card */
.metric-card {
    background: white;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    border-top: 4px solid #1F497D;
}
.metric-value { font-size: 32px; font-weight: 800; color: #1F497D; }
.metric-label { font-size: 13px; color: #6B7280; margin-top: 4px; }
.metric-delta-pos { color: #059669; font-size: 14px; font-weight: 600; }
.metric-delta-neg { color: #DC2626; font-size: 14px; font-weight: 600; }

/* Pipeline stage box */
.stage-box {
    background: white;
    border-radius: 10px;
    border: 2px solid #E2E8F0;
    padding: 16px;
    text-align: center;
    height: 130px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
.stage-icon { font-size: 28px; margin-bottom: 6px; }
.stage-title { font-weight: 700; color: #1F497D; font-size: 14px; }
.stage-sub { font-size: 11px; color: #6B7280; margin-top: 4px; }

/* Failure badge */
.badge-fixed { background:#DCFCE7; color:#166534; padding:3px 10px;
               border-radius:12px; font-size:12px; font-weight:600; }
.badge-persistent { background:#FEE2E2; color:#991B1B; padding:3px 10px;
                    border-radius:12px; font-size:12px; font-weight:600; }
.badge-partial { background:#FEF3C7; color:#92400E; padding:3px 10px;
                 border-radius:12px; font-size:12px; font-weight:600; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: white; border-radius: 8px 8px 0 0;
    padding: 8px 20px; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫁 WHO Assistant")
    st.markdown("**Course:** CSAI-413 NLP Applications")
    st.markdown("**Deliverable 2** — Experimental RAG")
    st.divider()
    st.markdown("**Knowledge Base**")
    st.markdown("- Influenza Clinical Guidelines")
    st.markdown("- COVID-19 SARI Management")
    st.markdown("- COVID-19 Policy Brief")
    st.markdown("- TB Evidence Gaps")
    st.divider()
    st.markdown("**Team**")
    st.markdown("Baraa Kabbani · 22001553")
    st.markdown("Faisal Sahloul · 22000669")
    st.markdown("Khalid Daqqaq · 22000958")
    st.markdown("Salim Arnous · 22001301")
    st.markdown("Yousef Osama · 22000959")
    st.divider()
    groq_key = st.text_input("Groq API Key", type="password",
                              value=os.environ.get("GROQ_API_KEY", ""),
                              help="Get a free key at console.groq.com")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key


# ── Load RAG systems (cached so they only load once) ─────────────────────────
@st.cache_resource(show_spinner="Loading improved RAG system...")
def load_improved_rag():
    from improved_rag import ImprovedRAG
    return ImprovedRAG()


@st.cache_resource(show_spinner="Loading baseline index...")
def load_baseline_vs():
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local("./faiss_index", emb,
                            allow_dangerous_deserialization=True)


def get_domain_color(source: str) -> str:
    if "9789240097759" in source: return "#1F497D"
    if "tuberculosis" in source:  return "#7B3F00"
    return "#155724"


def get_domain_label(source: str) -> str:
    if "9789240097759" in source: return "Influenza"
    if "tuberculosis" in source:  return "Tuberculosis"
    if "policy-brief" in source:  return "COVID-19 Policy"
    return "COVID-19"


def short_source(source: str) -> str:
    if "9789240097759" in source: return "Influenza Guidelines"
    if "tuberculosis" in source:  return "TB Evidence Gaps"
    if "policy-brief" in source:  return "COVID-19 Policy Brief"
    if "clinical-management" in source: return "COVID-19 SARI Mgmt"
    return source[:30]


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "💬  Chat Assistant",
    "📊  Experiment Results",
    "⚖️  Baseline vs Improved",
    "🏗️  System Architecture",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT INTERFACE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("# 💬 Clinical Knowledge Assistant")
    st.markdown("Ask any question about WHO guidelines for **Influenza**, **COVID-19**, or **Tuberculosis**.")

    # Example queries
    st.markdown("**Try one of these:**")
    example_cols = st.columns(3)
    examples = [
        "What is the oseltamivir dose for an adult with influenza?",
        "What corticosteroid is recommended for severe COVID-19?",
        "What is the standard TB treatment duration?",
        "At what SpO2 should oxygen therapy start in COVID-19?",
        "What antivirals are recommended for non-severe COVID-19?",
        "How should a patient with chronic cough and night sweats be managed?",
    ]
    if "chat_query" not in st.session_state:
        st.session_state.chat_query = ""

    for i, ex in enumerate(examples):
        col = example_cols[i % 3]
        with col:
            if st.button(ex[:55] + ("..." if len(ex) > 55 else ""),
                         key=f"ex_{i}", use_container_width=True):
                st.session_state.chat_query = ex

    st.divider()

    # Query input
    query = st.text_input(
        "Your clinical question:",
        value=st.session_state.chat_query,
        placeholder="e.g. What is the dexamethasone dose for severe COVID-19?",
        key="query_input",
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        ask_btn = st.button("Ask", type="primary", use_container_width=True)

    if ask_btn and query.strip():
        rag = load_improved_rag()

        with st.spinner("Retrieving from WHO guidelines..."):
            t0 = time.time()
            results = rag.retrieve(query)
            retrieval_time = time.time() - t0

        answer = ""
        if os.environ.get("GROQ_API_KEY"):
            with st.spinner("Generating answer with LLaMA-3.1..."):
                context = rag._format_context(results)
                answer = rag.generate(query, context)
                total_time = time.time() - t0
        else:
            answer = "**Set your Groq API key in the sidebar to enable answer generation.**\nRetrieval results are shown below."
            total_time = retrieval_time

        # Answer display
        st.markdown("### Answer")
        st.markdown(f'<div class="answer-box">{answer}</div>',
                    unsafe_allow_html=True)

        st.caption(f"Total time: {total_time:.1f}s  |  "
                   f"Retrieval: {retrieval_time:.1f}s  |  "
                   f"Model: llama-3.1-8b-instant  |  "
                   f"Pipeline: Hybrid BM25+Dense -> Cross-Encoder -> LLaMA-3.1")

        # Sources
        st.markdown("### Retrieved Sources")
        max_score = max(score for _, score in results) if results else 1

        for rank, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            domain_color = get_domain_color(meta.get("source", ""))
            bar_pct = int((score / max(max_score, 0.01)) * 100)

            with st.container():
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="source-rank">{rank}</span>'
                        f'<strong>{short_source(meta.get("source","?"))}</strong>'
                        f' &nbsp;·&nbsp; Page {meta.get("page","?")} '
                        f'&nbsp;·&nbsp; <em>{meta.get("section_header","?")[:60]}</em><br>'
                        f'<div style="margin-top:8px;background:#E2E8F0;border-radius:4px;height:8px;">'
                        f'<div style="width:{bar_pct}%;background:{domain_color};'
                        f'height:8px;border-radius:4px;"></div></div>'
                        f'<small style="color:#6B7280;">CE Score: {score:.2f}</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    with st.expander("View chunk"):
                        st.write(doc.page_content[:500])

    elif ask_btn and not query.strip():
        st.warning("Please enter a question.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: EXPERIMENT RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("# 📊 Experiment Results")
    st.markdown("Four controlled experiments measured the impact of each system improvement on 10 clinical test queries.")

    results_path = Path("./experiment_results.json")
    if not results_path.exists():
        st.warning("Run `python experiments.py` first to generate results.")
        st.stop()

    with open(results_path) as f:
        exp_data = json.load(f)

    # ── Experiment 1: Chunk Size ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Experiment 1 — Chunk Size")
    st.markdown("""
    **Hypothesis:** Smaller chunks (400 chars) improve Recall@1 for dosage-lookup queries.
    Larger chunks may reduce Grounding by diluting embedding signal.
    """)

    chunk_configs = exp_data["exp1"]["configs"]
    chunk_labels  = [c["label"] for c in chunk_configs]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for metric, color, name in [
            ("avg_recall@1", "#1F497D", "Recall@1"),
            ("avg_recall@3", "#2E74B5", "Recall@3"),
            ("avg_domain@3", "#00838F", "Domain@3"),
        ]:
            fig.add_trace(go.Bar(
                name=name,
                x=chunk_labels,
                y=[c[metric] for c in chunk_configs],
                marker_color=color,
            ))
        fig.update_layout(barmode="group", title="Retrieval Metrics by Chunk Size",
                          yaxis=dict(range=[0, 1.1], title="Score"),
                          legend=dict(orientation="h", y=-0.2),
                          plot_bgcolor="white", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure(go.Bar(
            x=chunk_labels,
            y=[c["avg_grounding"] for c in chunk_configs],
            marker_color=["#F59E0B", "#1F497D", "#059669"],
            text=[f"{c['avg_grounding']:.2f}" for c in chunk_configs],
            textposition="outside",
        ))
        fig2.update_layout(title="Avg Grounding Score by Chunk Size",
                           yaxis=dict(range=[0, 0.9], title="Grounding"),
                           plot_bgcolor="white", height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.info("**Finding:** 800-char chunks are optimal. 400-char chunks lower Grounding (0.46 vs 0.61) because dosage tables span multiple small chunks. 1200-char chunks lower Recall@1 (0.50 vs 0.60) by diluting the embedding signal.")

    # ── Experiment 2: Reranking ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Experiment 2 — Cross-Encoder Reranking")
    st.markdown("""
    **Hypothesis:** Jointly scoring (query, chunk) pairs with a cross-encoder improves Recall@1
    over cosine-similarity-only FAISS ranking.
    """)

    e2 = exp_data["exp2"]
    base_r2 = e2["baseline"]
    impr_r2 = e2["improved"]

    metrics_e2 = ["avg_recall@1", "avg_recall@3", "avg_recall@5", "avg_domain@3", "avg_grounding"]
    labels_e2  = ["Recall@1", "Recall@3", "Recall@5", "Domain@3", "Grounding"]

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="No Reranker (Baseline)",
                          x=labels_e2,
                          y=[base_r2[m] for m in metrics_e2],
                          marker_color="#94A3B8"))
    fig3.add_trace(go.Bar(name="Cross-Encoder Reranked",
                          x=labels_e2,
                          y=[impr_r2[m] for m in metrics_e2],
                          marker_color="#1F497D"))
    fig3.update_layout(barmode="group", title="Baseline vs Cross-Encoder Reranked",
                       yaxis=dict(range=[0, 1.1], title="Score"),
                       plot_bgcolor="white", height=380,
                       legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig3, use_container_width=True)

    # Per-query change table
    pq = exp_data["exp2"]["per_query"]
    rows = []
    for b, r in zip(pq["baseline"], pq["reranked"]):
        changed = b["recall@1"] != r["recall@1"]
        rows.append({
            "Query": b["query"],
            "Baseline R@1": "✓" if b["recall@1"] else "✗",
            "Reranked R@1": "✓" if r["recall@1"] else "✗",
            "Change": "FIXED" if (not b["recall@1"] and r["recall@1"]) else
                      ("BROKEN" if (b["recall@1"] and not r["recall@1"]) else "—"),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    r1_delta = impr_r2["avg_recall@1"] - base_r2["avg_recall@1"]
    st.success(f"**Finding:** Cross-encoder reranking improved Recall@1 by **+{r1_delta:.0%}** "
               f"({base_r2['avg_recall@1']:.2f} -> {impr_r2['avg_recall@1']:.2f}). "
               f"Fixed Q08 (non-severe COVID antivirals) and Q10 (IPC measures).")

    # ── Experiment 3: Hybrid Retrieval ────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Experiment 3 — Hybrid BM25 + Dense Retrieval (RRF)")
    st.markdown("""
    **Hypothesis:** Combining BM25 lexical matching with dense semantic retrieval via
    Reciprocal Rank Fusion improves Recall@1, especially for drug-name queries.
    """)

    e3 = exp_data["exp3"]
    systems = {"Dense-Only (Baseline)": e3["dense"],
               "BM25-Only": e3["bm25"],
               "Hybrid RRF": e3["hybrid"]}
    sys_colors = {"Dense-Only (Baseline)": "#94A3B8",
                  "BM25-Only": "#F59E0B",
                  "Hybrid RRF": "#1F497D"}

    fig4 = go.Figure()
    for sys_name, sys_data in systems.items():
        fig4.add_trace(go.Scatterpolar(
            r=[sys_data["avg_recall@1"], sys_data["avg_recall@3"],
               sys_data["avg_recall@5"], sys_data["avg_domain@3"],
               sys_data["avg_grounding"], sys_data["avg_recall@1"]],
            theta=["Recall@1", "Recall@3", "Recall@5", "Domain@3", "Grounding", "Recall@1"],
            fill="toself", name=sys_name,
            line_color=sys_colors[sys_name],
        ))
    fig4.update_layout(polar=dict(radialaxis=dict(range=[0, 1])),
                       title="Retrieval Strategy Comparison (Radar)",
                       height=420, legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig4, use_container_width=True)

    # Q03 spotlight
    st.markdown("**Critical fix — Q03 (COVID corticosteroid query):**")
    q03_cols = st.columns(3)
    for col, (sys_name, sys_data) in zip(q03_cols, systems.items()):
        q03_row = next(r for r in exp_data["exp3"]["per_query"][
            list(exp_data["exp3"]["per_query"].keys())[
                ["Dense-Only (Baseline)", "BM25-Only", "Hybrid RRF"].index(sys_name)
            ]
        ] if r["id"] == "Q03")
        status = "Fixed" if q03_row["recall@1"] else "Wrong source at Rank 1"
        color = "#059669" if q03_row["recall@1"] else "#DC2626"
        with col:
            st.markdown(f"**{sys_name}**")
            st.markdown(f'<span style="color:{color};font-weight:700">{status}</span>',
                        unsafe_allow_html=True)

    hybrid_delta = e3["hybrid"]["avg_recall@1"] - e3["dense"]["avg_recall@1"]
    st.success(f"**Finding:** Hybrid RRF improved Recall@1 by **+{hybrid_delta:.0%}** "
               f"and fixed the D1 failure case (influenza PDF returned for COVID query).")

    # ── Experiment 4: Prompt Design ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Experiment 4 — Prompt Design")

    e4 = exp_data.get("exp4", {})
    if e4.get("skipped") or not e4.get("results"):
        st.warning("Experiment 4 was skipped (GROQ_API_KEY not set when experiments.py ran). "
                   "Set your key in the sidebar and re-run: `python experiments.py`")
    else:
        prompt_results = e4["results"]
        prompt_names   = list(prompt_results.keys())

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            name="Grounding Score",
            x=prompt_names,
            y=[v["avg_grounding"] for v in prompt_results.values()],
            marker_color="#1F497D",
        ))
        fig5.add_trace(go.Bar(
            name="Citation Rate",
            x=prompt_names,
            y=[v["avg_cites_source"] for v in prompt_results.values()],
            marker_color="#00838F",
        ))
        fig5.update_layout(barmode="group", title="Prompt Design: Grounding vs Citation Rate",
                           yaxis=dict(range=[0, 1.1]),
                           plot_bgcolor="white", height=360)
        st.plotly_chart(fig5, use_container_width=True)

        best = max(prompt_results.items(), key=lambda x: x[1]["avg_cites_source"])
        st.success(f"**Finding:** '{best[0]}' achieved the highest citation rate "
                   f"({best[1]['avg_cites_source']:.0%}), confirming that structured "
                   f"prompts are essential for clinical answer traceability.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: BASELINE vs IMPROVED
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("# ⚖️ Baseline vs Improved System")
    st.markdown("Run the same query through both systems side-by-side to see the difference.")

    compare_query = st.text_input(
        "Query to compare:",
        value="What corticosteroid is recommended for severe COVID-19?",
        key="compare_input",
    )

    st.markdown("**Quick examples:**")
    qc1, qc2, qc3 = st.columns(3)
    compare_examples = [
        "What corticosteroid is recommended for severe COVID-19?",
        "What antiviral is recommended for non-severe COVID-19?",
        "What is the oseltamivir dose for an adult with influenza?",
    ]
    for i, (col, ex) in enumerate(zip([qc1, qc2, qc3], compare_examples)):
        with col:
            if st.button(ex[:40] + "...", key=f"cex_{i}", use_container_width=True):
                st.session_state["compare_input"] = ex
                st.rerun()

    if st.button("Run Comparison", type="primary"):
        rag = load_improved_rag()
        baseline_vs = load_baseline_vs()

        with st.spinner("Running both systems..."):
            # Baseline: dense-only FAISS top-5
            base_results = baseline_vs.similarity_search_with_score(compare_query, k=5)
            # Improved: hybrid + cross-encoder
            impr_results = rag.retrieve(compare_query)

        col_base, col_impr = st.columns(2)

        def render_results(results, label, col):
            with col:
                st.markdown(f"### {label}")
                for rank, (doc, score) in enumerate(results, 1):
                    meta = doc.metadata
                    src  = short_source(meta.get("source", "?"))
                    dc   = get_domain_color(meta.get("source", ""))
                    dl   = get_domain_label(meta.get("source", ""))
                    st.markdown(
                        f'<div class="source-card">'
                        f'<b style="color:{dc}">#{rank} {src}</b>'
                        f'<span style="float:right;background:{dc};color:white;'
                        f'padding:2px 8px;border-radius:10px;font-size:11px">{dl}</span><br>'
                        f'Page {meta.get("page","?")} &middot; '
                        f'<em>{meta.get("section_header","?")[:50]}</em><br>'
                        f'<small style="color:#6B7280;">Score: {score:.4f}</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        render_results(base_results, "Baseline (Dense-Only FAISS)", col_base)
        render_results(impr_results, "Improved (Hybrid + Cross-Encoder)", col_impr)

        # Comparison insight
        st.divider()
        base_sources = [d.metadata.get("source", "") for d, _ in base_results[:3]]
        impr_sources = [d.metadata.get("source", "") for d, _ in impr_results[:3]]

        base_domains = [get_domain_label(s) for s in base_sources]
        impr_domains = [get_domain_label(s) for s in impr_sources]

        st.markdown("**Comparison Summary:**")
        ci1, ci2 = st.columns(2)
        with ci1:
            st.markdown(f"**Baseline Top-3 Domains:** {', '.join(base_domains)}")
        with ci2:
            st.markdown(f"**Improved Top-3 Domains:** {', '.join(impr_domains)}")

        # Venn-like overlap
        base_set = set(d.page_content[:80] for d, _ in base_results[:5])
        impr_set = set(d.page_content[:80] for d, _ in impr_results[:5])
        overlap  = len(base_set & impr_set)
        st.info(f"Chunk overlap between systems: **{overlap}/5** chunks shared in top-5. "
                f"The improved system promotes {5-overlap} different chunks.")

    # Summary metrics table
    st.divider()
    st.markdown("### Full Test Set: Quantitative Comparison")
    summary_data = {
        "Metric": ["Recall@1", "Recall@3", "Recall@5", "Domain@3", "Grounding"],
        "Baseline": [0.60, 0.90, 0.90, 0.90, 0.61],
        "Improved": [0.80, 0.80, 0.90, 0.90, 0.71],
        "Delta": ["+0.20", "-0.10", "0.00", "0.00", "+0.10"],
        "% Change": ["+33%", "-11%", "0%", "0%", "+16%"],
    }
    df_sum = pd.DataFrame(summary_data)
    st.dataframe(df_sum, use_container_width=True, hide_index=True)

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(name="Baseline", x=summary_data["Metric"],
                                  y=summary_data["Baseline"], marker_color="#94A3B8"))
    fig_compare.add_trace(go.Bar(name="Improved", x=summary_data["Metric"],
                                  y=summary_data["Improved"], marker_color="#1F497D"))
    fig_compare.update_layout(barmode="group", yaxis=dict(range=[0, 1.1]),
                               plot_bgcolor="white", height=360,
                               title="Baseline vs Improved — All Metrics",
                               legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_compare, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: SYSTEM ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("# 🏗️ System Architecture")

    arch_tab1, arch_tab2 = st.tabs(["Improved Pipeline", "Baseline vs Improved"])

    with arch_tab1:
        st.markdown("### Improved RAG Pipeline")
        st.markdown("The improved system adds **two new stages** after FAISS retrieval: hybrid fusion and cross-encoder reranking.")

        stages = [
            ("📄", "PDF Preprocessing", "PyMuPDF\nFont metadata\nHeader detection", "#1F497D"),
            ("✂️", "Semantic Chunking", "800-char chunks\nRecursive splitter\n150-char overlap", "#2E74B5"),
            ("🔢", "Embedding", "all-MiniLM-L6-v2\n384-dim vectors\nL2 normalized", "#1F497D"),
            ("🗄️", "FAISS Index", "IndexFlatIP\nCosine similarity\nPersisted to disk", "#2E74B5"),
            ("🔀", "Hybrid Retrieval", "BM25 + Dense\nRRF fusion\nTop-10 candidates", "#00838F"),
            ("🎯", "Cross-Encoder\nReranking", "ms-marco-MiniLM\nJoint scoring\nTop-5 selected", "#059669"),
            ("💬", "Clinical Prompt\n+ Generation", "Structured prompt\nGroq LLaMA-3.1\nCited answer", "#7B3F00"),
        ]

        cols = st.columns(len(stages))
        for col, (icon, title, sub, color) in zip(cols, stages):
            with col:
                st.markdown(
                    f'<div style="background:white;border-radius:10px;'
                    f'border-top:4px solid {color};padding:16px;text-align:center;'
                    f'min-height:130px;box-shadow:0 1px 4px rgba(0,0,0,0.08);">'
                    f'<div style="font-size:24px">{icon}</div>'
                    f'<div style="font-weight:700;color:{color};font-size:13px;'
                    f'margin:6px 0">{title}</div>'
                    f'<div style="font-size:11px;color:#6B7280;white-space:pre-line">{sub}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Arrows between stages
        arrow_cols = st.columns(len(stages))
        for i, col in enumerate(arrow_cols):
            if i < len(stages) - 1:
                with col:
                    st.markdown('<div style="text-align:right;font-size:20px;'
                                'color:#94A3B8;margin-top:-8px">--></div>',
                                unsafe_allow_html=True)

        st.divider()
        st.markdown("### Component Details")

        comp_data = {
            "Component": ["PDF Loader", "Chunker", "Embedding Model", "Vector Store",
                          "BM25 Retriever", "Cross-Encoder", "LLM Generator"],
            "Technology": ["PyMuPDF (fitz)", "LangChain RecursiveCharacterTextSplitter",
                           "sentence-transformers/all-MiniLM-L6-v2",
                           "FAISS IndexFlatIP", "rank_bm25 BM25Okapi",
                           "cross-encoder/ms-marco-MiniLM-L-6-v2",
                           "Groq API / llama-3.1-8b-instant"],
            "Key Config": ["Font size >= 14pt heading detection", "800 chars, 150 overlap",
                           "384-dim, L2 normalized", "Exact cosine, persisted to disk",
                           "Token-level BM25 scoring", "Joint (query, chunk) scoring",
                           "Structured clinical prompt, temp=0.1"],
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    with arch_tab2:
        st.markdown("### Baseline vs Improved: What Changed")

        changes = [
            ("Retrieval", "Dense-only FAISS cosine similarity",
             "Hybrid BM25 + Dense RRF (top-10 candidates)",
             "Fixed Q03 wrong-document failure; Recall@1 +17%"),
            ("Ranking", "FAISS similarity order (no reranking)",
             "Cross-encoder reranks top-10 -> top-5",
             "Recall@1 0.60 -> 0.80 (+33%); Fixed Q08, Q10"),
            ("Generation", "Not implemented",
             "Groq LLaMA-3.1 8B with clinical structured prompt",
             "Citation rate: 20% -> 90%"),
            ("Evaluation", "4 ad-hoc queries, qualitative observation",
             "10-query test set, 6 quantitative metrics",
             "Reproducible, comparable measurement"),
            ("Chunk size", "800 chars (empirical)",
             "800 chars (confirmed optimal by experiment)",
             "Exp 1 confirmed; 400-char drops grounding to 0.46"),
        ]

        for comp, before, after, impact in changes:
            with st.expander(f"**{comp}** — {before[:40]}... -> {after[:40]}..."):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Before (Baseline)**")
                    st.markdown(f'<div style="background:#FEE2E2;padding:12px;'
                                f'border-radius:8px;font-size:13px">{before}</div>',
                                unsafe_allow_html=True)
                with c2:
                    st.markdown("**After (Improved)**")
                    st.markdown(f'<div style="background:#DCFCE7;padding:12px;'
                                f'border-radius:8px;font-size:13px">{after}</div>',
                                unsafe_allow_html=True)
                st.markdown(f"**Impact:** {impact}")

        st.divider()
        st.markdown("### Known Limitations")

        failures = [
            ("Q09 — Cross-Disease Contamination", "persistent",
             "Chronic cough + fever + night sweats (TB) retrieves COVID documents across all configurations.",
             "Pre-retrieval disease domain classifier (Deliverable 3)"),
            ("Q03 — Incomplete Grounding", "partial",
             "Dexamethasone 6mg dosage not in retrieved context despite correct document at Rank 1.",
             "Table-aware extraction + multi-chunk synthesis"),
            ("Q08, Q10 — Wrong-document Provenance", "fixed",
             "Baseline retrieved wrong specific document at Rank 1.",
             "Fixed by cross-encoder reranking"),
        ]

        for title, status, desc, fix in failures:
            badge_class = {"fixed": "badge-fixed", "partial": "badge-partial",
                           "persistent": "badge-persistent"}[status]
            st.markdown(
                f'<div style="background:white;border-radius:8px;padding:16px;'
                f'margin:8px 0;border:1px solid #E2E8F0;">'
                f'<b>{title}</b> '
                f'<span class="{badge_class}">{status.upper()}</span><br>'
                f'<span style="font-size:13px;color:#374151">{desc}</span><br>'
                f'<span style="font-size:12px;color:#6B7280">Fix: {fix}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
