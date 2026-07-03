"""
evaluation_framework.py
-----------------------
Shared evaluation test set and metric functions for Deliverable 2.
All experiments import from here to ensure consistent measurement.

Metrics implemented:
  - source_recall_at_k   : is the correct source file in top-k results?
  - domain_accuracy_at_k : is any chunk from the correct disease domain in top-k?
  - mean_similarity      : average similarity score of top-k results
  - hallucination_score  : fraction of expected key terms present in retrieved context
"""

from __future__ import annotations

# ?? Ground-truth test set ?????????????????????????????????????????????????????

# Short aliases for the long TB filename
TB_FILE = (
    "evidence-gaps-and-research-needs-identified-during-tuberculosis-"
    "policy-guideline-development500ff7b0a2144c89a9f78f9df0389b6f.pdf"
)

TEST_QUERIES: list[dict] = [
    {
        "id": "Q01",
        "query": "What is the recommended oseltamivir dosage for an adult with confirmed influenza?",
        "correct_source": "9789240097759-eng.pdf",
        "disease_domain": "influenza",
        "key_terms": ["75 mg", "twice daily", "5 days"],
        "category": "dosage_lookup",
    },
    {
        "id": "Q02",
        "query": "What is the standard first-line treatment duration for drug-susceptible tuberculosis?",
        "correct_source": TB_FILE,
        "disease_domain": "tuberculosis",
        "key_terms": ["6-month", "6 month", "isoniazid", "rifampicin"],
        "category": "treatment_protocol",
    },
    {
        "id": "Q03",
        "query": "What corticosteroid is recommended for severe COVID-19 and at what dose?",
        "correct_source": "clinical-management-of-novel-cov.pdf",
        "disease_domain": "covid",
        "key_terms": ["dexamethasone", "6 mg"],
        "category": "treatment_protocol",
    },
    {
        "id": "Q04",
        "query": "At what oxygen saturation level should supplemental oxygen be started in COVID-19?",
        "correct_source": "clinical-management-of-novel-cov.pdf",
        "disease_domain": "covid",
        "key_terms": ["SpO2", "94", "oxygen"],
        "category": "clinical_threshold",
    },
    {
        "id": "Q05",
        "query": "What is the WHO recommended dose of baloxavir marboxil for influenza treatment?",
        "correct_source": "9789240097759-eng.pdf",
        "disease_domain": "influenza",
        "key_terms": ["baloxavir", "40 mg", "80 mg"],
        "category": "dosage_lookup",
    },
    {
        "id": "Q06",
        "query": "What is the oseltamivir dose for post-exposure prophylaxis of seasonal influenza?",
        "correct_source": "9789240097759-eng.pdf",
        "disease_domain": "influenza",
        "key_terms": ["75 mg", "once daily", "10 days"],
        "category": "dosage_lookup",
    },
    {
        "id": "Q07",
        "query": "How should tuberculosis be diagnosed in children according to WHO?",
        "correct_source": TB_FILE,
        "disease_domain": "tuberculosis",
        "key_terms": ["children", "diagnosis", "sputum"],
        "category": "diagnosis",
    },
    {
        "id": "Q08",
        "query": "What antiviral is recommended for non-severe COVID-19 patients at high risk of progression?",
        "correct_source": "policy-brief_covid-19_clinical-management.pdf",
        "disease_domain": "covid",
        "key_terms": ["nirmatrelvir", "molnupiravir", "remdesivir"],
        "category": "treatment_protocol",
    },
    {
        "id": "Q09",
        "query": "How should I manage a patient with chronic cough fever and night sweats?",
        "correct_source": TB_FILE,
        "disease_domain": "tuberculosis",
        "key_terms": ["tuberculosis", "TB", "sputum"],
        "category": "failure_case",  # known D1 failure -- cross-disease contamination
    },
    {
        "id": "Q10",
        "query": "What infection prevention and control measures are required for suspected COVID-19 patients?",
        "correct_source": "clinical-management-of-novel-cov.pdf",
        "disease_domain": "covid",
        "key_terms": ["IPC", "isolation", "mask"],
        "category": "clinical_protocol",
    },
]

# Domain membership map -- used for domain_accuracy_at_k
DOMAIN_FILES: dict[str, list[str]] = {
    "influenza":   ["9789240097759-eng.pdf"],
    "covid":       ["clinical-management-of-novel-cov.pdf",
                    "policy-brief_covid-19_clinical-management.pdf"],
    "tuberculosis": [TB_FILE],
}


# ?? Metric functions ??????????????????????????????????????????????????????????

def source_recall_at_k(results: list, correct_source: str, k: int) -> bool:
    """Return True if correct_source appears in the top-k retrieved chunks."""
    for doc, _ in results[:k]:
        if doc.metadata.get("source", "") == correct_source:
            return True
    return False


def domain_accuracy_at_k(results: list, disease_domain: str, k: int) -> bool:
    """Return True if any chunk from the correct disease domain is in top-k."""
    domain_sources = DOMAIN_FILES.get(disease_domain, [])
    for doc, _ in results[:k]:
        if doc.metadata.get("source", "") in domain_sources:
            return True
    return False


def mean_similarity(results: list, k: int) -> float:
    """Average similarity score of the top-k retrieved chunks."""
    scores = [score for _, score in results[:k]]
    return sum(scores) / len(scores) if scores else 0.0


def hallucination_score(answer: str, context: str, key_terms: list[str]) -> float:
    """
    Grounding score: fraction of expected key_terms found in the retrieved context.
    A score of 1.0 means all expected clinical facts were present in the context
    (low hallucination risk). A score of 0.0 means the answer's key facts have
    no grounding in the retrieved context (high hallucination risk).
    """
    if not key_terms:
        return 1.0
    context_lower = context.lower()
    found = sum(1 for t in key_terms if t.lower() in context_lower)
    return found / len(key_terms)


def evaluate_system(vectorstore, retrieve_fn, label: str = "System") -> dict:
    """
    Run all TEST_QUERIES through retrieve_fn and compute aggregate metrics.

    retrieve_fn(query: str) -> list[(Document, float)]  -- returns (doc, score) pairs

    Returns a results dict with per-query rows and aggregate averages.
    """
    rows = []
    for tq in TEST_QUERIES:
        results = retrieve_fn(tq["query"])
        row = {
            "id":            tq["id"],
            "query":         tq["query"][:55] + "...",
            "category":      tq["category"],
            "recall@1":      source_recall_at_k(results, tq["correct_source"], 1),
            "recall@3":      source_recall_at_k(results, tq["correct_source"], 3),
            "recall@5":      source_recall_at_k(results, tq["correct_source"], 5),
            "domain@3":      domain_accuracy_at_k(results, tq["disease_domain"], 3),
            "sim@1":         results[0][1] if results else 0.0,
            "grounding":     hallucination_score("",
                                 " ".join(d.page_content for d, _ in results[:5]),
                                 tq["key_terms"]),
        }
        rows.append(row)

    n = len(rows)
    summary = {
        "label":       label,
        "rows":        rows,
        "avg_recall@1": sum(r["recall@1"] for r in rows) / n,
        "avg_recall@3": sum(r["recall@3"] for r in rows) / n,
        "avg_recall@5": sum(r["recall@5"] for r in rows) / n,
        "avg_domain@3": sum(r["domain@3"] for r in rows) / n,
        "avg_sim@1":    sum(r["sim@1"]    for r in rows) / n,
        "avg_grounding":sum(r["grounding"] for r in rows) / n,
    }
    return summary


def print_summary(summary: dict):
    """Pretty-print a summary dict to the terminal."""
    print(f"\n{'='*65}")
    print(f"  Results: {summary['label']}")
    print(f"{'='*65}")
    print(f"  {'Query':<40} R@1   R@3   R@5   Dom@3 Gnd")
    print(f"  {'-'*60}")
    for r in summary["rows"]:
        def b(v): return "[OK]" if v else "[X]"
        print(f"  {r['query']:<40} {b(r['recall@1'])}     {b(r['recall@3'])}     "
              f"{b(r['recall@5'])}     {b(r['domain@3'])}     {r['grounding']:.2f}")
    print(f"  {'-'*60}")
    print(f"  {'AVERAGE':<40} "
          f"{summary['avg_recall@1']:.2f}  "
          f"{summary['avg_recall@3']:.2f}  "
          f"{summary['avg_recall@5']:.2f}  "
          f"{summary['avg_domain@3']:.2f}  "
          f"{summary['avg_grounding']:.2f}")
    print(f"{'='*65}\n")
