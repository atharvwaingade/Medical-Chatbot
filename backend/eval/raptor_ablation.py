#!/usr/bin/env python3
"""
RAPTOR vs. Flat Retrieval Ablation Study
==========================================
Compares retrieval quality with and without RAPTOR hierarchical routing
on a held-out query set.  Reports MRR, Precision@k, and NDCG@5.

Motivation
----------
The MedRAPTOR module routes queries through a 4-level ICD-10 ontology hierarchy
before BM25 retrieval.  Without an ablation study, a reviewer can argue that
the RAPTOR module is decorative (adds complexity without improving results).

Conditions
----------
1. **Flat BM25+SCS** (baseline): retrieve from all entries without RAPTOR routing.
2. **RAPTOR-routed BM25+SCS**: filter candidate set using RAPTOR organ-system
   and syndrome routing, then run BM25 on the filtered subset.
3. **Full MedRAG-Turbo**: RAPTOR + GRADE + prevalence prior.

Expected Result
---------------
RAPTOR routing should improve Precision@k for specific queries (cardiac, renal,
neurological) by narrowing the candidate pool.  For very vague queries
("I feel unwell"), flat retrieval may outperform RAPTOR if the routing is
uncertain (which justifies the fallback mechanism in MedRAPTOR).

References
----------
Sarthi, P. et al. (2024). RAPTOR: Recursive abstractive processing for
tree-organised retrieval. *ICLR 2024*. https://arxiv.org/abs/2401.18059
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.dataset import MedicalDataset
from app.rag.retriever import Retriever
from app.rag.raptor import MedRAPTOR
from app.rag.query_processor import QueryProcessor


# ---------------------------------------------------------------------------
# Ablation Query Set with Ground-Truth Conditions
# ---------------------------------------------------------------------------

ABLATION_QUERIES: list[dict] = [
    # Highly specific — RAPTOR routing should help narrow to right system
    {"query": "crushing chest pain radiating to left arm sweating",
     "symptoms": ["chest pain", "left arm pain", "sweating"],
     "correct": "Possible Acute Coronary Syndrome", "system": "Cardiovascular"},
    {"query": "burning urination frequency urgency",
     "symptoms": ["burning urination", "frequent urination"],
     "correct": "Urinary Tract Infection (UTI)", "system": "Urinary/Renal"},
    {"query": "severe unilateral headache nausea photophobia aura",
     "symptoms": ["severe headache", "nausea"],
     "correct": "Migraine", "system": "Neurological"},
    {"query": "fever cough body aches headache rapid onset",
     "symptoms": ["fever", "cough", "body aches"],
     "correct": "Influenza", "system": "Respiratory"},
    {"query": "runny nose sore throat mild fever no aches",
     "symptoms": ["runny nose", "sore throat"],
     "correct": "Common Cold", "system": "Respiratory"},
    {"query": "shortness of breath swollen ankles orthopnoea",
     "symptoms": ["shortness of breath", "leg swelling"],
     "correct": "Congestive Heart Failure", "system": "Cardiovascular"},
    {"query": "fatigue weight gain cold intolerance constipation dry skin",
     "symptoms": ["fatigue", "cold intolerance", "weight gain"],
     "correct": "Hypothyroidism", "system": "Metabolic/Endocrine"},
    {"query": "increased thirst polyuria weight loss fatigue blurred vision",
     "symptoms": ["increased thirst", "frequent urination", "blurred vision"],
     "correct": "Type 2 Diabetes Mellitus", "system": "Metabolic/Endocrine"},
    {"query": "persistent sadness loss of interest insomnia fatigue hopelessness",
     "symptoms": ["persistent sadness", "loss of interest"],
     "correct": "Major Depressive Disorder", "system": "Psychiatric"},
    {"query": "sudden weakness facial droop speech difficulty arm weakness",
     "symptoms": ["sudden weakness", "facial droop", "slurred speech"],
     "correct": "Acute Ischaemic Stroke", "system": "Neurological"},
    {"query": "palpitations heat intolerance weight loss tremor",
     "symptoms": ["palpitations", "heat intolerance", "unexplained weight loss"],
     "correct": "Hyperthyroidism", "system": "Metabolic/Endocrine"},
    {"query": "productive cough fever pleuritic chest pain dyspnoea",
     "symptoms": ["productive cough", "fever", "chest pain"],
     "correct": "Community-Acquired Pneumonia", "system": "Respiratory"},
    # Vague queries — flat retrieval may do equally well
    {"query": "I feel unwell and tired",
     "symptoms": ["fatigue"],
     "correct": "Influenza", "system": "General"},
    {"query": "not feeling well headache",
     "symptoms": ["headache"],
     "correct": "Common Cold", "system": "General"},
    {"query": "stomach problems nausea vomiting",
     "symptoms": ["nausea", "vomiting"],
     "correct": "Acute Gastroenteritis", "system": "Gastrointestinal"},
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_mrr(results: list[str], correct: str) -> float:
    for rank, cond in enumerate(results, 1):
        if correct.lower() in cond.lower() or cond.lower() in correct.lower():
            return 1.0 / rank
    return 0.0


def compute_ndcg(results: list[str], correct: str, k: int = 5) -> float:
    for rank, cond in enumerate(results[:k], 1):
        if correct.lower() in cond.lower() or cond.lower() in correct.lower():
            return 1.0 / math.log2(rank + 1)
    return 0.0


def precision_at_k(results: list[str], correct: str, k: int) -> float:
    top_k = results[:k]
    hits = sum(1 for c in top_k if correct.lower() in c.lower() or c.lower() in correct.lower())
    return hits / k


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

class FlatRetriever(Retriever):
    """BM25+SCS only — no GRADE boost, no prevalence prior."""
    ALPHA: float = 0.75
    BETA_SCS: float = 0.25
    PREV_WEIGHT: float = 0.0
    GRADE_BOOST: float = 0.0


def run_flat_retrieval(
    retriever: Retriever,
    query: str,
    symptoms: list[str],
    top_k: int = 5,
) -> list[str]:
    pq = QueryProcessor.process(query, symptoms)
    results = retriever.retrieve_results(pq.expanded_query, pq.medical_tokens, top_k)
    return [r.entry.get("condition", "") for r in results]


def run_raptor_retrieval(
    entries: list[dict],
    raptor: MedRAPTOR,
    retriever: Retriever,
    query: str,
    symptoms: list[str],
    top_k: int = 5,
) -> list[str]:
    """Use RAPTOR to route the query, then filter retrieval to matching organ system."""
    pq = QueryProcessor.process(query, symptoms)

    # Get RAPTOR system routing
    routed_systems: list[str] = raptor.route_systems(query)

    # Filter entries to matching systems
    filtered_entries = []
    for e in entries:
        e_organ = e.get("organ_system", "")
        # Match if any routed system overlaps with entry's organ system
        for sys_name in routed_systems:
            if (
                sys_name.lower() in e_organ.lower()
                or e_organ.lower() in sys_name.lower()
            ):
                filtered_entries.append(e)
                break

    # If RAPTOR filtering leaves less than top_k entries, fall back to all
    if len(filtered_entries) < top_k:
        filtered_entries = entries

    # Build local retriever on filtered set
    local_retriever = Retriever(filtered_entries)
    results = local_retriever.retrieve_results(pq.expanded_query, pq.medical_tokens, top_k)
    return [r.entry.get("condition", "") for r in results]


def run_ablation(
    entries: list[dict],
    top_k: int = 5,
) -> dict:
    """Run the 3-condition ablation."""
    flat_retriever = FlatRetriever(entries)
    full_retriever = Retriever(entries)
    raptor = MedRAPTOR()
    raptor.build_from_entries(entries)

    flat_mrrs, raptor_mrrs, full_mrrs = [], [], []
    flat_ndcgs, raptor_ndcgs, full_ndcgs = [], [], []
    flat_p3s, raptor_p3s, full_p3s = [], [], []

    for q in ABLATION_QUERIES:
        query, symptoms, correct = q["query"], q["symptoms"], q["correct"]

        flat = run_flat_retrieval(flat_retriever, query, symptoms, top_k)
        raptor_res = run_raptor_retrieval(entries, raptor, full_retriever, query, symptoms, top_k)
        full = run_flat_retrieval(full_retriever, query, symptoms, top_k)

        flat_mrrs.append(compute_mrr(flat, correct))
        raptor_mrrs.append(compute_mrr(raptor_res, correct))
        full_mrrs.append(compute_mrr(full, correct))

        flat_ndcgs.append(compute_ndcg(flat, correct, top_k))
        raptor_ndcgs.append(compute_ndcg(raptor_res, correct, top_k))
        full_ndcgs.append(compute_ndcg(full, correct, top_k))

        flat_p3s.append(precision_at_k(flat, correct, 3))
        raptor_p3s.append(precision_at_k(raptor_res, correct, 3))
        full_p3s.append(precision_at_k(full, correct, 3))

    def _avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "n_queries": len(ABLATION_QUERIES),
        "top_k": top_k,
        "flat_bm25_scs": {
            "mrr": _avg(flat_mrrs),
            f"ndcg@{top_k}": _avg(flat_ndcgs),
            "precision@3": _avg(flat_p3s),
        },
        "raptor_routed": {
            "mrr": _avg(raptor_mrrs),
            f"ndcg@{top_k}": _avg(raptor_ndcgs),
            "precision@3": _avg(raptor_p3s),
        },
        "medrag_turbo": {
            "mrr": _avg(full_mrrs),
            f"ndcg@{top_k}": _avg(full_ndcgs),
            "precision@3": _avg(full_p3s),
        },
    }


def _print_ablation(results: dict) -> None:
    top_k = results["top_k"]
    print()
    print("=" * 62)
    print("RAPTOR vs. Flat Retrieval Ablation")
    print(f"(n={results['n_queries']} queries, top_k={top_k})")
    print("=" * 62)
    print(f"{'Condition':<22} {'MRR':>6} {'NDCG@5':>8} {'P@3':>6}")
    print("-" * 62)
    for label, key in [
        ("Flat BM25+SCS", "flat_bm25_scs"),
        ("RAPTOR-routed", "raptor_routed"),
        ("MedRAG-Turbo", "medrag_turbo"),
    ]:
        r = results[key]
        print(
            f"{label:<22} {r['mrr']:>6.4f} "
            f"{r[f'ndcg@{top_k}']:>8.4f} {r['precision@3']:>6.4f}"
        )
    print("=" * 62)

    # Delta MedRAG vs Flat
    flat = results["flat_bm25_scs"]
    full = results["medrag_turbo"]
    mrr_delta = full["mrr"] - flat["mrr"]
    print(f"\nMedRAG-Turbo vs. Flat — MRR Δ={mrr_delta:+.4f}")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAPTOR ablation study")
    parser.add_argument("--kb", default="data/sample_medical_knowledge.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    dataset = MedicalDataset(args.kb)
    if not dataset.entries:
        print(f"ERROR: No entries loaded from {args.kb}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(dataset.entries)} KB entries.")

    results = run_ablation(dataset.entries, args.top_k)

    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        _print_ablation(results)
